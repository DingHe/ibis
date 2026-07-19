"""Lower the ibis expression graph to a SQL-like relational algebra."""

from __future__ import annotations

import operator
import sys
from collections.abc import Mapping
from functools import reduce
from typing import TYPE_CHECKING, Any

import toolz
from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict  # noqa: TC001
from ibis.common.deferred import var
from ibis.common.graph import Graph
from ibis.common.patterns import InstanceOf, Object, Pattern, replace
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.rewrites import d, p, replace_parameter
from ibis.expr.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Sequence

x = var("x")
y = var("y")


@public
class CTE(ops.Relation):
    """Common table expression."""

    parent: ops.Relation

    @attribute
    def schema(self):
        return self.parent.schema

    @attribute
    def values(self):
        return self.parent.values


@public
class Select(ops.Relation):
    """Relation modelled after SQL's SELECT statement."""

    parent: ops.Relation
    selections: FrozenDict[str, ops.Value] = {}
    predicates: VarTuple[ops.Value[dt.Boolean]] = ()
    qualified: VarTuple[ops.Value[dt.Boolean]] = ()
    sort_keys: VarTuple[ops.SortKey] = ()
    distinct: bool = False

    def is_star_selection(self):
        return tuple(self.values.items()) == tuple(self.parent.fields.items())

    @attribute
    def values(self):
        return self.selections

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.selections.items()})


@public
class FirstValue(ops.Analytic):
    """Retrieve the first element."""

    arg: ops.Column[dt.Any]

    @attribute
    def dtype(self):
        return self.arg.dtype


@public
class LastValue(ops.Analytic):
    """Retrieve the last element."""

    arg: ops.Column[dt.Any]

    @attribute
    def dtype(self):
        return self.arg.dtype


# TODO(kszucs): there is a better strategy to rewrite the relational operations
# to Select nodes by wrapping the leaf nodes in a Select node and then merging
# Project, Filter, Sort, etc. incrementally into the Select node. This way we
# can have tighter control over simplification logic.


@replace(p.Project)
def project_to_select(_, **kwargs):
    """Convert a Project node to a Select node."""
    return Select(_.parent, selections=_.values)


def partition_predicates(predicates):
    qualified = []
    unqualified = []

    for predicate in predicates:
        if predicate.find(ops.WindowFunction, filter=ops.Value):
            qualified.append(predicate)
        else:
            unqualified.append(predicate)

    return unqualified, qualified


@replace(p.Filter)
def filter_to_select(_, **kwargs):
    """Convert a Filter node to a Select node."""
    predicates, qualified = partition_predicates(_.predicates)
    return Select(
        _.parent, selections=_.values, predicates=predicates, qualified=qualified
    )


@replace(p.Sort)
def sort_to_select(_, **kwargs):
    """Convert a Sort node to a Select node."""
    return Select(_.parent, selections=_.values, sort_keys=_.keys)


@replace(p.Distinct)
def distinct_to_select(_, **kwargs):
    """Convert a Distinct node to a Select node."""
    return Select(_.parent, selections=_.values, distinct=True)


@replace(p.DropColumns)
def drop_columns_to_select(_, **kwargs):
    """Convert a DropColumns node to a Select node."""
    # if we're dropping fewer than 50% of the parent table's columns then the
    # compiled query will likely be smaller than if we list everything *NOT*
    # being dropped
    if len(_.columns_to_drop) < len(_.schema) // 2:
        return _
    return Select(_.parent, selections=_.values)


@replace(p.FillNull)
def fill_null_to_select(_, **kwargs):
    """Rewrite FillNull to a Select node."""
    if isinstance(_.replacements, Mapping):
        mapping = _.replacements
    else:
        mapping = {
            name: _.replacements
            for name, type in _.parent.schema.items()
            if type.nullable
        }

    if not mapping:
        return _.parent

    selections = {}
    for name in _.parent.schema.names:
        col = ops.Field(_.parent, name)
        if (value := mapping.get(name)) is not None:
            col = ops.Coalesce((col, value))
        selections[name] = col

    return Select(_.parent, selections=selections)


@replace(p.DropNull)
def drop_null_to_select(_, **kwargs):
    """Rewrite DropNull to a Select node."""
    if _.subset is None:
        columns = [ops.Field(_.parent, name) for name in _.parent.schema.names]
    else:
        columns = _.subset

    if columns:
        preds = [
            reduce(
                ops.And if _.how == "any" else ops.Or,
                [ops.NotNull(c) for c in columns],
            )
        ]
    elif _.how == "all":
        preds = [ops.Literal(False, dtype=dt.bool)]
    else:
        return _.parent

    return Select(_.parent, selections=_.values, predicates=tuple(preds))


@replace(p.WindowFunction(p.First | p.Last))
def first_to_firstvalue(_, **kwargs):
    """Convert a First or Last node to a FirstValue or LastValue node."""
    if _.func.where is not None:
        raise com.UnsupportedOperationError(
            f"`{type(_.func).__name__.lower()}` with `where` is unsupported "
            "in a window function"
        )
    klass = FirstValue if isinstance(_.func, ops.First) else LastValue
    return _.copy(func=klass(_.func.arg))


@replace(p.Alias)
def remove_aliases(_, **kwargs):
    """Remove all remaining aliases, they're not needed for remaining compilation."""
    return _.arg


def complexity(node):
    """Assign a complexity score to a node.

    Subsequent projections can be merged into a single projection by replacing
    the fields referenced in the outer projection with the computed expressions
    from the inner projection. This inlining can result in very complex value
    expressions depending on the projections. In order to prevent excessive
    inlining, we assign a complexity score to each node.

    The complexity score assigns 1 to each value expression and adds up in the
    tree hierarchy unless there is a Field node where we don't add up the
    complexity of the referenced relation. This way we treat fields kind of like
    reusable variables considering them less complex than they were inlined.
    """

    def accum(node, *args):
        if isinstance(node, ops.Field):
            return 1
        elif isinstance(node, ops.Impure):
            # consider (potentially) impure functions maximally complex
            return sys.maxsize
        else:
            return 1 + sum(args)

    return node.map_nodes(accum)[node]


@replace(Object(Select, Object(Select)))
def merge_select_select(_, **kwargs):
    """Merge subsequent Select relations into one.

    This rewrites eliminates `_.parent` by merging the outer and the inner
    `predicates`, `sort_keys` and keeping the outer `selections`. All selections
    from the inner Select are inlined into the outer Select.
    """
    # don't merge if either the outer or the inner select has window functions
    blocking = (
        ops.WindowFunction,
        ops.ExistsSubquery,
        ops.InSubquery,
        ops.Unnest,
        ops.Impure,
    )
    if _.find_below(blocking, filter=ops.Value):
        return _
    if _.parent.find_below(blocking, filter=ops.Value):
        return _

    if _.parent.distinct:
        # The inner query is distinct.
        #
        # If the outer query is distinct, it's only safe to merge if it's a simple subselection:
        # - Fusing in the presence of non-deterministic calls in the select would lead to
        #   incorrect results
        # - Fusing in the presence of expensive calls in the select would lead to potential
        #   performance pitfalls
        if _.distinct and not all(
            isinstance(v, ops.Field) for v in _.selections.values()
        ):
            return _

        # If the outer query isn't distinct, it's only safe to merge if the outer is a SELECT *:
        # - If new columns are added, they might be non-distinct, changing the distinctness
        # - If previous columns are removed, that would also change the distinctness
        if not _.distinct and not _.is_star_selection():
            return _

        distinct = True
    elif _.distinct:
        # The outer query is distinct and the inner isn't. It's only safe to merge if either
        # - The inner query isn't ordered
        # - The outer query is a SELECT *
        #
        # Otherwise we run the risk that the outer query drops columns needed for the ordering of
        # the inner query - many backends don't allow select distinc queries to order by columns
        # that aren't present in their selection, like
        #
        #   SELECT DISTINCT a, b FROM t ORDER BY c  --- some backends will explode at this
        #
        # An alternate solution would be to drop the inner ORDER BY clause, since the backend will
        # ignore it anyway since it's a subquery. That feels potentially risky though, better
        # to generate the SQL as written.
        if _.parent.sort_keys and not _.is_star_selection():
            return _

        distinct = True
    else:
        # Neither query is distinct, safe to merge
        distinct = False

    subs = {ops.Field(_.parent, k): v for k, v in _.parent.values.items()}
    selections = {k: v.replace(subs, filter=ops.Value) for k, v in _.selections.items()}

    predicates = tuple(p.replace(subs, filter=ops.Value) for p in _.predicates)
    unique_predicates = toolz.unique(_.parent.predicates + predicates)

    qualified = tuple(p.replace(subs, filter=ops.Value) for p in _.qualified)
    unique_qualified = toolz.unique(_.parent.qualified + qualified)

    sort_keys = tuple(s.replace(subs, filter=ops.Value) for s in _.sort_keys)
    sort_key_args = {s.arg for s in sort_keys}
    parent_sort_keys = tuple(
        k for k in _.parent.sort_keys if k.arg not in sort_key_args
    )
    unique_sort_keys = sort_keys + parent_sort_keys

    result = Select(
        _.parent.parent,
        selections=selections,
        predicates=unique_predicates,
        qualified=unique_qualified,
        sort_keys=tuple(
            key for key in unique_sort_keys if not isinstance(key.arg, ops.Literal)
        ),
        distinct=distinct,
    )
    return result if complexity(result) <= complexity(_) else _


def extract_ctes(node: ops.Relation) -> set[ops.Relation]:
    cte_types = (Select, ops.Aggregate, ops.JoinChain, ops.Set, ops.Limit, ops.Sample)
    dont_count = (ops.Field, ops.CountStar, ops.CountDistinctStar)

    g = Graph.from_bfs(node, filter=~InstanceOf(dont_count))
    result = set()
    for op, dependents in g.invert().items():
        if isinstance(op, ops.AliasedRelation) or (
            len(dependents) > 1 and isinstance(op, cte_types)
        ):
            result.add(op)

    return result

# 把 ibis 的表达式树"降级"(lower)成一个更接近 SQL 关系代数(relational algebra)的形式——也就是把 ibis 里那些比较高层、抽象的算子概念(比如 filter、project、sort 单独存在)转换成 SQL 里 select 语句真正拥有的结构(即一个 select 节点里同时带有投影、过滤、排序等属性),
# 并识别出需要被提升为 CTE 的公共子表达式
# node: ops.Node:待处理的 ibis 表达式树的根节点(通常是一个关系节点 Relation)。
# rewrites: Sequence[Pattern] = ():在"SQL 特定转换"之前要应用的一组补充重写规则(通常是某个具体 SQL 方言/后端自定义的重写,比如把某个 ibis 算子转换成该方言支持的等价形式)。
# post_rewrites: Sequence[Pattern] = ():在"SQL 特定转换"之后要应用的一组补充重写规则(用于收尾处理)。
# fuse_selects: bool = True:是否要把连续多个 Select 节点合并/融合成一个(减少不必要的嵌套子查询,让生成的 SQL 更精简)。
def sqlize(
    node: ops.Node,
    params: Mapping[ops.ScalarParameter, Any],
    rewrites: Sequence[Pattern] = (),
    post_rewrites: Sequence[Pattern] = (),
    fuse_selects: bool = True,
) -> tuple[ops.Node, list[ops.Node]]:
    """Lower the ibis expression graph to a SQL-like relational algebra.

    Parameters
    ----------
    node
        The root node of the expression graph.
    params
        A mapping of scalar parameters to their values.
    rewrites
        Supplementary rewrites to apply before SQL-specific transforms.
    post_rewrites
        Supplementary rewrites to apply after SQL-specific transforms.
    fuse_selects
        Whether to merge subsequent Select nodes into one where possible.

    Returns
    -------
    Tuple of the rewritten expression graph and a list of CTEs.

    """
    # 传入的根节点必须是一个"关系"类型的节点(比如 select、join、表引用等),因为整个函数的目的就是处理关系代数层面的转换,如果传入的不是 Relation(比如传入了一个标量表达式),说明调用方用法有误。
    assert isinstance(node, ops.Relation)

    # apply the backend specific rewrites
    # 如果调用方传入了非空的 rewrites 序列(某方言特有的重写规则),先把这些规则依次用 |(operator.or_)合并成一个统一的规则集合(在 ibis 的 pattern-matching 系统里,| 表示"规则组合",即匹配到任意一条就应用对应替换),
    # 然后调用 node.replace(...) 对整棵表达式树做一次全局重写。这一步是"方言特定的前置处理",在真正做 SQL 化转换之前先处理掉那些该方言需要特殊对待的算子。
    if rewrites:
        node = node.replace(reduce(operator.or_, rewrites))

    # lower the expression graph to a SQL-like relational algebra
    # 核心的"SQL 化"步骤,把一组固定的、内置的重写规则通过 | 全部合并成一个规则集合,对表达式树做一次统一的替换遍历。每条规则的作用(从命名可以推断):
    context = {"params": params}
    result = node.replace(
        # 表达式树中的 ScalarParameter(标量参数占位符)节点,依据 context["params"] 替换成具体的字面量值。
        replace_parameter
        # 去除多余的/冗余的别名包装节点,简化表达式树结构。
        | remove_aliases
        # 把 ibis 的 Project(投影,选择列)算子转换成 SQL 的 Select 节点形式。
        | project_to_select
        # 把 Filter(过滤,即 WHERE 条件)算子转换/合并进 Select 节点。
        | filter_to_select
        # 把 Sort(排序,即 ORDER BY)算子转换/合并进 Select 节点。
        | sort_to_select
        # 把 Distinct 算子转换成 Select 里的 DISTINCT 属性。
        | distinct_to_select
        # 把"填充空值"(类似 fillna)的高层算子转换成用 Select + CASE WHEN(或 COALESCE)之类的形式表达。
        | fill_null_to_select
        # 把"丢弃空值行"(类似 dropna)的算子转换成带有相应过滤条件的 Select。
        | drop_null_to_select
        # 把"丢弃某些列"的算子转换成对应投影列表的 Select。
        | drop_columns_to_select
        # 把某种"取第一个值"的聚合/窗口算子(First)转换成 SQL 标准或方言认可的 FIRST_VALUE 窗口函数形式。
        | first_to_firstvalue,
        context=context,
    )

    # squash subsequent Select nodes into one
    # 如果 fuse_selects 为真(默认是),再应用一次 merge_select_select 规则,把表达式树中"连续嵌套的两个 Select 节点"尽可能合并成一个 Select(比如 SELECT a FROM (SELECT a, b FROM t WHERE ...)
    # 这种可以合并简化的嵌套结构,合并后减少不必要的子查询嵌套层级,让最终 SQL 更简洁、执行效率也可能更好)。
    if fuse_selects:
        result = result.replace(merge_select_select)
    # 如果调用方传入了非空的 post_rewrites,同样先用 | 合并成一个规则集合,再对 result 做一次替换。这是"SQL 化之后"的收尾重写,给各方言一个机会在标准转换流程跑完之后,再做一些额外的、针对性的调整。
    if post_rewrites:
        result = result.replace(reduce(operator.or_, post_rewrites))

    # extract common table expressions while wrapping them in a CTE node
    # 调用 extract_ctes 函数,分析当前的表达式树,识别出哪些子表达式(子树)应该被提升成 CTE(即需要用 WITH name AS (...) 单独定义、
    # 然后被多处引用的公共子查询——典型场景是同一个子查询被多次复用,或者显式要求生成 CTE 形式)。返回一个需要被视为 CTE 的节点集合/列表。
    ctes = extract_ctes(result)

    if ctes:
        # 如果这个节点有需要更新的子节点(kwargs 里包含了子节点重写后的新值),
        # 就用 __recreate__ 重新构造一个新的、内容更新过的同类型节点;否则保持 node 不变。
        def apply_ctes(node, kwargs):
            new = node.__recreate__(kwargs) if kwargs else node
            return CTE(new) if node in ctes else new

        result = result.replace(apply_ctes)
        return result, [cte.parent for cte in result.find(CTE, ordered=True)]
    return result, []


# supplemental rewrites selectively used on a per-backend basis


@replace(Select)
def split_select_distinct_with_order_by(_):
    """Split a `SELECT DISTINCT ... ORDER BY` query when needed.

    Some databases (postgres, pyspark, ...) have issues with two types of
    ordered select distinct statements:

    ```
    --- ORDER BY with an expression instead of a name in the select list
    SELECT DISTINCT a, b FROM t ORDER BY a + 1

    --- ORDER BY using a qualified column name, rather than the alias in the select list
    SELECT DISTINCT a, b as x FROM t ORDER BY b  --- or t.b
    ```

    We solve both these cases by splitting everything except the `ORDER BY`
    into a subquery.

    ```
    SELECT DISTINCT a, b FROM t WHERE a > 10 ORDER BY a + 1
    --- is rewritten as ->
    SELECT * FROM (SELECT DISTINCT a, b FROM t WHERE a > 10) ORDER BY a + 1
    ```
    """
    # risingwave and pyspark also don't allow qualified names as sort keys, like
    #   SELECT DISTINCT t.a FROM t ORDER BY t.a
    # To avoid having specific rewrite rules for these backends to use only
    # local names, we always split SELECT DISTINCT from ORDER BY here. Otherwise we
    # could also avoid splitting if all sort keys appear in the select list.
    if _.distinct and _.sort_keys:
        # select every visible field across all properties from the current
        # query (e.g., include sort_keys and not just selections), in case the
        # that query's selections don't include fields used in sort keys
        #
        # 1. start with all fields
        additional_fields = {
            field.name: field for field in _.find_below(ops.Field, filter=ops.Value)
        }
        # 2. then find any fields that are part of the current selection set,
        # either as a more complex expression or as a simple field reference
        # 3. remove the fields that are already present
        # 4. what remains are the fields that must be added to the select set
        # to be a valid query for any backend opting into this rewrite
        for selection in _.selections.values():
            for field in selection.find_below(ops.Field, filter=ops.Value):
                del additional_fields[field.name]

        inner = _.copy(selections=_.selections | additional_fields, sort_keys=())
        subs = {v: ops.Field(inner, k) for k, v in inner.values.items()}
        sort_keys = tuple(s.replace(subs, filter=ops.Value) for s in _.sort_keys)
        selections = {
            k: v.replace(subs, filter=ops.Value) for k, v in _.selections.items()
        }
        return Select(inner, selections=selections, sort_keys=sort_keys)
    return _


@replace(p.WindowFunction(func=p.NTile(y), order_by=()))
def add_order_by_to_empty_ranking_window_functions(_, **kwargs):
    """Add an ORDER BY clause to rank window functions that don't have one."""
    return _.copy(order_by=(y,))


"""Replace checks against an empty right side with `False`."""
empty_in_values_right_side = p.InValues(options=()) >> d.Literal(False, dtype=dt.bool)


@replace(
    p.WindowFunction(p.RankBase | p.NTile)
    | p.StringFind
    | p.FindInSet
    | p.ArrayPosition
)
def one_to_zero_index(_, **kwargs):
    """Subtract one from one-index functions."""
    return ops.Subtract(_, 1)


@replace(ops.NthValue)
def add_one_to_nth_value_input(_, **kwargs):
    if isinstance(_.nth, ops.Literal):
        nth = ops.Literal(_.nth.value + 1, dtype=_.nth.dtype)
    else:
        nth = ops.Add(_.nth, 1)
    return _.copy(nth=nth)


@replace(p.WindowFunction(order_by=()))
def rewrite_empty_order_by_window(_, **kwargs):
    return _.copy(order_by=(ops.NULL,))


@replace(p.WindowFunction(p.RowNumber | p.NTile))
def exclude_unsupported_window_frame_from_row_number(_, **kwargs):
    return ops.Subtract(_.copy(start=None, end=0), 1)


@replace(p.WindowFunction(p.MinRank | p.DenseRank, start=None))
def exclude_unsupported_window_frame_from_rank(_, **kwargs):
    return ops.Subtract(
        _.copy(start=None, end=0, order_by=_.order_by or (ops.NULL,)), 1
    )


@replace(
    p.WindowFunction(
        p.Lag | p.Lead | p.PercentRank | p.CumeDist | p.Any | p.All, start=None
    )
)
def exclude_unsupported_window_frame_from_ops(_, **kwargs):
    return _.copy(start=None, end=0, order_by=_.order_by or (ops.NULL,))


# Rewrite rules for lowering a high-level operation into one composed of more
# primitive operations.


@replace(p.Log2)
def lower_log2(_, **kwargs):
    """Rewrite `log2` as `log`."""
    return ops.Log(_.arg, base=2)


@replace(p.Log10)
def lower_log10(_, **kwargs):
    """Rewrite `log10` as `log`."""
    return ops.Log(_.arg, base=10)


@replace(p.Bucket)
def lower_bucket(_, **kwargs):
    """Rewrite `Bucket` as `SearchedCase`."""
    cases = []
    results = []

    if _.closed == "left":
        l_cmp = ops.LessEqual
        r_cmp = ops.Less
    else:
        l_cmp = ops.Less
        r_cmp = ops.LessEqual

    user_num_buckets = len(_.buckets) - 1

    bucket_id = 0
    if _.include_under:
        if user_num_buckets > 0:
            cmp = ops.Less if _.close_extreme else r_cmp
        else:
            cmp = ops.LessEqual if _.closed == "right" else ops.Less
        cases.append(cmp(_.arg, _.buckets[0]))
        results.append(bucket_id)
        bucket_id += 1

    for j, (lower, upper) in enumerate(zip(_.buckets, _.buckets[1:])):
        if _.close_extreme and (
            (_.closed == "right" and j == 0)
            or (_.closed == "left" and j == (user_num_buckets - 1))
        ):
            cases.append(
                ops.And(ops.LessEqual(lower, _.arg), ops.LessEqual(_.arg, upper))
            )
            results.append(bucket_id)
        else:
            cases.append(ops.And(l_cmp(lower, _.arg), r_cmp(_.arg, upper)))
            results.append(bucket_id)
        bucket_id += 1

    if _.include_over:
        if user_num_buckets > 0:
            cmp = ops.Less if _.close_extreme else l_cmp
        else:
            cmp = ops.Less if _.closed == "right" else ops.LessEqual

        cases.append(cmp(_.buckets[-1], _.arg))
        results.append(bucket_id)
        bucket_id += 1

    return ops.SearchedCase(
        cases=tuple(cases), results=tuple(results), default=ops.NULL
    )


@replace(p.Capitalize)
def lower_capitalize(_, **kwargs):
    """Rewrite Capitalize in terms of substring, concat, upper, and lower."""
    first = ops.Uppercase(ops.Substring(_.arg, start=0, length=1))
    # use length instead of length - 1 to avoid backends complaining about
    # asking for negative length
    #
    # there are at most length - 1 characters, so asking for length is fine
    rest = ops.Lowercase(ops.Substring(_.arg, start=1, length=ops.StringLength(_.arg)))
    return ops.StringConcat((first, rest))


def lower_sample(
    supported_methods=("row", "block"),
    supports_seed=True,
    physical_tables_only=False,
):
    """Create a rewrite rule for lowering Sample.

    If the `Sample` operation matches the specified criteria, it will compile
    to the backend's `TABLESAMPLE` operation, otherwise it will fallback to
    `t.filter(ibis.random() <= fraction)`.

    Parameters
    ----------
    supported_methods
        The sampling methods supported by the backend's native TABLESAMPLE operation.
    supports_seed
        Whether the backend's native TABLESAMPLE supports setting a `seed`.
    physical_tables_only
        If true, only sampling on physical tables will compile to a `TABLESAMPLE`.
    """

    @replace(p.Sample)
    def lower(_, **kwargs):
        if (
            _.method not in supported_methods
            or (_.seed is not None and not supports_seed)
            or (
                physical_tables_only
                and not isinstance(_.parent, (ops.DatabaseTable, ops.UnboundTable))
            )
        ):
            # TABLESAMPLE not supported in this context, lower to `t.filter(random() <= fraction)`
            if _.seed is not None:
                raise com.UnsupportedOperationError(
                    "`Table.sample` with a random seed is unsupported for this backend"
                )
            return ops.Filter(
                _.parent, (ops.LessEqual(ops.RandomScalar(), _.fraction),)
            )
        return _

    return lower


@replace(p.ArrayMap | p.ArrayFilter)
def subtract_one_from_array_map_filter_index(_, **kwargs):
    # no index argument, so do nothing
    if _.index is None:
        return _

    @replace(y @ p.Argument(name=_.index.name))
    def argument_replacer(_, y, **kwargs):
        return ops.Subtract(y, 1)

    return _.copy(body=_.body.replace(argument_replacer))
