"""Some common rewrite functions to be shared between backends."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import toolz
from typing_extensions import Self

import ibis.expr.operations as ops
from ibis.common.collections import FrozenDict  # noqa: TC001
from ibis.common.deferred import Item, _, deferred, var
from ibis.common.exceptions import ExpressionError, IbisInputError
from ibis.common.graph import Node as Traversable
from ibis.common.graph import traverse
from ibis.common.grounds import Annotable
from ibis.common.patterns import Check, pattern, replace
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.util import Namespace, promote_list

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import ibis.expr.types as ir

p = Namespace(pattern, module=ops)
d = Namespace(deferred, module=ops)


x = var("x")
y = var("y")
name = var("name")

# DerefMap 是一个核心工具类，主要用于关系代数表达式的“解引用”（Dereferencing）。它是 Ibis 能够实现优雅、链式 API 的关键基础设施
# 在 Ibis 的关系表达式（IR）中，每个操作通常只能引用其“直接父节点”的字段。然而，为了用户体验，Ibis 允许用户在链式操作中引用更早期的关系（例如：t2.filter(t.a > 0) 中，用户引用了最原始的表 t，而不是 t2 的直接来源 t1）
# 核心职责是：
# 追踪溯源：在关系层级中向上遍历，找到字段的原始定义
# 重写表达式：将用户在表达式中使用的“跨层级”字段引用，自动替换为当前关系能够直接识别的字段引用。
# 处理歧义：如果同一个字段名在多个父关系中存在，它会检测并报错，防止逻辑混淆。
class DerefMap(Annotable, Traversable):
    """Trace and replace fields from earlier relations in the hierarchy.

    In order to provide a nice user experience, we need to allow expressions
    from earlier relations in the hierarchy. Consider the following example:

    t = ibis.table([('a', 'int64'), ('b', 'string')], name='t')
    t1 = t.select([t.a, t.b])
    t2 = t1.filter(t.a > 0)  # note that not t1.a is referenced here
    t3 = t2.select(t.a)  # note that not t2.a is referenced here

    However the relational operations in the IR are strictly enforcing that
    the expressions are referencing the immediate parent only. So we need to
    track fields upwards the hierarchy to replace `t.a` with `t1.a` and `t2.a`
    in the example above. This is called dereferencing.

    Whether we can treat or not a field of a relation semantically equivalent
    with a field of an earlier relation in the hierarchy depends on the
    `.values` mapping of the relation. Leaf relations, like `t` in the example
    above, have an empty `.values` mapping, so we cannot dereference fields
    from them. On the other hand a projection, like `t1` in the example above,
    has a `.values` mapping like `{'a': t.a, 'b': t.b}`, so we can deduce that
    `t1.a` is semantically equivalent with `t.a` and so on.
    """

    """The relations we want the values to point to."""
    # 存储当前解引用操作的目标关系集合。
    # 所有待处理的字段最终都会被尝试映射为指向这些关系（或其子集）的引用。
    rels: frozenset[ops.Relation]

    """Extra substitutions to be added to the dereference map. Stored on the
    instance to facilitate lazy dereferencing."""
    # 允许用户手动注入额外的替换规则。常用于复杂的子查询或特定优化场景，实现“懒加载”式的替换。
    extra: Optional[FrozenDict[ops.Node, ops.Node]]

    """Substitution mapping from values of earlier relations to the fields of `rels`."""
    # 核心映射表。缓存了“早期字段 → 当前可直接引用的字段”的映射关系。
    # 该属性在首次调用 dereference 时通过 _fill_substitution_mappings 延迟计算填充。
    subs: Optional[FrozenDict[ops.Value, ops.Field]] = None

    """Ambiguous field references."""
    # 记录歧义字段。如果一个表达式在多个父级中都能找到对应，且无法确定唯一来源，则记录在此，在解引用时触发 IbisInputError。
    ambigs: Optional[FrozenDict[ops.Value, VarTuple[ops.Value]]] = None
    # 将输入的单个或多个关系包装为 frozenset，确保了内部处理的一致性，是创建该类的标准入口。
    @classmethod
    def from_targets(
        cls, rels, extra: Mapping[ops.Node, ops.Node] | None = None
    ) -> Self:
        """Create a dereference map from a list of target relations.

        Usually a single relation is passed except for joins where multiple
        relations are involved.

        Parameters
        ----------
        rels
            The target relations to dereference to.
        extra
            Extra substitutions to be added to the dereference map.

        Returns
        -------
        DerefMap
        """
        return cls(rels=frozenset(promote_list(rels)), extra=extra)
    # 实现血缘追踪（Lineage Tracing）的核心逻辑。
    # 作用是：沿着表达式树向上“爬”，找到一个字段在当前关系层级中的原始定义来源。
    @classmethod
    def backtrack(cls, value) -> Iterator[tuple[ops.Field, int]]:
        """Backtrack the field in the relation hierarchy.

        The field is traced back until no modification is made, so only follow
        ops.Field nodes not arbitrary values.

        Parameters
        ----------
        value : ops.Value
            The value to backtrack.

        Yields
        ------
        tuple[ops.Field, int]
            The value node and the distance from the original value.
        """
        # distance 用于记录“辈分”。
        # 原始字段距离为 0，每向上回溯一层（通过 rel.values 查找），距离加 1。
        distance = 0
        # track down the field in the hierarchy until no modification
        # is made so only follow ops.Field nodes not arbitrary values;
        # 只追踪“字段引用”节点
        # 如果当前节点不再是字段（例如变成了计算表达式 a + 1），循环就会停止
        while isinstance(value, ops.Field):
            yield value, distance
            # 从当前字段的父关系（rel）中，根据字段名重新获取该字段在上一层定义的值（values 映射表）。这实现了跨关系层级的向上搜索。
            value = value.rel.values.get(value.name)
            distance += 1
        if (
            value is not None # 确保不是空值
            and value.relations # 确保该节点确实关联了某个表，是合法的关系表达式。
            and not value.find(ops.Impure, filter=ops.Value) # 如果字段的计算涉及到“不纯”操作（例如调用了 random() 或其他非确定性函数），则停止追踪。因为不可预测的表达式无法保证跨层级的语义等价，不能盲目地进行解引用替换。
        ):
            yield value, distance


    # 主要作用是预计算：通过分析当前的各个关系（rels）及其字段来源，构建出一个“查找表”（Lookup Table）。
    # 表明确了“如果用户引用了祖先节点字段 X，那么在当前节点下，我应该用哪个字段 Y 来替换它”。
    def _fill_substitution_mappings(self) -> None:
        # 首先检查 self.subs 和 self.ambigs 是否已经存在。如果是，直接返回。这确保了昂贵的映射计算过程每个实例只会运行一次（懒加载模式）
        if self.subs is not None and self.ambigs is not None:
            return

        mapping = defaultdict(dict)
        # 遍历所有的目标关系（rel），以及这些关系中的每一个字段（field）
        for rel in self.rels:
            for field in rel.fields.values():
                for val, distance in self.__class__.backtrack(field):
                    # { 原始根节点 (val) : { 当前层级可用字段 (field) : 距离 (distance) } }
                    # 原始字段 val 可以通过 field 访问，代价是 distance 层
                    mapping[val][field] = distance
        # 决策：生成替换表或歧义表
        subs, ambigs = {}, {}
        for from_, to in mapping.items():
            mindist = min(to.values())
            minkeys = [k for k, v in to.items() if v == mindist]
            # if all the closest fields are from the same relation, then we
            # can safely substitute them and we pick the first one arbitrarily
            # 对于每个原始字段（from_），它在当前关系中可能有多个访问路径。我们取距离最近的那个（mindist）
            # 安全情况：如果所有“距离最短”的候选字段都源自同一个关系（minkeys[0].relations == k.relations），
            # 则说明它们在语义上是等价的，可以直接替换。我们任选其一（通常是第一个）存入 subs。
            if all(minkeys[0].relations == k.relations for k in minkeys):
                subs[from_] = minkeys[0]
            # 歧义情况：如果最优距离的候选者来自不同的关系（例如左右表都有一个叫 id 的字段），Ibis 无法自动决定用哪一个，此时将其存入 ambigs，后续调用 dereference 时会报错。
            else:
                ambigs[from_] = minkeys
        # 将初始化时传入的 extra（用户自定义的额外替换规则）合并到 subs 中。这允许在自动推断之外，进行人工干预或强制指定映射。
        if extra := self.extra:
            subs.update(extra)

        self.subs = subs
        self.ambigs = ambigs
    # 方法处理传入的多个表达式 (*values)，逐一进行解引用
    def dereference(self, *values: ir.Value) -> Iterator[ops.Value]:
        """Dereference values to target relations.

        Also check for ambiguous field references. If a field reference is found
        which is marked as ambiguous, then raise an error.

        Parameters
        ----------
        values
            Expression values to dereference.

        Returns
        -------
        tuple[ops.Value]
            The dereferenced values.
        """
        for v in values:

            if (rels := v.relations) and rels != self.rels:
                # called on every iteration but only does work once per
                # instance
                self._fill_substitution_mappings()
                # 在执行替换前，使用 v.find() 在表达式树中搜索是否存在“歧义字段”。
                if ambigs := v.find(self.ambigs.__contains__, filter=ops.Value):
                    raise IbisInputError(
                        f"Ambiguous field reference {ambigs!r} in expression {v!r}"
                    )
                # 执行表达式重写
                # 会遍历表达式树中的每一个节点。如果某个节点（如 ops.Field）存在于 self.subs 映射表中，它就会被替换为指向当前关系的目标节点。
                yield v.replace(self.subs, filter=ops.Value)
            else:
                # 如果表达式 v 的关联关系（v.relations）已经属于当前目标 self.rels，则说明它已经是“本地化”的，无需任何处理，直接 yield v 返回
                yield v


def flatten_predicates(node):
    """Yield the expressions corresponding to the `And` nodes of a predicate.

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([("a", "int64"), ("b", "string")], name="t")
    >>> filt = (t.a == 1) & (t.b == "foo")
    >>> predicates = flatten_predicates(filt.op())
    >>> len(predicates)
    2
    >>> predicates[0].to_expr().name("left")
    r0 := UnboundTable: t
      a int64
      b string
    left: r0.a == 1
    >>> predicates[1].to_expr().name("right")
    r0 := UnboundTable: t
      a int64
      b string
    right: r0.b == 'foo'

    """

    def predicate(node):
        if isinstance(node, ops.And):
            # proceed and don't yield the node
            return True, None
        else:
            # halt and yield the node
            return False, node

    return list(traverse(predicate, node))


@replace(p.Field(p.JoinChain))
def peel_join_field(_):
    return _.rel.values[_.name]


@replace(p.ScalarParameter)
def replace_parameter(_, params, **kwargs):
    """Replace scalar parameters with their values."""
    return ops.Literal(value=params[_], dtype=_.dtype)


@replace(p.StringSlice)
def lower_stringslice(_, **kwargs):
    """Rewrite StringSlice in terms of Substring."""
    if _.end is None:
        return ops.Substring(_.arg, start=_.start)
    if _.start is None:
        return ops.Substring(_.arg, start=0, length=_.end)
    if (
        isinstance(_.start, ops.Literal)
        and isinstance(_.start.value, int)
        and isinstance(_.end, ops.Literal)
        and isinstance(_.end.value, int)
    ):
        # optimization for constant values
        length = _.end.value - _.start.value
    else:
        length = ops.Subtract(_.end, _.start)
    return ops.Substring(_.arg, start=_.start, length=length)


@replace(p.Analytic)
def wrap_analytic(_, **__):
    # Wrap analytic functions in a window function
    return ops.WindowFunction(_)


@replace(p.Reduction)
def project_wrap_reduction(_, rel):
    # Query all the tables that the reduction depends on
    if _.relations == {rel}:
        # The reduction is fully originating from the `rel`, so turn
        # it into a window function of `rel`
        return ops.WindowFunction(_, order_by=getattr(_, "order_by", ()))
    else:
        # 1. The reduction doesn't depend on any table, constructed from
        #    scalar values, so turn it into a scalar subquery.
        # 2. The reduction is originating from `rel` and other tables,
        #    so this is a correlated scalar subquery.
        # 3. The reduction is originating entirely from other tables,
        #    so this is an uncorrelated scalar subquery.
        return ops.ScalarSubquery(_.to_expr().as_table())


def rewrite_project_input(value, relation):
    # we need to detect reductions which are either turned into window functions
    # or scalar subqueries depending on whether they are originating from the
    # relation
    return value.replace(
        wrap_analytic | project_wrap_reduction,
        filter=p.Value & ~p.WindowFunction,
        context={"rel": relation},
    )


ReductionLike = p.Reduction | p.Field(p.Aggregate(groups={}))


@replace(ReductionLike)
def filter_wrap_reduction(_):
    # Wrap reductions or fields referencing an aggregation without a group by -
    # which are scalar fields - in a scalar subquery. In the latter case we
    # use the reduction value from the aggregation.
    if isinstance(_, ops.Field):
        value = _.rel.values[_.name]
    else:
        value = _
    return ops.ScalarSubquery(value.to_expr().as_table())


def rewrite_filter_input(value):
    return value.replace(
        wrap_analytic | filter_wrap_reduction, filter=p.Value & ~p.WindowFunction
    )


@replace(p.Analytic | p.Reduction)
def window_wrap_reduction(_, window):
    # Wrap analytic and reduction functions in a window function. Used in the
    # value.over() API.
    return ops.WindowFunction(
        _,
        how=window.how,
        start=window.start,
        end=window.end,
        group_by=window.groupings,
        order_by=window.orderings,
    )


@replace(p.WindowFunction)
def window_merge_frames(_, window):
    # Merge window frames, used in the value.over() and groupby.select() APIs.
    if _.how != window.how:
        raise ExpressionError(
            f"Unable to merge {_.how} window with {window.how} window"
        )
    elif _.start and window.start and _.start != window.start:
        raise ExpressionError(
            "Unable to merge windows with conflicting `start` boundary"
        )
    elif _.end and window.end and _.end != window.end:
        raise ExpressionError("Unable to merge windows with conflicting `end` boundary")

    start = _.start or window.start
    end = _.end or window.end
    group_by = tuple(toolz.unique(_.group_by + window.groupings))

    order_keys = {}
    for sort_key in window.orderings + _.order_by:
        order_keys[sort_key.arg] = sort_key.ascending, sort_key.nulls_first

    order_by = (
        ops.SortKey(expr, ascending=ascending, nulls_first=nulls_first)
        for expr, (ascending, nulls_first) in order_keys.items()
    )
    return _.copy(start=start, end=end, group_by=group_by, order_by=order_by)


def rewrite_window_input(value, window):
    context = {"window": window}
    # if self is a reduction or analytic function, wrap it in a window function
    node = value.replace(
        window_wrap_reduction,
        filter=p.Value & ~p.WindowFunction,
        context=context,
    )
    # if self is already a window function, merge the existing window frame
    # with the requested window frame
    return node.replace(window_merge_frames, filter=p.Value, context=context)


# TODO(kszucs): schema comparison should be updated to not distinguish between
# different column order
@replace(p.Project(y @ p.Relation) & Check(_.schema == y.schema))
def complete_reprojection(_, y):
    # TODO(kszucs): this could be moved to the pattern itself but not sure how
    # to express it, especially in a shorter way then the following check
    for name in _.schema:
        if _.values[name] != ops.Field(y, name):
            return _
    return y


@replace(p.Project(y @ p.Project))
def subsequent_projects(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    values = {k: v.replace(rule, filter=ops.Value) for k, v in _.values.items()}
    return ops.Project(y.parent, values)


@replace(p.Filter(y @ p.Filter))
def subsequent_filters(_, y):
    rule = p.Field(y, name) >> d.Field(y.parent, name)
    preds = tuple(v.replace(rule, filter=ops.Value) for v in _.predicates)
    return ops.Filter(y.parent, y.predicates + preds)


@replace(p.Filter(y @ p.Project))
def reorder_filter_project(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    preds = tuple(v.replace(rule, filter=ops.Value) for v in _.predicates)

    inner = ops.Filter(y.parent, preds)
    rule = p.Field(y.parent, name) >> d.Field(inner, name)
    projs = {k: v.replace(rule, filter=ops.Value) for k, v in y.values.items()}

    return ops.Project(inner, projs)


def simplify(node):
    # TODO(kszucs): add a utility to the graph module to do rewrites in multiple
    # passes after each other
    node = node.replace(reorder_filter_project)
    node = node.replace(reorder_filter_project)
    node = node.replace(subsequent_projects | subsequent_filters)
    node = node.replace(complete_reprojection)
    return node
