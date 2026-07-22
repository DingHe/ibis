from __future__ import annotations

import abc
import calendar
import itertools
import math
import operator
import string
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

import ibis.common.exceptions as com
import ibis.common.patterns as pats
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers._compat import WITH_ARG
from ibis.backends.sql.rewrites import (
    FirstValue,
    LastValue,
    add_one_to_nth_value_input,
    add_order_by_to_empty_ranking_window_functions,
    empty_in_values_right_side,
    lower_bucket,
    lower_capitalize,
    lower_sample,
    one_to_zero_index,
    sqlize,
)
from ibis.config import options
from ibis.expr.operations.udf import InputType
from ibis.expr.rewrites import lower_stringslice
from ibis.util import get_subclasses

try:
    from sqlglot.expressions import Alter
except ImportError:
    from sqlglot.expressions import AlterTable
else:

    def AlterTable(*args, kind="TABLE", **kwargs):
        return Alter(*args, kind=kind, **kwargs)


try:
    from sqlglot.expressions import AlterRename as RenameTable
except ImportError:
    from sqlglot.expressions import RenameTable  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import ibis.expr.schema as sch
    import ibis.expr.types as ir
    from ibis.backends.sql.datatypes import SqlglotType


ALL_OPERATIONS = frozenset(get_subclasses(ops.Node))

# AggGen 是一个描述符类,专门用来生成/编译 SQL 里的聚合函数调用(比如 SUM、COUNT、AVG 等)。
# 核心价值在于:不同的聚合函数在生成 SQL 时,可能需要处理一些"通用但又不完全一致"的额外逻辑,比如:
# FILTER 子句:某些数据库支持 SUM(x) FILTER (WHERE cond) 这种语法,在聚合前先过滤行;
# ORDER BY:某些聚合函数(比如 STRING_AGG、ARRAY_AGG)在聚合之前需要指定排序顺序,即 STRING_AGG(x ORDER BY y)。
# 但并不是所有 SQL 方言都原生支持这两种子句。AggGen 把这些"支持与否"的差异,抽象成两个布尔开关(supports_filter、supports_order_by),通过统一的 aggregate 方法,自动决定:
# 如果方言原生支持 FILTER,就直接生成 FILTER 子句;
# 如果不支持,就退化成用 CASE WHEN(即 if_)在参数层面手动做过滤,把不满足条件的值替换成 NULL,让聚合函数自然忽略它们;
# 如果方言不支持 ORDER BY 却调用时传了 order_by,直接抛异常,提示该方言不支持这种"排序敏感的聚合"。
class AggGen:
    """A descriptor for compiling aggregate functions.

    Common cases can be handled by setting configuration flags,
    special cases should override the `aggregate` method directly.

    Parameters
    ----------
    supports_filter
        Whether the backend supports a FILTER clause in the aggregate.
        Defaults to False.
    supports_order_by
        Whether the backend supports an ORDER BY clause in (relevant)
        aggregates. Defaults to False.
    """
    # 专门用来支持"属性访问"和"下标访问"两种语法糖,让调用方可以写 compiler.agg.sum(...) 或 compiler.agg["sum"](...),
    # 而不必写成 compiler.agg.aggregate(compiler, "sum", ...) 这种繁琐形式。
    class _Accessor:
        """An internal type to handle getattr/getitem access."""
        # __slots__ = ("compiler", "handler"):限定这个类的实例只能有 compiler 和 handler 两个属性(不能动态添加别的属性),这是一种内存优化手段,避免每个实例都携带一个 __dict__。
        __slots__ = ("compiler", "handler")
        # handler:实际要调用的处理函数(在这里就是 AggGen.aggregate 这个方法本身,还未绑定具体名字)。
        # compiler:所属的编译器实例(比如某个具体方言的 SQLGlotCompiler 子类实例),后续调用聚合函数时要用到它(比如取 compiler.f[name]、compiler.dialect 等)。
        def __init__(self, handler: Callable, compiler: SQLGlotCompiler):
            self.handler = handler
            self.compiler = compiler
        # 当访问这个对象上任意一个属性名(比如 .sum、.count、.avg)时,Python 找不到已定义的同名属性,就会触发 __getattr__。
        # 这里的处理是:用 functools.partial 把 self.handler(即 AggGen.aggregate 方法)、self.compiler(编译器实例)和 name(访问的属性名,即聚合函数名,比如 "sum")预先绑定,
        # 返回一个"只需要再传参数(和可选的 where/order_by)"就能调用的偏函数。
        def __getattr__(self, name: str) -> Callable:
            return partial(self.handler, self.compiler, name)
        # 把 __getitem__(下标访问,比如 accessor["sum"])也指向同一个 __getattr__ 实现,这样 accessor["sum"](col) 和 accessor.sum(col) 效果完全一致——支持了两种等价的调用语法(属性式和字典式),方便在某些聚合函数名恰好是 Python 关键字或存在特殊字符时,也可以用下标方式访问。
        __getitem__ = __getattr__

    __slots__ = ("supports_filter", "supports_order_by")
    # 接收两个仅限关键字的布尔参数(前面的 * 强制调用方必须用关键字方式传参,不能位置传参),并保存为实例属性:
    def __init__(
        self, *, supports_filter: bool = False, supports_order_by: bool = False
    ):
        self.supports_filter = supports_filter
        self.supports_order_by = supports_order_by

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        return AggGen._Accessor(self.aggregate, instance)
    # 真正执行"编译某个具体聚合函数"逻辑的方法
    # compiler:调用方所属的编译器实例,用来访问该方言下的函数构造器 compiler.f、条件表达式辅助方法 compiler.if_、方言名 compiler.dialect 等。
    # name:要编译的聚合函数名字(比如 "sum"、"count")。
    # *args:传给该聚合函数的位置参数(通常是要聚合的列表达式,可能有多个,比如某些聚合函数需要不止一个参数)。
    # where(关键字参数,默认 None):一个可选的过滤条件,对应 SQL 里聚合前的行过滤(即 FILTER 或退化成 CASE WHEN)。
    # order_by(关键字参数,默认空元组):一个可选的排序键元组,对应聚合内部的排序(即 ORDER BY)。
    def aggregate(
        self,
        compiler: SQLGlotCompiler,
        name: str,
        *args: Any,
        where: Any = None,
        order_by: tuple = (),
    ):
        """Compile the specified aggregate.

        Parameters
        ----------
        compiler
            The backend's compiler.
        name
            The aggregate name (e.g. `"sum"`).
        args
            Any arguments to pass to the aggregate.
        where
            An optional column filter to apply before performing the aggregate.
        order_by
            Optional ordering keys to use to order the rows before performing
            the aggregate.
        """
        # 从编译器实例的 f(应该是一个"函数工厂",支持通过下标获取某个 SQL 函数的可调用构造器)里,取出对应 name 名字的函数构造器,赋值给 func。之后 func(*args) 就相当于构造出诸如 SUM(x) 这样的 sqlglot 函数表达式。
        func = compiler.f[name]

        if order_by and not self.supports_order_by:
            raise com.UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                f"not supported for the {compiler.dialect} backend"
            )

        if where is not None and not self.supports_filter:
            args = tuple(compiler.if_(where, arg, NULL) for arg in args)

        if order_by and self.supports_order_by:
            *rest, last = args
            out = func(*rest, sge.Order(this=last, expressions=order_by))
        else:
            out = func(*args)

        if where is not None and self.supports_filter:
            out = sge.Filter(this=out, expression=sge.Where(this=where))

        return out


class VarGen:
    __slots__ = ()

    def __getattr__(self, name: str) -> sge.Var:
        return sge.Var(this=name)

    def __getitem__(self, key: str) -> sge.Var:
        return sge.Var(this=key)


class AnonymousFuncGen:
    __slots__ = ()

    def __getattr__(self, name: str) -> Callable[..., sge.Anonymous]:
        return lambda *args: sge.Anonymous(
            this=name, expressions=list(map(sge.convert, args))
        )

    def __getitem__(self, key: str) -> Callable[..., sge.Anonymous]:
        return getattr(self, key)

# FuncGen 是一个函数生成器 / 函数工厂类,专门用来把"函数名字符串"动态转换成对应的 sqlglot 函数调用表达式(sge.Func 及其子类)
# 核心价值在于:SQL 里有海量的内置函数(ABS、UPPER、DATE_TRUNC 等等),不可能给每个函数都手写一个 Python 方法。FuncGen 通过重写 __getattr__,
# 实现了"访问任意属性名,都自动生成对应名字的 SQL 函数调用"的动态分发机制——比如 f.abs(x) 会自动生成 ABS(x) 对应的 sqlglot 表达式,而不需要预先在类里定义一个 abs 方法。
# 同时,对于少数几个"不能简单套用通用函数调用模板"的特殊结构(比如数组字面量 ARRAY[...]、EXISTS(...)、字符串拼接 CONCAT 等,
# 它们在 sqlglot 里对应专门的表达式类而非普通函数调用形式),FuncGen 显式定义了对应的方法来覆盖默认的动态分发行为,确保生成正确的 AST 结构。
class FuncGen:
    # 限定实例只能拥有这四个属性,不能动态添加其它属性,这是内存优化手段(避免每个实例都携带 __dict__)
    # anon:一个 AnonymousFuncGen 实例,用于生成"匿名函数调用"(即那些不在 sqlglot 已知函数注册表里的函数,需要特殊处理)。
    # copy:控制 sqlglot 在构造表达式时是否要深拷贝子节点。
    # dialect:目标 SQL 方言。
    # namespace:函数名的命名空间前缀(比如某些方言的函数需要带 schema/package 前缀,如 pg_catalog.abs)。
    __slots__ = ("anon", "copy", "dialect", "namespace")

    # namespace: str | None = None:可选的命名空间前缀字符串,默认为 None。如果某个具体方言的函数需要带命名空间前缀(比如某些数据库的内置函数要写成 schema.func_name 的形式),就通过这个参数指定。保存为 self.namespace。
    # dialect: sg.Dialect:目标 SQL 方言对象,后续生成函数表达式时会用到它,确保按该方言的语法规则渲染/校验函数名和参数格式。保存为 self.dialect。
    # copy: bool = False:控制后续调用 sg.func(...) 构造表达式时是否要拷贝参数节点,默认为 False(不拷贝,性能更优,因为通常没必要在构造时额外深拷贝)。保存为 self.copy。
    def __init__(
        self, *, dialect: sg.Dialect, namespace: str | None = None, copy: bool = False
    ) -> None:
        self.dialect = dialect
        self.namespace = namespace
        self.anon = AnonymousFuncGen()
        self.copy = copy
    # 实现"访问任意属性名,自动生成对应 SQL 函数调用"的机制:
    # name: str:被访问的属性名,比如调用 f.upper(col) 时,这里的 name 就是字符串 "upper"。
    def __getattr__(self, name: str) -> Callable[..., sge.Func]:
        # filter(None, (self.namespace, name)):过滤掉元组里的假值(比如 self.namespace 为 None 时会被过滤掉,只留下真正的 name)。
        name = ".".join(filter(None, (self.namespace, name)))
        # 返回一个闭包函数(lambda),这个 lambda 才是真正被调用的、生成函数表达式的可调用对象。
        # sg.func(...):sqlglot 提供的通用函数构造辅助函数,根据函数名字符串和参数,构造出对应的 sge.Func(或其已知子类,比如 sqlglot 认识 upper 就会构造出对应的 Upper 表达式类;
        # 如果不认识这个函数名,会构造成通用的 Anonymous 函数表达式)。
        return lambda *args, **kwargs: sg.func(
            name,
            *map(sge.convert, args),
            **kwargs,
            copy=self.copy,
            dialect=self.dialect,
        )

    def __getitem__(self, key: str) -> Callable[..., sge.Func]:
        return getattr(self, key)

    def array(self, *args: Any) -> sge.Array:
        if not args:
            return sge.Array(expressions=[])

        first, *rest = args

        if isinstance(first, sge.Select):
            assert not rest, (
                "only one argument allowed when `first` is a select statement"
            )

        return sge.Array(expressions=list(map(sge.convert, (first, *rest))))

    def tuple(self, *args: Any) -> sge.Anonymous:
        return self.anon.tuple(*args)

    def exists(self, query: sge.Expression) -> sge.Exists:
        return sge.Exists(this=query)

    def concat(self, *args: Any) -> sge.Concat:
        return sge.Concat(expressions=list(map(sge.convert, args)))

    def map(self, keys: Iterable, values: Iterable) -> sge.Map:
        return sge.Map(keys=keys, values=values)


class ColGen:
    __slots__ = ("table",)

    def __init__(self, table: str | None = None) -> None:
        self.table = table

    def __getattr__(self, name: str) -> sge.Column:
        return sg.column(name, table=self.table, copy=False)

    def __getitem__(self, key: str) -> sge.Column:
        return sg.column(key, table=self.table, copy=False)


C = ColGen()
NULL = sge.Null()
FALSE = sge.false()
TRUE = sge.true()
STAR = sge.Star()

# SQLGlotCompiler 的核心作用是担任“翻译官”的角色：将 Ibis 的内部逻辑表达（一种与具体数据库无关的抽象语法树，即 Ibis Expression Tree）转换成 sqlglot 库所理解的 SQL 表达式树，
# 进而生成特定数据库（如 PostgreSQL, MySQL, BigQuery, ClickHouse 等）的合法 SQL 语句。
# 核心工作流：
# 标准化（Rewrite）： 在编译前通过一系列预定义规则（rewrites）简化表达式，处理不同数据库对 SQL 标准实现不一致的问题。
# 树遍历（Visitor Pattern）： 采用访问者模式，遍历 Ibis 的操作节点（ops.Node），递归调用 visit_* 方法将其映射为 sqlglot.expressions。
# 方言化（Dialect）： 利用 sqlglot 的方言能力，将通用的表达式树渲染成目标数据库的特定语法。
@public
class SQLGlotCompiler(abc.ABC):
    __slots__ = "f", "v"

    agg = AggGen()
    """A generator for handling aggregate functions"""
    # 编译前后的变换规则元组，用于将 Ibis 的复杂操作拆解为目标数据库支持的原子操作。
    rewrites: tuple[type[pats.Replace], ...] = (
        empty_in_values_right_side,
        add_order_by_to_empty_ranking_window_functions,
        one_to_zero_index,
        add_one_to_nth_value_input,
    )
    """A sequence of rewrites to apply to the expression tree before SQL-specific transforms."""

    post_rewrites: tuple[type[pats.Replace], ...] = ()
    """A sequence of rewrites to apply to the expression tree after SQL-specific transforms."""

    no_limit_value: sge.Null | None = None
    """The value to use to indicate no limit."""
    # 决定生成的标识符（列名、表名）是否强制加引号
    quoted: bool = True
    """Whether to always quote identifiers."""

    copy_func_args: bool = False
    """Whether to copy function arguments when generating SQL."""
    # 标识目标数据库是否支持 QUALIFY 子句（常用于窗口函数过滤）。
    supports_qualify: bool = False
    """Whether the backend supports the QUALIFY clause."""
    # 定义了该后端处理浮点数特殊值（NaN/Inf）的 SQL 字面量形式。
    NAN: ClassVar[sge.Expression] = sge.Cast(
        this=sge.convert("NaN"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    """Backend's NaN literal."""
    # 定义了该后端处理浮点数特殊值（NaN/Inf）的 SQL 字面量形式。
    POS_INF: ClassVar[sge.Expression] = sge.Cast(
        this=sge.convert("Inf"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    """Backend's positive infinity literal."""
    # 定义了该后端处理浮点数特殊值（NaN/Inf）的 SQL 字面量形式。
    NEG_INF: ClassVar[sge.Expression] = sge.Cast(
        this=sge.convert("-Inf"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    """Backend's negative infinity literal."""

    EXTRA_SUPPORTED_OPS: tuple[type[ops.Node], ...] = (
        ops.Project,
        ops.Filter,
        ops.Sort,
        ops.WindowFunction,
    )
    """A tuple of ops classes that are supported, but don't have explicit
    `visit_*` methods (usually due to being handled by rewrite rules). Used by
    `has_operation`"""
    # 显式声明该后端不支持的 Ibis 操作，编译遇到时将抛出错误
    UNSUPPORTED_OPS: tuple[type[ops.Node], ...] = ()
    """Tuple of operations the backend doesn't support."""
    # 定义“降级”规则，将高级 Ibis 操作（如 Bucket 分桶）重写为更基础的 SQL 表达式。
    LOWERED_OPS: dict[type[ops.Node], pats.Replace | None] = {
        ops.Bucket: lower_bucket,
        ops.Capitalize: lower_capitalize,
        ops.Sample: lower_sample(supported_methods=()),
        ops.StringSlice: lower_stringslice,
    }
    """A mapping from an operation class to either a rewrite rule for rewriting that
    operation to one composed of lower-level operations ("lowering"), or `None` to
    remove an existing rewrite rule for that operation added in a base class"""
    # 一个映射字典，将 Ibis 的具体算子类（如 ops.Abs）直接映射为目标数据库的函数名字符串（如 "abs"）
    SIMPLE_OPS = {
        ops.Abs: "abs",
        ops.Acos: "acos",
        ops.All: "bool_and",
        ops.Any: "bool_or",
        ops.ApproxCountDistinct: "approx_distinct",
        ops.ArrayContains: "array_contains",
        ops.ArrayFlatten: "flatten",
        ops.ArrayLength: "array_size",
        ops.ArraySort: "array_sort",
        ops.ArrayStringJoin: "array_to_string",
        ops.ArgMax: "max_by",
        ops.ArgMin: "min_by",
        ops.Asin: "asin",
        ops.Atan2: "atan2",
        ops.Atan: "atan",
        ops.Cos: "cos",
        ops.Cot: "cot",
        ops.Count: "count",
        ops.CumeDist: "cume_dist",
        ops.Date: "date",
        ops.DateFromYMD: "datefromparts",
        ops.Degrees: "degrees",
        ops.DenseRank: "dense_rank",
        ops.Exp: "exp",
        FirstValue: "first_value",
        ops.GroupConcat: "group_concat",
        ops.IfElse: "if",
        ops.IsInf: "isinf",
        ops.IsNan: "isnan",
        ops.JSONGetItem: "json_extract",
        LastValue: "last_value",
        ops.Levenshtein: "levenshtein",
        ops.Ln: "ln",
        ops.Log10: "log",
        ops.Log2: "log2",
        ops.Lowercase: "lower",
        ops.Map: "map",
        ops.Median: "median",
        ops.MinRank: "rank",
        ops.NTile: "ntile",
        ops.NthValue: "nth_value",
        ops.NullIf: "nullif",
        ops.PercentRank: "percent_rank",
        ops.Pi: "pi",
        ops.Power: "pow",
        ops.Radians: "radians",
        ops.RegexSearch: "regexp_like",
        ops.RegexSplit: "regexp_split",
        ops.RegexExtract: "regexp_extract",
        ops.Repeat: "repeat",
        ops.Reverse: "reverse",
        ops.RowNumber: "row_number",
        ops.Sign: "sign",
        ops.Sin: "sin",
        ops.Sqrt: "sqrt",
        ops.StartsWith: "starts_with",
        ops.StrRight: "right",
        ops.StringAscii: "ascii",
        ops.StringContains: "contains",
        ops.StringLength: "length",
        ops.StringReplace: "replace",
        ops.StringSplit: "split",
        ops.StringToDate: "str_to_date",
        ops.StringToTimestamp: "str_to_time",
        ops.Tan: "tan",
        ops.Translate: "translate",
        ops.Unnest: "explode",
        ops.Uppercase: "upper",
        ops.RandomUUID: "uuid",
        ops.RandomScalar: "rand",
    }
    # 定义二元操作符（如加减乘除、逻辑与或）对应的 sqlglot 表达式类型（如 sge.Add）
    BINARY_INFIX_OPS = {
        # Numeric
        ops.Add: sge.Add,
        ops.Subtract: sge.Sub,
        ops.Multiply: sge.Mul,
        ops.Divide: sge.Div,
        ops.Modulus: sge.Mod,
        ops.Power: sge.Pow,
        # Comparisons
        ops.GreaterEqual: sge.GTE,
        ops.Greater: sge.GT,
        ops.LessEqual: sge.LTE,
        ops.Less: sge.LT,
        ops.Equals: sge.EQ,
        ops.NotEquals: sge.NEQ,
        # Logical
        ops.And: sge.And,
        ops.Or: sge.Or,
        ops.Xor: sge.Xor,
        # Bitwise
        ops.BitwiseLeftShift: sge.BitwiseLeftShift,
        ops.BitwiseRightShift: sge.BitwiseRightShift,
        ops.BitwiseAnd: sge.BitwiseAnd,
        ops.BitwiseOr: sge.BitwiseOr,
        ops.BitwiseXor: sge.BitwiseXor,
        # Date
        ops.DateAdd: sge.Add,
        ops.DateSub: sge.Sub,
        ops.DateDiff: sge.Sub,
        # Time
        ops.TimeAdd: sge.Add,
        ops.TimeSub: sge.Sub,
        ops.TimeDiff: sge.Sub,
        # Timestamp
        ops.TimestampAdd: sge.Add,
        ops.TimestampSub: sge.Sub,
        ops.TimestampDiff: sge.Sub,
        # Interval
        ops.IntervalAdd: sge.Add,
        ops.IntervalMultiply: sge.Mul,
        ops.IntervalSubtract: sge.Sub,
    }

    # A set of SQLGlot classes that may need to be parenthesized
    SQLGLOT_NEEDS_PARENS = set(BINARY_INFIX_OPS.values()).union((sge.Is,))

    # A set of SQLGlot classes that are associative operations
    SQLGLOT_ASSOCIATIVE_OPS = {
        sge.Add,
        sge.Mul,
        sge.And,
        sge.Or,
        sge.Xor,
        sge.BitwiseAnd,
        sge.BitwiseOr,
        sge.BitwiseXor,
    }

    # Constructed dynamically in `__init_subclass__` from their respective
    # UPPERCASE values to handle inheritance, do not modify directly here.
    extra_supported_ops: ClassVar[frozenset[type[ops.Node]]] = frozenset()
    lowered_ops: ClassVar[dict[type[ops.Node], pats.Replace]] = {}
    # 初始化 FuncGen（函数生成器）和 VarGen（变量生成器），用于在编译时便捷地创建 SQL 函数调用和变量引用。
    def __init__(self) -> None:
        self.f = FuncGen(
            dialect=self.__class__.dialect, copy=self.__class__.copy_func_args
        )
        self.v = VarGen()
    # __init_subclass__ 在 Python 中用于拦截类的创建。
    # 这段代码的目的是：当用户定义一个新的 SQL 方言编译器（继承自 SQLGlotCompiler）时，自动根据类属性（如 SIMPLE_OPS）生成对应的 visit_ 方法，从而减少大量重复的样板代码。
    # cls: 代表当前正在被定义的子类（即继承了 SQLGlotCompiler 的类）
    # **kwargs: 传递给父类初始化的额外参数，保证标准的类创建流程不被中断。
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # # 将算子类名（如 Addition）映射为编译器方法名（如 visit_Addition）
        def methodname(op: type) -> str:
            assert isinstance(type(op), type), type(op)
            return f"visit_{op.__name__}"
        # 生成简单算子的实现 (SIMPLE_OPS)
        # 处理那些直接映射到 SQL 函数的操作（如 Sum -> SUM(), Abs -> ABS()）
        def make_impl(op, target_name):
            assert isinstance(type(op), type), type(op)
            # # 如果算子属于 Reduction（聚合操作，如 SUM/AVG）
            if issubclass(op, ops.Reduction):

                def impl(
                    self, _, *, _name: str = target_name, where, order_by=(), **kw
                ):
                    return self.agg[_name](*kw.values(), where=where, order_by=order_by)
            # 普通函数调用（如 ABS/LENGTH）
            else:

                def impl(self, _, *, _name: str = target_name, **kw):
                    return self.f[_name](*kw.values())

            return impl
        # # 遍历字典自动注册方法：setattr 会将生成的 impl 绑定到新类 cls 上
        for op, target_name in cls.SIMPLE_OPS.items():
            setattr(cls, methodname(op), make_impl(op, target_name))

        # Define binary op methods, only if BINARY_INFIX_OPS is set on the
        # compiler class.
        # 处理二元运算符 (BINARY_INFIX_OPS)
        # 处理形如 a + b 或 a > b 的中缀表达式，将其映射为 SQLGlot 的二元表达式类。
        if binops := cls.__dict__.get("BINARY_INFIX_OPS", {}):

            def make_binop(sge_cls):
                def impl(self, op, *, left, right):
                    # # 调用统一的 binop 转换逻辑
                    return self.binop(sge_cls, left, right)

                return impl

            for op, sge_cls in binops.items():
                setattr(cls, methodname(op), make_binop(sge_cls))

        # unconditionally raise an exception for unsupported operations
        #
        # these *must* be defined after SIMPLE_OPS to handle compilers that
        # subclass other compilers
        # 处理不支持的算子
        # 强制要求在遇到不支持的操作时抛出错误，防止编译器静默失败。
        # 标记显式声明不支持的算子
        for op in cls.UNSUPPORTED_OPS:
            # change to visit_Unsupported in a follow up
            # TODO: handle geoespatial ops as a separate case?
            setattr(cls, methodname(op), cls.visit_Undefined)

        # raise on any remaining unsupported operations
        # # 兜底逻辑：遍历所有 Ibis 已知算子，如果子类中未定义 visit_ 方法，则统一指向 Undefined
        for op in ALL_OPERATIONS:
            name = methodname(op)
            if not hasattr(cls, name):
                setattr(cls, name, cls.visit_Undefined)

        # Amend `lowered_ops` and `extra_supported_ops` using their
        # respective UPPERCASE classvar values.
        # 合并 lowered_ops 和 extra_supported_ops
        # 为了处理“操作重写”逻辑（即编译器在处理复杂算子时，将其降级为多个简单算子的组合）。
        # # 将子类的配置（UPPERCASE）与父类的基础配置进行合并更新
        extra_supported_ops = set(cls.extra_supported_ops)
        lowered_ops = dict(cls.lowered_ops)
        extra_supported_ops.update(cls.EXTRA_SUPPORTED_OPS)
        for op_cls, rewrite in cls.LOWERED_OPS.items():
            if rewrite is not None:
                lowered_ops[op_cls] = rewrite  # 添加或覆盖重写规则
                extra_supported_ops.add(op_cls)
            else:
                lowered_ops.pop(op_cls, None)  # 如果设为 None，则移除支持
                extra_supported_ops.discard(op_cls)
        # 将结果存回类属性，确保后续编译流程可见
        cls.lowered_ops = lowered_ops
        cls.extra_supported_ops = frozenset(extra_supported_ops)

    @property
    @abc.abstractmethod
    def dialect(self) -> type[sg.Dialect]:
        """Backend dialect."""

    @property
    @abc.abstractmethod
    def type_mapper(self) -> type[SqlglotType]:
        """The type mapper for the backend."""

    def _compile_builtin_udf(self, udf_node: ops.ScalarUDF) -> None:  # noqa: B027
        """No-op."""

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> None:
        raise NotImplementedError(
            f"Python UDFs are not supported in the {self.dialect} backend"
        )

    def _compile_pyarrow_udf(self, udf_node: ops.ScalarUDF) -> None:
        raise NotImplementedError(
            f"PyArrow UDFs are not supported in the {self.dialect} backend"
        )

    def _compile_pandas_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(
            f"pandas UDFs are not supported in the {self.dialect} backend"
        )

    # Concrete API

    def if_(self, condition, true, false: sge.Expression | None = None) -> sge.If:
        return sge.If(
            this=sge.convert(condition),
            true=sge.convert(true),
            false=None if false is None else sge.convert(false),
        )

    def cast(self, arg, to: dt.DataType) -> sge.Cast:
        return sge.Cast(
            this=sge.convert(arg), to=self.type_mapper.from_ibis(to), copy=False
        )
    # 把用户传入的"参数字典"从面向用户的 ibis 表达式形式,转换成面向内部的 op 节点形式,方便后续在遍历表达式树时直接用 op 节点作为 key 去查找对应的具体值
    def _prepare_params(self, params):
        result = {}
        for param, value in params.items():
            node = param.op()
            # 如果这个参数节点是一个 Alias(即用户可能写了类似 ibis.param(...).name("some_alias") 这种带别名的形式),那么真正“有意义”的、
            # 参与计算的其实是被包在 Alias 里面的那个原始表达式节点(node.arg,即别名内部真正的算子),
            # 别名本身只是一层包装,不影响参数值的绑定关系
            if isinstance(node, ops.Alias):
                node = node.arg
            result[node] = value
        return result
    # 将 Ibis 表达式树（Expression Tree）转换为 SQLGlot 抽象语法树（AST）的核心入口。它是连接 Ibis 高级 API 与底层 SQL 生成逻辑的桥梁。
    # expr: ir.Expr: 传入的 Ibis 表达式对象（例如 table.filter(...).group_by(...)）
    def to_sqlglot(
        self,
        expr: ir.Expr,
        *,
        limit: Literal["default"] | int | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
    ):
        import ibis
        # 1. 强制将表达式转换为表格式
        # 无论用户传入的是标量（Scalar）还是表表达式，都统一转化为 Table 表达式，
        # 因为所有 SQL 查询最终的输出都是一张“表”。
        table_expr = expr.as_table()
        # 2. 处理 limit 参数
        # 如果用户指定了 "default"，则从 Ibis 全局配置中获取默认行数。
        if limit == "default":
            limit = ibis.options.sql.default_limit
        # 如果存在 limit，则在表达式树上动态附加一个 limit 算子。
        if limit is not None:
            table_expr = table_expr.limit(limit)

        if params is None:
            params = {}
        # 4. 核心编译步骤：翻译算子树
        # self.translate 是 SQLGlotCompiler 的递归核心，负责遍历 Ibis 的算子图 (op)，
        # 并将其逐个映射为 SQLGlot 的表达式 (sge)。
        sql = self.translate(table_expr.op(), params=params)
        # 5. 断言检查：防止顶级查询意外变成子查询
        # 在 SQL 顶层，必须是一个 SELECT 语句，不能是包装在括号里的子查询。
        assert not isinstance(sql, sge.Subquery)
        # 6. 标准化输出：处理纯表对象的情况
        # 如果 translate 的结果仅仅是一个表节点（例如直接执行 t.to_sqlglot()），
        # SQLGlot 需要将其显式转换为 SELECT * FROM t 的形式，才能作为合法的 SQL 查询。
        if isinstance(sql, sge.Table):
            sql = sg.select(STAR, copy=False).from_(sql, copy=False)

        assert not isinstance(sql, sge.Subquery)
        return sql
    # 整个 SQL 生成流程的逻辑枢纽。
    # 将高层的 Ibis 算子图（DAG）转换为 SQLGlot 的 AST 表达式，并负责处理表别名（Alias）和通用表表达式（CTE）。
    # op: 要翻译的根 Ibis 算子节点（通常是一个 ops.Relation 或 ops.Value）
    # params: Mapping[ir.Value, Any]: 外部传入的参数映射（例如将变量名替换为具体的过滤值）。
    def translate(self, op, *, params: Mapping[ir.Value, Any]) -> sge.Expression:
        """Translate an ibis operation to a sqlglot expression.

        Parameters
        ----------
        op
            An ibis operation
        params
            A mapping of expressions to concrete values
        compiler
            An instance of SQLGlotCompiler
        translate_rel
            Relation node translator
        translate_val
            Value node translator

        Returns
        -------
        sqlglot.expressions.Expression
            A sqlglot expression

        """
        # substitute parameters immediately to avoid having to define a
        # ScalarParameter translation rule
        # 1. 替换参数：将 Ibis 表达式中的参数节点替换为具体值，避免后续定义复杂的翻译规则
        params = self._prepare_params(params)
        # 2. 算子降级 (Lowering)：如果某些算子在目标方言中不支持，将其拆解为基础算子组合
        if self.lowered_ops:
            op = op.replace(reduce(operator.or_, self.lowered_ops.values()))
        # 3. 规范化 SQL 化 (sqlize)：这是关键步骤，它将算子图重写为“友好的 SQL 结构”
        # 返回重写后的算子树 `op` 和需要提取为 CTE 的节点列表 `ctes`
        op, ctes = sqlize(
            op,
            params=params,
            # 应用 self.rewrites(该方言特定的重写规则,比如把某些 ibis 专属语义转换成 SQL 友好的形式)
            rewrites=self.rewrites,
            # 应用 self.post_rewrites(重写之后的收尾处理规则)
            post_rewrites=self.post_rewrites,
            # 根据 options.sql.fuse_selects 配置决定是否要"融合"连续的 select(减少嵌套子查询,生成更精简的 SQL)
            fuse_selects=options.sql.fuse_selects,
        )
        # 记录每个关系节点(Relation)对应的别名(比如 t0、t1 或显式指定的别名)。
        aliases = {}
        # 用来给没有显式别名的关系节点生成唯一的默认别名,如 t0, t1, t2...。
        counter = itertools.count()
        # 第二阶段：定义翻译规则(递归遍历)
        # 定义了一个闭包 fn，作为 op.map(fn) 的核心逻辑，按拓扑排序从叶子节点向根节点翻译：
        # node:当前正在处理的 ibis 算子节点
        def fn(node, __unused__, **kwargs):
            # # 调用 visit_node：根据 node 类型寻找对应的 visit_ 方法（如 visit_Selection）
            result = self.visit_node(node, **kwargs)

            # if it's not a relation then we don't need to do anything special
            # 如果当前节点就是最顶层的根节点(node is op),或者当前节点根本不是一个"关系"类型的节点(比如它是一个标量值/列表达式,而非 select 语句这样的表格结构),
            # 那么直接返回翻译结果,不需要做"打别名、包成子查询"这类特殊处理——因为别名/子查询包装只对"作为子查询嵌入到别的 select 中的关系节点"才有意义。
            if node is op or not isinstance(node, ops.Relation):
                return result

            # alias ops.AliasedRelations to their explicitly assigned name otherwise generate
            # 自动生成别名：SQL 语句中每个子查询都必须有别名 (t0, t1...)
            alias = (
                node.name
                if isinstance(node, ops.AliasedRelation)
                else f"t{next(counter)}"
            )
            aliases[node] = alias
            # 把字符串别名转换成一个正式的 sqlglot 标识符(Identifier)对象
            alias = sg.to_identifier(alias, quoted=self.quoted)
            if isinstance(result, sge.Subquery):
                return result.as_(alias, quoted=self.quoted) # 将 result 包装为 (SELECT ...) AS tN
            else:
                try:
                    return result.subquery(alias, copy=False)
                except AttributeError:
                    return result.as_(
                        alias, quoted=self.quoted, table=isinstance(result, sge.Table)
                    )

        # apply translate rules in topological order
        # 第三阶段：构建最终查询
        # 1. 应用翻译规则：生成所有节点的 SQLGlot 表达式映射表
        # 自底向上(先子节点后父节点)地对整棵表达式树的每一个节点调用 fn,并把每个节点对应的翻译结果收集到一个字典 results(key 是 op 节点,value 是对应的 sqlglot 表达式)里
        results = op.map(fn)

        # get the root node as a sqlglot select statement
        # 2. 获取根节点结果，并清理（如果是表则转为 SELECT *，如果是子查询则拆解）
        # 取出根节点 op 对应的翻译结果,这就是"最终 select 语句"的候选者。
        out = results[op]
        if isinstance(out, sge.Table):
            out = sg.select(STAR, copy=False).from_(out, copy=False)
        elif isinstance(out, sge.Subquery):
            # 如果根节点被翻译成了一个 Subquery(即之前 fn 函数里给它套上了括号+别名的形式,
            # 虽然对根节点这一层通常在 fn 里会被跳过打别名,但这里做个兜底),就把它"拆开",取出 .this(子查询内部真正的 select 语句),不需要保留最外层的括号包装和别名,
            # 因为这是最终输出,不需要作为子查询嵌入到别的地方。
            out = out.this
        # 3. 组装 CTE (WITH 子句)
        merged_ctes = []
        # 遍历每一个 cte 节点,取出它对应的翻译结果 this。
        # 如果这个结果本身已经带有 alias(说明前面 fn 函数已经给它包了一层别名/子查询),就取 .this 拿到里面真正的 select 语句本体,去掉多余的别名包装(因为 CTE 自己会单独指定别名)。
        for cte in ctes:
            this = results[cte]
            if "alias" in this.args:
                this = this.this
            # 处理别名逻辑，构建 SQLGlot 的 CTE 结构
            modified_cte = sge.CTE(
                alias=sg.to_identifier(aliases[cte], quoted=self.quoted), this=this
            )
            merged_ctes.append(modified_cte)
        merged_ctes.extend(out.ctes)
        out.args.pop(WITH_ARG, None)
        # 4. 合并 CTE 到主查询：使用 reduce 将所有 CTE 挂载到主查询的 WITH 子句中
        # 用 reduce 把所有 merged_ctes 依次通过 .with_(...) 方法附加到 out 上,最终重建出一个包含全部 WITH cte1 AS (...), cte2 AS (...) SELECT ... 结构的完整 select 语句
        out = reduce(
            lambda parsed, cte: parsed.with_(
                cte.args["alias"],
                as_=cte.args["this"],
                dialect=self.dialect,
                copy=False,
            ),
            merged_ctes,
            out,
        )

        return out
    # 根据传入的 ibis 算子节点(op)的具体类型,找到并调用对应的翻译规则方法,把该节点翻译成 sqlglot 表达式。
    # 它本质上实现了一种"基于类型名字符串反射查找"的多态分发机制(类似手写的 single-dispatch)。
    def visit_node(self, op: ops.Node, **kwargs):
        # 如果 op 是 ops.ScalarUDF(标量用户自定义函数,scalar user-defined function)的实例,直接调用固定的 self.visit_ScalarUDF(op, **kwargs) 方法来处理
        if isinstance(op, ops.ScalarUDF):
            return self.visit_ScalarUDF(op, **kwargs)
        # 如果 op 是 ops.AggUDF(聚合类型的用户自定义函数,aggregate UDF)的实例,也做同样的特殊处理,统一路由到 self.visit_AggUDF(op, **kwargs)。
        # 这是因为聚合 UDF 同样存在"类名是动态生成、无法一一枚举"的问题。
        elif isinstance(op, ops.AggUDF):
            return self.visit_AggUDF(op, **kwargs)
        else:
            # 如果 op 既不是 ScalarUDF 也不是 AggUDF(也就是常规的、内置的 ibis 算子类型),就走通用的分发逻辑:
            method = getattr(self, f"visit_{type(op).__name__}", None)
            if method is not None:
                return method(op, **kwargs)
            else:
                raise com.OperationNotDefinedError(
                    f"No translation rule for {type(op).__name__}"
                )

    def visit_Field(self, op, *, rel, name):
        return sg.column(
            self._gen_valid_name(name), table=rel.alias_or_name, quoted=self.quoted
        )

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype

        if from_.is_integer() and to.is_interval():
            return self._make_interval(arg, to.unit)

        return self.cast(arg, to)

    def visit_ScalarSubquery(self, op, *, rel):
        return rel.this.subquery(copy=False)

    def visit_Literal(self, op, *, value, dtype):
        """Compile a literal value.

        This is the default implementation for compiling literal values.

        Most backends should not need to override this method unless they want
        to handle NULL literals as well as every other type of non-null literal
        including integers, floating point numbers, decimals, strings, etc.

        The logic here is:

        1. If the value is None and the type is nullable, return NULL
        1. If the value is None and the type is not nullable, raise an error
        1. Call `visit_NonNullLiteral` method.
        1. If the previous returns `None`, call `visit_DefaultLiteral` method
           else return the result of the previous step.
        """
        if value is None:
            if dtype.nullable:
                return NULL if dtype.is_null() else self.cast(NULL, dtype)
            raise com.UnsupportedOperationError(
                f"Unsupported NULL for non-nullable type: {dtype!r}"
            )
        else:
            result = self.visit_NonNullLiteral(op, value=value, dtype=dtype)
            if result is None:
                return self.visit_DefaultLiteral(op, value=value, dtype=dtype)
            return result

    def visit_NonNullLiteral(self, op, *, value, dtype):
        """Compile a non-null literal differently than the default implementation.

        Most backends should implement this, but only when they need to handle
        some non-null literal differently than the default implementation
        (`visit_DefaultLiteral`).

        Return `None` from an override of this method to fall back to
        `visit_DefaultLiteral`.
        """
        return self.visit_DefaultLiteral(op, value=value, dtype=dtype)

    def visit_DefaultLiteral(self, op, *, value, dtype):
        """Compile a literal with a non-null value.

        This is the default implementation for compiling non-null literals.

        Most backends should not need to override this method unless they want
        to handle compiling every kind of non-null literal value.
        """
        if dtype.is_integer():
            return sge.convert(value)
        elif dtype.is_floating():
            if math.isnan(value):
                return self.NAN
            elif math.isinf(value):
                return self.POS_INF if value > 0 else self.NEG_INF
            return sge.convert(value)
        elif dtype.is_decimal():
            return self.cast(str(value), dtype)
        elif dtype.is_interval():
            return sge.Interval(
                this=sge.convert(str(value)),
                unit=sge.Var(this=dtype.resolution.upper()),
            )
        elif dtype.is_boolean():
            return sge.Boolean(this=bool(value))
        elif dtype.is_json():
            return sge.JSON(this=sge.convert(str(value)))
        elif dtype.is_string():
            return sge.convert(value)
        elif dtype.is_inet() or dtype.is_macaddr():
            return sge.convert(str(value))
        elif dtype.is_timestamp() or dtype.is_time():
            return self.cast(value.isoformat(), dtype)
        elif dtype.is_date():
            return self.f.datefromparts(value.year, value.month, value.day)
        elif dtype.is_array():
            value_type = dtype.value_type
            return self.f.array(
                *(
                    self.visit_Literal(
                        ops.Literal(v, value_type), value=v, dtype=value_type
                    )
                    for v in value
                )
            )
        elif dtype.is_map():
            key_type = dtype.key_type
            keys = self.f.array(
                *(
                    self.visit_Literal(
                        ops.Literal(k, key_type), value=k, dtype=key_type
                    )
                    for k in value.keys()
                )
            )

            value_type = dtype.value_type
            values = self.f.array(
                *(
                    self.visit_Literal(
                        ops.Literal(v, value_type), value=v, dtype=value_type
                    )
                    for v in value.values()
                )
            )

            return self.f.map(keys, values)
        elif dtype.is_struct():
            items = [
                self.visit_Literal(
                    ops.Literal(v, field_dtype), value=v, dtype=field_dtype
                ).as_(k, quoted=self.quoted)
                for field_dtype, (k, v) in zip(dtype.types, value.items())
            ]
            return sge.Struct.from_arg_list(items)
        elif dtype.is_uuid():
            return self.cast(str(value), dtype)
        elif dtype.is_geospatial():
            args = [value.wkt]
            if (srid := dtype.srid) is not None:
                args.append(srid)
            return self.f.st_geomfromtext(*args)

        raise NotImplementedError(f"Unsupported type: {dtype!r}")

    def visit_BitwiseNot(self, op, *, arg):
        return sge.BitwiseNot(this=arg)

    ### Mathematical Calisthenics

    def visit_E(self, op):
        return self.f.exp(1)

    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.ln(arg)
        elif str(base) in ("2", "10"):
            return self.f[f"log{base}"](arg)
        else:
            return self.f.ln(arg) / self.f.ln(base)

    def visit_Clip(self, op, *, arg, lower, upper):
        if upper is not None:
            arg = self.if_(arg.is_(NULL), arg, self.f.least(upper, arg))

        if lower is not None:
            arg = self.if_(arg.is_(NULL), arg, self.f.greatest(lower, arg))

        return arg

    def visit_FloorDivide(self, op, *, left, right):
        return self.cast(self.f.floor(sge.paren(left) / sge.paren(right)), op.dtype)

    def visit_Ceil(self, op, *, arg):
        return self.cast(self.f.ceil(arg), op.dtype)

    def visit_Floor(self, op, *, arg):
        return self.cast(self.f.floor(arg), op.dtype)

    def visit_Round(self, op, *, arg, digits):
        return self.cast(self.f.round(arg, digits), op.dtype)

    ### Dtype Dysmorphia

    def visit_TryCast(self, op, *, arg, to):
        return sge.TryCast(this=arg, to=self.type_mapper.from_ibis(to), safe=True)

    ### Comparator Conundrums

    def visit_Between(self, op, *, arg, lower_bound, upper_bound):
        return sge.Between(this=arg, low=lower_bound, high=upper_bound)

    def visit_Negate(self, op, *, arg):
        return -sge.paren(arg, copy=False)

    def visit_Not(self, op, *, arg):
        if isinstance(arg, sge.Filter):
            return sge.Filter(
                this=sg.not_(arg.this, copy=False), expression=arg.expression
            )
        return sg.not_(sge.paren(arg, copy=False))

    ### Timey McTimeFace

    def visit_Time(self, op, *, arg):
        return self.cast(arg, to=dt.time)

    def visit_TimestampNow(self, op):
        return sge.CurrentTimestamp()

    def visit_DateNow(self, op):
        return sge.CurrentDate()

    def visit_Strftime(self, op, *, arg, format_str):
        return sge.TimeToStr(this=arg, format=format_str)

    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.epoch(self.cast(arg, dt.timestamp))

    def visit_ExtractYear(self, op, *, arg):
        return self.f.extract(self.v.year, arg)

    def visit_ExtractMonth(self, op, *, arg):
        return self.f.extract(self.v.month, arg)

    def visit_ExtractDay(self, op, *, arg):
        return self.f.extract(self.v.day, arg)

    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.f.extract(self.v.dayofyear, arg)

    def visit_ExtractQuarter(self, op, *, arg):
        return self.f.extract(self.v.quarter, arg)

    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.extract(self.v.week, arg)

    def visit_ExtractHour(self, op, *, arg):
        return self.f.extract(self.v.hour, arg)

    def visit_ExtractMinute(self, op, *, arg):
        return self.f.extract(self.v.minute, arg)

    def visit_ExtractSecond(self, op, *, arg):
        return self.f.extract(self.v.second, arg)

    def visit_TimestampTruncate(self, op, *, arg, unit):
        unit_mapping = {
            "Y": "year",
            "Q": "quarter",
            "M": "month",
            "W": "week",
            "D": "day",
            "h": "hour",
            "m": "minute",
            "s": "second",
            "ms": "ms",
            "us": "us",
        }

        if (raw_unit := unit_mapping.get(unit.short)) is None:
            raise com.UnsupportedOperationError(
                f"Unsupported truncate unit {unit.short!r}"
            )

        return self.f.date_trunc(raw_unit, arg)

    def visit_DateTruncate(self, op, *, arg, unit):
        return self.visit_TimestampTruncate(op, arg=arg, unit=unit)

    def visit_TimeTruncate(self, op, *, arg, unit):
        return self.visit_TimestampTruncate(op, arg=arg, unit=unit)

    def visit_DayOfWeekIndex(self, op, *, arg):
        return (self.f.dayofweek(arg) + 6) % 7

    def visit_DayOfWeekName(self, op, *, arg):
        # day of week number is 0-indexed
        # Sunday == 0
        # Saturday == 6
        return sge.Case(
            this=(self.f.dayofweek(arg) + 6) % 7,
            ifs=list(itertools.starmap(self.if_, enumerate(calendar.day_name))),
        )

    def _make_interval(self, arg, unit):
        return sge.Interval(this=arg, unit=self.v[unit.singular])

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        return self._make_interval(arg, unit)

    ### String Instruments
    def visit_Strip(self, op, *, arg):
        return self.f.trim(arg, string.whitespace)

    def visit_RStrip(self, op, *, arg):
        return self.f.rtrim(arg, string.whitespace)

    def visit_LStrip(self, op, *, arg):
        return self.f.ltrim(arg, string.whitespace)

    def visit_LPad(self, op, *, arg, length, pad):
        return self.f.lpad(arg, self.f.greatest(self.f.length(arg), length), pad)

    def visit_RPad(self, op, *, arg, length, pad):
        return self.f.rpad(arg, self.f.greatest(self.f.length(arg), length), pad)

    def visit_Substring(self, op, *, arg, start, length):
        if isinstance(op.length, ops.Literal) and (value := op.length.value) < 0:
            raise com.IbisInputError(
                f"Length parameter must be a non-negative value; got {value}"
            )
        start += 1
        start = self.if_(start >= 1, start, start + self.f.length(arg))
        if length is None:
            return self.f.substring(arg, start)
        return self.f.substring(arg, start, length)

    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise com.UnsupportedOperationError(
                "String find doesn't support `end` argument"
            )

        if start is not None:
            arg = self.f.substr(arg, start + 1)
            pos = self.f.strpos(arg, substr)
            return self.if_(pos > 0, pos + start, 0)

        return self.f.strpos(arg, substr)

    def visit_RegexReplace(self, op, *, arg, pattern, replacement):
        return self.f.regexp_replace(arg, pattern, replacement, "g")

    def visit_StringConcat(self, op, *, arg):
        return self.f.concat(*arg)

    def visit_StringJoin(self, op, *, sep, arg):
        return self.f.concat_ws(sep, *arg)

    def visit_StringSQLLike(self, op, *, arg, pattern, escape):
        return arg.like(pattern)

    def visit_StringSQLILike(self, op, *, arg, pattern, escape):
        return arg.ilike(pattern)

    ### NULL PLAYER CHARACTER
    def visit_IsNull(self, op, *, arg):
        return arg.is_(NULL)

    def visit_NotNull(self, op, *, arg):
        return arg.is_(sg.not_(NULL, copy=False))

    def visit_InValues(self, op, *, value, options):
        return value.isin(*options)

    def visit_StringToTime(self, op, *, arg, format_str):
        return self.f.time(self.f.str_to_time(arg, format_str))

    ### Counting

    def visit_CountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    def visit_CountDistinctStar(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[STAR]), where=where)

    def visit_CountStar(self, op, *, arg, where):
        return self.agg.count(STAR, where=where)

    def visit_Kurtosis(self, op, *, arg, where, how: Literal["sample", "pop"]):
        if op.arg.dtype.is_boolean():
            arg = self.cast(arg, dt.int32)

        if how == "sample":
            return self.agg.kurtosis(arg, where=where)
        else:
            return self.agg.kurtosis_pop(arg, where=where)

    def visit_Sum(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.cast(arg, dt.int32)
        return self.agg.sum(arg, where=where)

    def visit_Mean(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            arg = self.cast(arg, dt.int32)
        return self.agg.avg(arg, where=where)

    def visit_Min(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            return self.agg.bool_and(arg, where=where)
        return self.agg.min(arg, where=where)

    def visit_Max(self, op, *, arg, where):
        if op.arg.dtype.is_boolean():
            return self.agg.bool_or(arg, where=where)
        return self.agg.max(arg, where=where)

    ### Stats

    def visit_VarianceStandardDevCovariance(self, op, *, how, where, **kw):
        hows = {"sample": "samp", "pop": "pop"}
        funcs = {
            ops.Variance: "var",
            ops.StandardDev: "stddev",
            ops.Covariance: "covar",
        }

        args = []

        for oparg, arg in zip(op.args, kw.values()):
            if (arg_dtype := oparg.dtype).is_boolean():
                arg = self.cast(arg, dt.Int32(nullable=arg_dtype.nullable))
            args.append(arg)

        funcname = f"{funcs[type(op)]}_{hows[how]}"
        return self.agg[funcname](*args, where=where)

    visit_Variance = visit_StandardDev = visit_Covariance = (
        visit_VarianceStandardDevCovariance
    )

    def visit_SimpleCase(self, op, *, base=None, cases, results, default):
        return sge.Case(
            this=base, ifs=list(map(self.if_, cases, results)), default=default
        )

    visit_SearchedCase = visit_SimpleCase

    def visit_ExistsSubquery(self, op, *, rel):
        select = rel.this.select(1, append=False)
        return self.f.exists(select)

    def visit_InSubquery(self, op, *, rel, needle):
        query = rel.this
        if not isinstance(query, sge.Select):
            query = sg.select(STAR).from_(query)
        return needle.isin(query=query)

    def visit_Array(self, op, *, exprs):
        return self.f.array(*exprs)

    def visit_StructColumn(self, op, *, names, values):
        return sge.Struct.from_arg_list(
            [value.as_(name, quoted=self.quoted) for name, value in zip(names, values)]
        )

    def visit_StructField(self, op, *, arg, field):
        return sge.Dot(this=arg, expression=sg.to_identifier(field, quoted=self.quoted))

    def visit_IdenticalTo(self, op, *, left, right):
        return sge.NullSafeEQ(this=left, expression=right)

    def visit_Greatest(self, op, *, arg):
        return self.f.greatest(*arg)

    def visit_Least(self, op, *, arg):
        return self.f.least(*arg)

    def visit_Coalesce(self, op, *, arg):
        return self.f.coalesce(*arg)

    ### Ordering and window functions

    def visit_SortKey(self, op, *, arg, ascending: bool, nulls_first: bool):
        return sge.Ordered(this=arg, desc=not ascending, nulls_first=nulls_first)

    def visit_ApproxMedian(self, op, *, arg, where):
        return self.agg.approx_quantile(arg, 0.5, where=where)

    def visit_WindowBoundary(self, op, *, value, preceding):
        # TODO: bit of a hack to return a dict, but there's no sqlglot expression
        # that corresponds to _only_ this information
        return {"value": value, "side": "preceding" if preceding else "following"}

    def visit_WindowFunction(self, op, *, how, func, start, end, group_by, order_by):
        if start is None:
            start = {}
        if end is None:
            end = {}

        start_value = start.get("value", "UNBOUNDED")
        start_side = start.get("side", "PRECEDING")
        end_value = end.get("value", "UNBOUNDED")
        end_side = end.get("side", "FOLLOWING")

        if getattr(start_value, "this", None) == "0":
            start_value = "CURRENT ROW"
            start_side = None

        if getattr(end_value, "this", None) == "0":
            end_value = "CURRENT ROW"
            end_side = None

        spec = sge.WindowSpec(
            kind=how.upper(),
            start=start_value,
            start_side=start_side,
            end=end_value,
            end_side=end_side,
            over="OVER",
        )
        order = sge.Order(expressions=order_by) if order_by else None

        spec = self._minimize_spec(op, spec)

        return sge.Window(this=func, partition_by=group_by, order=order, spec=spec)

    @staticmethod
    def _minimize_spec(op, spec):
        return spec

    def visit_LagLead(self, op, *, arg, offset, default):
        args = [arg]

        if default is not None:
            if offset is None:
                offset = 1

            args.append(offset)
            args.append(default)
        elif offset is not None:
            args.append(offset)

        return self.f[type(op).__name__.lower()](*args)

    visit_Lag = visit_Lead = visit_LagLead

    def visit_Argument(self, op, *, name: str, shape, dtype):
        return sg.to_identifier(op.param)

    def visit_RowID(self, op, *, table):
        return sg.column(
            op.name, table=table.alias_or_name, quoted=self.quoted, copy=False
        )

    # TODO(kszucs): this should be renamed to something UDF related
    def __sql_name__(self, op: ops.ScalarUDF | ops.AggUDF) -> str:
        # for builtin functions use the exact function name, otherwise use the
        # generated name to handle the case of redefinition
        funcname = (
            op.__func_name__
            if op.__input_type__ == InputType.BUILTIN
            else type(op).__name__
        )

        # not actually a table, but easier to quote individual namespace
        # components this way
        namespace = op.__udf_namespace__
        return sg.table(funcname, db=namespace.database, catalog=namespace.catalog).sql(
            self.dialect
        )

    def visit_ScalarUDF(self, op, **kw):
        return self.f[self.__sql_name__(op)](*kw.values())

    def visit_AggUDF(self, op, *, where, **kw):
        return self.agg[self.__sql_name__(op)](*kw.values(), where=where)

    def visit_TimestampDelta(self, op, *, part, left, right):
        # dialect is necessary due to sqlglot's default behavior
        # of `part` coming last
        return sge.DateDiff(
            this=left, expression=right, unit=part, dialect=self.dialect
        )

    visit_TimeDelta = visit_DateDelta = visit_TimestampDelta

    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        origin = self.f.cast("epoch", self.type_mapper.from_ibis(dt.timestamp))
        if offset is not None:
            origin += offset
        return self.f.time_bucket(interval, arg, origin)

    def visit_ArrayConcat(self, op, *, arg):
        return sge.ArrayConcat(this=arg[0], expressions=list(arg[1:]))

    ## relations

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        """Generate a valid name for a value expression.

        Override this method if the dialect has restrictions on valid
        identifiers even when quoted.

        See the BigQuery backend's implementation for an example.
        """
        return name

    def _cleanup_names(self, exprs: Mapping[str, sge.Expression]):
        """Compose `_gen_valid_name` and `_dedup_name` to clean up names in projections."""

        for name, value in exprs.items():
            name = self._gen_valid_name(name)
            if isinstance(value, sge.Column) and name == value.name:
                # don't alias columns that are already named the same as their alias
                yield value
            else:
                yield value.as_(name, quoted=self.quoted, copy=False)

    def visit_Select(
        self, op, *, parent, selections, predicates, qualified, sort_keys, distinct
    ):
        # if we've constructed a useless projection return the parent relation
        if not (selections or predicates or qualified or sort_keys or distinct):
            return parent

        result = parent

        if selections:
            # if there are `qualify` predicates then sqlglot adds a hidden
            # column to implement the functionality if the dialect doesn't
            # support it
            #
            # using STAR in that case would lead to an extra column, so in that
            # case we have to spell out the columns
            if op.is_star_selection() and (not qualified or self.supports_qualify):
                fields = [STAR]
            else:
                fields = self._cleanup_names(selections)
            result = sg.select(*fields, copy=False).from_(result, copy=False)

        if predicates:
            result = result.where(*predicates, copy=False)

        if qualified:
            result = result.qualify(*qualified, copy=False)

        if sort_keys:
            result = result.order_by(*sort_keys, copy=False)

        if distinct:
            result = result.distinct()

        return result

    def visit_DummyTable(self, op, *, values):
        return sg.select(*self._cleanup_names(values), copy=False)

    def visit_UnboundTable(
        self, op, *, name: str, schema: sch.Schema, namespace: ops.Namespace
    ) -> sg.Table:
        return sg.table(
            name, db=namespace.database, catalog=namespace.catalog, quoted=self.quoted
        )

    def visit_InMemoryTable(
        self, op, *, name: str, schema: sch.Schema, data
    ) -> sg.Table:
        return sg.table(name, quoted=self.quoted)

    def visit_DatabaseTable(
        self,
        op,
        *,
        name: str,
        schema: sch.Schema,
        source: Any,
        namespace: ops.Namespace,
    ) -> sg.Table:
        return sg.table(
            name, db=namespace.database, catalog=namespace.catalog, quoted=self.quoted
        )

    def visit_SelfReference(self, op, *, parent, identifier):
        return parent

    visit_JoinReference = visit_SelfReference

    def visit_JoinChain(self, op, *, first, rest, values):
        result = sg.select(*self._cleanup_names(values), copy=False).from_(
            first, copy=False
        )

        for link in rest:
            if isinstance(link, sge.Alias):
                link = link.this
            result = result.join(link, copy=False)
        return result

    def visit_JoinLink(self, op, *, how, table, predicates):
        sides = {
            "inner": None,
            "left": "left",
            "right": "right",
            "semi": "left",
            "anti": "left",
            "cross": None,
            "outer": "full",
            "asof": "asof",
            "any_left": "left",
            "any_inner": None,
            "positional": None,
        }
        kinds = {
            "any_left": "any",
            "any_inner": "any",
            "asof": "left",
            "inner": "inner",
            "left": "outer",
            "right": "outer",
            "semi": "semi",
            "anti": "anti",
            "cross": "cross",
            "outer": "outer",
            "positional": "positional",
        }
        assert predicates or how in {
            "cross",
            "positional",
        }, "expected non-empty predicates when not a cross join"
        on = sg.and_(*predicates) if predicates else None
        return sge.Join(this=table, side=sides[how], kind=kinds[how], on=on)

    @staticmethod
    def _generate_groups(groups):
        return map(sge.convert, range(1, len(groups) + 1))

    def visit_Aggregate(self, op, *, parent, groups, metrics):
        sel = sg.select(
            *self._cleanup_names(groups), *self._cleanup_names(metrics), copy=False
        ).from_(parent, copy=False)

        if groups:
            sel = sel.group_by(*self._generate_groups(groups.values()), copy=False)

        return sel

    @classmethod
    def _add_parens(cls, sg_expr):
        if type(sg_expr) in cls.SQLGLOT_NEEDS_PARENS:
            return sge.paren(sg_expr, copy=False)
        return sg_expr

    def visit_Union(self, op, *, left, right, distinct):
        if isinstance(left, (sge.Table, sge.Subquery)):
            left = sg.select(STAR, copy=False).from_(left, copy=False)

        if isinstance(right, (sge.Table, sge.Subquery)):
            right = sg.select(STAR, copy=False).from_(right, copy=False)

        return sg.union(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
            copy=False,
        )

    def visit_Intersection(self, op, *, left, right, distinct):
        if isinstance(left, (sge.Table, sge.Subquery)):
            left = sg.select(STAR, copy=False).from_(left, copy=False)

        if isinstance(right, (sge.Table, sge.Subquery)):
            right = sg.select(STAR, copy=False).from_(right, copy=False)

        return sg.intersect(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
            copy=False,
        )

    def visit_Difference(self, op, *, left, right, distinct):
        if isinstance(left, (sge.Table, sge.Subquery)):
            left = sg.select(STAR, copy=False).from_(left, copy=False)

        if isinstance(right, (sge.Table, sge.Subquery)):
            right = sg.select(STAR, copy=False).from_(right, copy=False)

        return sg.except_(
            left.args.get("this", left),
            right.args.get("this", right),
            distinct=distinct,
            copy=False,
        )

    def visit_Sample(
        self, op, *, parent, fraction: float, method: str, seed: int | None, **_
    ):
        sample = sge.TableSample(
            method="bernoulli" if method == "row" else "system",
            percent=sge.convert(fraction * 100.0),
            seed=None if seed is None else sge.convert(seed),
        )
        # sample was changed to be owned by the table being sampled in 25.17.0
        #
        # this is a small workaround for backwards compatibility
        if "this" in sample.__class__.arg_types:
            sample.args["this"] = parent
        else:
            parent.args["sample"] = sample
        return sg.select(STAR).from_(parent)

    def visit_Limit(self, op, *, parent, n, offset):
        # push limit/offset into subqueries
        if isinstance(parent, sge.Subquery) and parent.this.args.get("limit") is None:
            result = parent.this.copy()
            alias = parent.alias
        else:
            result = sg.select(STAR, copy=False).from_(parent, copy=False)
            alias = None

        if isinstance(n, int):
            result = result.limit(n, copy=False)
        elif n is not None:
            result = result.limit(
                sg.select(n, copy=False).from_(parent, copy=False).subquery(copy=False),
                copy=False,
            )
        else:
            assert n is None, n
            if self.no_limit_value is not None:
                result = result.limit(self.no_limit_value, copy=False)

        assert offset is not None, "offset is None"

        if not isinstance(offset, int):
            skip = offset
            skip = (
                sg.select(skip, copy=False)
                .from_(parent, copy=False)
                .subquery(copy=False)
            )
        elif not offset:
            if alias is not None:
                return result.subquery(alias, copy=False)
            return result
        else:
            skip = offset

        result = result.offset(skip, copy=False)
        if alias is not None:
            return result.subquery(alias, copy=False)
        return result

    def visit_CTE(self, op, *, parent):
        return sg.table(parent.alias_or_name, quoted=self.quoted)

    def visit_AliasedRelation(self, op, *, parent, name: str):
        if isinstance(parent, sge.Table):
            parent = sg.select(STAR, copy=False).from_(parent, copy=False)
        else:
            parent = parent.copy()

        if isinstance(parent, sge.Subquery):
            return parent.as_(name, quoted=self.quoted)
        else:
            try:
                return parent.subquery(name, copy=False)
            except AttributeError:
                return parent.as_(name, quoted=self.quoted)

    def visit_SQLStringView(self, op, *, query: str, parent, schema):
        return sg.parse_one(query, read=self.dialect)

    def visit_SQLQueryResult(self, op, *, query, schema, source):
        return sg.parse_one(query, dialect=self.dialect).subquery(copy=False)

    def binop(self, sg_cls, left, right):
        # If the op is associative we can skip parenthesizing ops of the same
        # type if they're on the left, since they would evaluate the same.
        # SQLGlot has an optimizer for generating long sql chains of the same
        # op of this form without recursion, by avoiding parenthesis in this
        # common case we can make use of this optimization to handle large
        # operator chains.
        if not (sg_cls in self.SQLGLOT_ASSOCIATIVE_OPS and type(left) is sg_cls):
            left = self._add_parens(left)
        return sg_cls(this=left, expression=self._add_parens(right))

    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(
            f"Compilation rule for {type(op).__name__!r} operation is not defined"
        )

    def visit_Unsupported(self, op, **_):
        raise com.UnsupportedOperationError(
            f"{type(op).__name__!r} operation is not supported in the {self.dialect} backend"
        )

    def visit_DropColumns(self, op, *, parent, columns_to_drop):
        # the generated query will be huge for wide tables
        #
        # TODO: figure out a way to produce an IR that only contains exactly
        # what is used
        parent_alias = parent.alias_or_name
        quoted = self.quoted
        columns_to_keep = (
            sg.column(column, table=parent_alias, quoted=quoted)
            for column in op.schema.names
        )
        return sg.select(*columns_to_keep).from_(parent)

    def add_query_to_expr(self, *, name: str, table: ir.Table, query: str) -> str:
        dialect = self.dialect

        compiled_ibis_expr = self.to_sqlglot(table)
        compiled_query = sg.parse_one(query, read=dialect)

        ctes = [
            *compiled_ibis_expr.ctes,
            sge.CTE(
                alias=sg.to_identifier(name, quoted=self.quoted),
                this=compiled_ibis_expr,
            ),
            *compiled_query.ctes,
        ]
        compiled_ibis_expr.args.pop(WITH_ARG, None)
        compiled_query.args.pop(WITH_ARG, None)

        # pull existing CTEs from the compiled Ibis expression and combine them
        # with the new query
        parsed = reduce(
            lambda parsed, cte: parsed.with_(cte.args["alias"], as_=cte.args["this"]),
            ctes,
            compiled_query,
        )

        # generate the SQL string
        return parsed.sql(dialect)

    def _make_sample_backwards_compatible(self, *, sample, parent):
        # sample was changed to be owned by the table being sampled in 25.17.0
        #
        # this is a small workaround for backwards compatibility
        if "this" in sample.__class__.arg_types:
            sample.args["this"] = parent
        else:
            parent.args["sample"] = sample
        return sg.select(STAR).from_(parent)


# `__init_subclass__` is uncalled for subclasses - we manually call it here to
# autogenerate the base class implementations as well.
SQLGlotCompiler.__init_subclass__()
