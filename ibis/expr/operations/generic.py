"""Generic value operations."""

from __future__ import annotations

import itertools
from typing import Annotated, Any, Optional
from typing import Literal as LiteralType

from public import public
from typing_extensions import TypeVar

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.deferred import Deferred
from ibis.common.grounds import Singleton
from ibis.common.patterns import InstanceOf, Length
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.operations.core import Scalar, Unary, Value
from ibis.expr.operations.relations import Relation  # noqa: TC001


@public
class RowID(Value):
    """The row number of the returned result."""

    name = "rowid"
    table: Relation

    shape = ds.columnar
    dtype = dt.int64

    @attribute
    def relations(self):
        return frozenset({self.table})


@public
class Cast(Value):
    """Explicitly cast a value to a specific data type."""

    arg: Value
    to: dt.DataType

    shape = rlz.shape_like("arg")

    @property
    def name(self):
        return f"{self.__class__.__name__}({self.arg.name}, {self.to})"

    @property
    def dtype(self):
        return self.to


@public
class TryCast(Value):
    """Try to cast a value to a specific data type."""

    arg: Value
    to: dt.DataType

    shape = rlz.shape_like("arg")

    @property
    def dtype(self):
        return self.to


@public
class TypeOf(Unary):
    """Return the _database_ data type of the input expression."""

    dtype = dt.string


@public
class IsNull(Unary):
    """Return true if values are null."""

    dtype = dt.boolean


@public
class NotNull(Unary):
    """Returns true if values are not null."""

    dtype = dt.boolean


@public
class NullIf(Value):
    """Return NULL if an expression equals some specific value."""

    arg: Value
    null_if_expr: Value

    dtype = rlz.dtype_like("args")
    shape = rlz.shape_like("args")


@public
class Coalesce(Value):
    """Return the first non-null expression from a tuple of expressions."""

    arg: Annotated[VarTuple[Value], Length(at_least=1)]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class Greatest(Value):
    """Return the largest value from a tuple of expressions."""

    arg: Annotated[VarTuple[Value], Length(at_least=1)]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class Least(Value):
    """Return the smallest value from a tuple of expressions."""

    arg: Annotated[VarTuple[Value], Length(at_least=1)]

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


T = TypeVar("T", bound=dt.DataType, covariant=True)


@public
class Literal(Scalar[T]):
    """A constant value."""

    value: Annotated[Any, ~InstanceOf(Deferred)]
    dtype: T

    shape = ds.scalar

    def __init__(self, value, dtype):
        # normalize ensures that the value is a valid value for the given dtype
        value = dt.normalize(dtype, value)
        super().__init__(value=value, dtype=dtype)

    @property
    def name(self):
        if self.dtype.is_interval():
            return f"{self.value!r}{self.dtype.unit.short}"
        return repr(self.value)


NULL = Literal(None, dt.null)

# ScalarParameter 是 Ibis 中代表标量参数（Scalar Parameter）的操作节点类。
# 在编写 SQL 或进行数据查询时，我们经常需要编写“占位符”，等真正执行查询时再动态传入具体的值。例如在 SQL 中写的 :val 或 %s。
# ScalarParameter 就扮演了这个“占位符”的角色：
# 未绑定的占位符：它允许你在不知道具体数值的情况下，先定义一个具有特定数据类型（如 int64、string）的参数节点，并用它参与复杂的表达式构建（例如 table.filter(table.age > ibis.param(dt.int64))）。
# 唯一标识性：为了防止多个参数在编译时混淆，每一个 ScalarParameter 节点在被创建时都会分配一个全局唯一的计数器编号，从而自动生成形如 param_0、param_1 这样唯一的参数名称。
# 延迟赋值编译：在最终把 Ibis 表达式编译为 SQL（如 Postgres、DuckDB SQL）时，它会被翻译为对应后端的参数占位符，并在 .execute(params={...}) 时将真实的值安全地注入进去（防止 SQL 注入）。
@public
class ScalarParameter(Scalar):
    # 全局的、线程安全的自动递增计数器（利用了 Python 标准库中的 itertools.count）
    # 是类属性（Class Attribute），所有 ScalarParameter 的实例都会共享同一个计数器。每当有新的参数节点被创建且没有手动指定 counter 时，
    # 它就会调用 next(self._counter) 产生一个新的整数（$0, 1, 2, \dots$），以此确保每一个参数节点的唯一性。
    _counter = itertools.count()
    # 声明参数的 Ibis 数据类型（Data Type）
    dtype: dt.DataType
    # 存储当前参数实例的唯一标识序号
    counter: Optional[int] = None

    shape = ds.scalar

    def __init__(self, dtype, counter):
        if counter is None:
            counter = next(self._counter)
        super().__init__(dtype=dtype, counter=counter)

    @property
    def name(self):
        return f"param_{self.counter:d}"


@public
class Constant(Scalar, Singleton):
    """A function that produces a constant."""

    shape = ds.scalar


@public
class Impure(Value):
    pass


@public
class TimestampNow(Impure):
    """Return the current timestamp."""

    dtype = dt.timestamp
    shape = ds.scalar


@public
class DateNow(Impure):
    """Return the current date."""

    dtype = dt.date
    shape = ds.scalar


@public
class RandomScalar(Impure):
    """Return a random scalar between 0 and 1."""

    dtype = dt.float64
    shape = ds.scalar


@public
class RandomUUID(Impure):
    """Return a random UUID."""

    dtype = dt.uuid
    shape = ds.scalar


@public
class E(Constant):
    """The mathematical constant e."""

    dtype = dt.float64


@public
class Pi(Constant):
    """The mathematical constant pi."""

    dtype = dt.float64


@public
class Hash(Value):
    """Return the hash of a value."""

    arg: Value

    dtype = dt.int64
    shape = rlz.shape_like("arg")


@public
class HashBytes(Value):
    arg: Value[dt.String | dt.Binary]
    how: LiteralType[
        "md5",
        "MD5",
        "sha1",
        "SHA1",
        "SHA224",
        "sha256",
        "SHA256",
        "sha512",
        "intHash32",
        "intHash64",
        "cityHash64",
        "sipHash64",
        "sipHash128",
    ]

    dtype = dt.binary
    shape = rlz.shape_like("arg")


@public
class HexDigest(Value):
    """Return the hexadecimal digest of a value."""

    arg: Value[dt.String | dt.Binary]
    how: LiteralType[
        "md5",
        "sha1",
        "sha256",
        "sha512",
    ]

    dtype = dt.str
    shape = rlz.shape_like("arg")


# TODO(kszucs): we should merge the case operations by making the
# cases, results and default optional arguments like they are in
# api.py
@public
class SimpleCase(Value):
    """Simple case statement."""

    base: Value
    cases: Annotated[VarTuple[Value], Length(at_least=1)]
    results: Annotated[VarTuple[Value], Length(at_least=1)]
    default: Value

    def __init__(self, base, cases, results, default):
        assert len(cases) == len(results)
        for case in cases:
            if not rlz.comparable(base, case):
                raise TypeError(
                    f"Base expression {rlz.arg_type_error_format(base)} and "
                    f"case {rlz.arg_type_error_format(case)} are not comparable"
                )
        super().__init__(base=base, cases=cases, results=results, default=default)

    @attribute
    def shape(self):
        exprs = [self.base, *self.cases, *self.results, self.default]
        return rlz.highest_precedence_shape(exprs)

    @attribute
    def dtype(self):
        values = [*self.results, self.default]
        return rlz.highest_precedence_dtype(values)


@public
class SearchedCase(Value):
    """Searched case statement."""

    cases: Annotated[VarTuple[Value[dt.Boolean]], Length(at_least=1)]
    results: Annotated[VarTuple[Value], Length(at_least=1)]
    default: Value

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        super().__init__(cases=cases, results=results, default=default)

    @attribute
    def shape(self):
        return rlz.highest_precedence_shape((*self.cases, *self.results, self.default))

    @attribute
    def dtype(self):
        exprs = [*self.results, self.default]
        return rlz.highest_precedence_dtype(exprs)


public(NULL=NULL)
