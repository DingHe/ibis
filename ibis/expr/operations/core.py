from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Optional

from public import public
from typing_extensions import Any, Self, TypeVar

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.graph import Node as Traversable
from ibis.common.grounds import Concrete
from ibis.common.patterns import Coercible, CoercionError
from ibis.common.typing import DefaultTypeVars
from ibis.util import is_iterable


@public
class Node(Concrete, Traversable):
    def equals(self, other) -> bool:
        if not isinstance(other, Node):
            raise TypeError(
                f"invalid equality comparison between Node and {type(other)}"
            )
        return self == other

    # Avoid custom repr for performance reasons
    __repr__ = object.__repr__

    # TODO(kszucs): hidrate the __children__ traversable attribute
    # @attribute
    # def __children__(self):
    #     return super().__children__

# 利用 Python 的 typing 模块定义了在表达式树中描述数据类型（DataType）和数据形状（DataShape）的类型变量（Type Variables）。
# T: 代表“数据类型”（DataType）
# bound=dt.DataType: 设定了上限约束。任何使用 T 的地方，传入的必须是 ibis.expr.datatypes.DataType 的子类（如 Int64, String, Timestamp 等）。这确保了类型安全，防止误传入非类型对象。
# covariant=True（协变）: 这是泛型设计的关键。它允许子类型关系保持一致。例如，如果 Int64 是 DataType 的子类，那么 Expression[Int64] 也被视为 Expression[DataType] 的子类。
# 这在处理复杂的表达式嵌套时非常重要。
T = TypeVar("T", bound=dt.DataType, covariant=True)
# S: 代表“数据形状”（DataShape，即标量、列或表的维度属性）。
S = TypeVar("S", bound=ds.DataShape, default=ds.Any, covariant=True)

# Value 类是表达式系统的核心基类。它连接了底层的“运算逻辑”（AST 节点）与上层的“用户 API”（如 table.column 或 lit(1)）
# Node 代表了“图中的一个运算步骤”，而 Value 则代表了“该运算步骤产生的一个具有类型（DataType）和形状（DataShape）的数据结果”。
# 统一类型化接口：确保所有运算节点都具备明确的 dtype（如 Int64）和 shape（如 Scalar 或 Column）。
# 自动类型强制转换（Coercion）：实现了 Coercible 协议，使得用户可以直接在函数参数中传入原生 Python 类型（如 int, str），系统会自动将其提升为 Literal 表达式节点。
# 表达式桥梁：将内部的计算节点（Value）封装并暴露为用户友好的 Ibis 表达式对象（Expr），实现底层 AST 到用户 API 的无缝转换。
@public
class Value(Node, Coercible, DefaultTypeVars, Generic[T, S]):
    # Value.__coerce__ 是一个类型转换协议（Coercion Protocol）的实现。
    # 核心作用是将用户传入的“任意格式的数据”（如原生 Python 类型、已有的表达式对象）标准化为 Ibis 内部统一的 Literal（字面量）运算节点
    # cls: 当前类（Value），用于返回该类的一个有效实例。
    # value: Any: 待转换的输入。它可以是：
    # Python 原生类型: 如 1, 'string', True。
    # Ibis 公共 API 对象: 如 ibis.literal(1) 返回的 Expr 对象。
    # Ibis 内部节点: 已经是 Value 类型的 Node 实例。
    @classmethod
    def __coerce__(
        cls, value: Any, T: Optional[type] = None, S: Optional[type] = None
    ) -> Self:
        # note that S=Shape is unused here since the pattern will check the
        # shape of the value expression after executing Value.__coerce__()
        from ibis.expr.operations.generic import NULL, Literal
        from ibis.expr.types import Expr
        # 解包 API 层。
        # 用户操作的是 Expr 对象（API 外壳），而 Ibis 内部逻辑处理的是底层的计算节点 Node（通过 .op() 获取）。
        if isinstance(value, Expr):
            value = value.op()
        # 去重与复用。
        # 如果输入本身已经是 Value 节点，直接返回，无需重复创建。
        # 特例是处理 NULL 常量，将其重置为 None，以便在后续步骤中重新绑定 T 指定的特定数据类型（例如将 NULL 显式转换为 Int64 类型的空值）。
        if isinstance(value, Value):
            if value == NULL:
                # treat the NULL literal the same as None to implicitly cast to
                # the requested datatype if any
                value = None
            else:
                return value
        # 如果提供了类型提示 T（如 Integer），则执行强制转换（int(value)）并推导类型。
        if T is dt.Integer:
            dtype = dt.infer(int(value))
        elif T is dt.Floating:
            dtype = dt.infer(float(value))
        else:
            try:
                dtype = dt.DataType.from_typehint(T)
            except TypeError:
                # 如果未提供或转换失败，调用 dt.infer(value)，利用 Ibis 的类型系统自动感知 Python 值的类型（例如 str 会被识别为 dt.String）。
                dtype = dt.infer(value)

        try:
            return Literal(value, dtype=dtype)
        except TypeError:
            raise CoercionError(f"Unable to coerce {value!r} to Value[{T!r}]")

    # TODO(kszucs): cover it with tests
    # TODO(kszucs): figure out how to represent not named arguments
    # 核心目的是为每一个复杂的表达式节点生成一个可读的、人类友好的字符串标识符。
    @property
    def name(self) -> str:
        names = []
        for arg in self.__args__:
            if is_iterable(arg):
                elements = [
                    element_name
                    for element in arg
                    if (element_name := getattr(element, "name", None)) is not None
                ]
                joined = ", ".join(elements)
                fmt = "({})" if len(elements) != 1 else "({},)"
                names.append(fmt.format(joined))
            elif (name := getattr(arg, "name", None)) is not None:
                names.append(name)
        return f"{self.__class__.__name__}({', '.join(names)})"

    @property
    @abstractmethod
    def dtype(self) -> T:
        """Ibis datatype of the produced value expression.

        Returns
        -------
        dt.DataType

        """

    @property
    @abstractmethod
    def shape(self) -> S:
        """Shape of the produced value expression.

        Possible values are: "scalar" and "columnar"

        Returns
        -------
        ds.Shape

        """
    # 核心作用是自动递归解析当前表达式依赖的所有“表（Relation）”，为 Ibis 生成 SQL 的 FROM 子句提供数据源信息。
    # 在 Ibis 中，复杂的表达式（如 (table_a.col1 + table_b.col2).sum()）往往跨越多个数据源。
    # relations 属性通过递归遍历表达式树，汇总该节点所引用的所有底层 Relation（通常是 Table 或 TableExpr 的底层节点）。
    @attribute
    def relations(self):
        """Set of relations the value node depends on."""
        children = (n.relations for n in self.__children__ if isinstance(n, Value))
        return frozenset().union(*children)
    # 作用是将底层的计算节点（Node）“升维”封装为用户可交互的表达式对象（Expr）。
    def to_expr(self):
        import ibis.expr.types as ir
        # 形状判断：区分“列运算”与“标量运算”
        # 列式运算（Columnar）：如果表达式依赖于表中的一整列（例如 table.col），其形状为 Column。Ibis 会查找该数据类型对应的列类名（如 Int64Column）。
        # 标量运算（Scalar）：如果表达式是一个固定的值或单行聚合结果（例如 1 或 table.col.sum()），其形状为 Scalar。Ibis 会查找该类型对应的标量类名（如 Int64Scalar）。
        # 动态映射：self.dtype.column 和 self.dtype.scalar 是 DataType 对象预定义的属性，存储了该类型对应的类名字符串（例如 Int64 类型对应的字符串是 "Int64Column" 和 "Int64Scalar"）。
        if self.shape.is_columnar():
            typename = self.dtype.column
        else:
            typename = self.dtype.scalar
        # 反射调用：利用 getattr(ir, typename) 从 types 模块中动态获取类，并传入 self（当前的底层计算节点）进行初始化。
        return getattr(ir, typename)(self)


# convenience aliases
Scalar = Value[T, ds.Scalar]
Column = Value[T, ds.Columnar]

# 代表了 SQL 中的 “别名”操作（即 AS 子句）。它是 Value 类的具体子类。
# 当你在 Ibis 中执行类似 table.col.name("new_col_name") 的操作时，底层就会创建一个 Alias 节点。
# 重命名节点：为某个表达式（arg）显式赋予一个新的字符串名称（name），用于最终生成的 SQL 语句中（如 SELECT col AS new_col_name）。
@public
class Alias(Value):
    # 被赋予别名的原始表达式节点（如一个列、一个函数计算结果或一个常量）。
    arg: Value
    # 新的别名字符串。
    name: str
    # 形状联动规则。
    # 声明 Alias 节点的 shape 属性必须与它内部的 "arg" 字段完全一致
    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class Unary(Value):
    """A unary operation."""

    arg: Value

    @attribute
    def shape(self) -> ds.DataShape:
        return self.arg.shape

    @attribute
    def relations(self):
        return self.arg.relations


@public
class Binary(Value):
    """A binary operation."""

    left: Value
    right: Value

    @attribute
    def shape(self) -> ds.DataShape:
        return max(self.left.shape, self.right.shape)

    @attribute
    def relations(self):
        return self.left.relations | self.right.relations


@public
class Argument(Value):
    name: str
    shape: ds.DataShape
    dtype: dt.DataType

    @attribute
    def param(self) -> str:
        return f"__ibis_param_{self.name}__"


public(ValueOp=Value, UnaryOp=Unary, BinaryOp=Binary, Scalar=Scalar, Column=Column)
