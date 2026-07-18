from __future__ import annotations

from public import public

from ibis.expr.types.generic import Column, Scalar, Value

# UUIDValue 是 Ibis 表达式树中 UUID 类型数据的基类。它的核心作用是：
# 向 Ibis 系统声明：“此表达式及其计算结果必须符合 UUID 数据类型”。
# 确保只有适用于 UUID 的操作（如字符串转换、特定数据库函数）可以作用于这些对象。
# 它继承自 Value，因此拥有 Value 类所有的基础操作（如 isnull()、cast()、name() 等）。
@public
class UUIDValue(Value):
    pass


@public
class UUIDScalar(Scalar, UUIDValue):
    pass


@public
class UUIDColumn(Column, UUIDValue):
    pass
