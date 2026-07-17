from __future__ import annotations

from typing import Any

from public import public

from ibis.common.grounds import Singleton

# 在数据分析和编译器（如 Ibis 将 Python 表达式翻译为 SQL）中，知道一个表达式返回的数据“形状”（是单值、一列，还是一张表）至关重要。datashape.py 正是为此而生的。
# DataShape（数据形状）类及其子类在 Ibis 中扮演着维度元数据（Dimensionality Metadata）的角色。它用来描述一个 Ibis 表达式计算结果的维度（Dimensionality / Shape）。
# 在 Ibis 中，数据形状被严格划分为三种：
# Scalar（标量）：维度为 0。代表一个单一的值，比如数字 42、字符串 "hello"、或者聚合计算的结果 max(列)。
# Columnar（列/向量）：维度为 1。代表单列数据，比如表中的某一列（如 table.age），它有多个元素，但只有一维。
# Tabular（表/矩阵）：维度为 2。代表一个完整的二维关系表（拥有行和列，如整个 table）。
# @public 是一个来自于第三方库 atpublic（通过 from public import public 导入）的装饰器。
# 核心作用非常简单而优雅：自动将类或函数添加到当前模块的 __all__ 导出列表中。
@public
class DataShape(Singleton):
    # 表示当前数据形状的维度值（Number of Dimensions）,标量（Scalar）为 0,一维列（Columnar）为 1
    ndim: int
    # 类属性。指向全局唯一的 Scalar 实例（用于向后兼容旧代码）。
    SCALAR: Scalar
    #类属性。指向全局唯一的 Columnar 实例。
    COLUMNAR: Columnar
    # 判断当前形状是否为标量（维度为 0）。
    def is_scalar(self) -> bool:
        return self.ndim == 0
    # 判断当前形状是否为列数据（维度为 1）。
    def is_columnar(self) -> bool:
        return self.ndim == 1
    # 判断当前形状是否为表格数据（维度为 2）。
    def is_tabular(self) -> bool:
        return self.ndim == 2
    # 定义“小于”（Less Than）比较运算符（<）。
    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DataShape):
            return NotImplemented
        return self.ndim < other.ndim

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, DataShape):
            return NotImplemented
        return self.ndim <= other.ndim

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataShape):
            return NotImplemented
        return self.ndim == other.ndim

    def __hash__(self) -> int:
        return hash((self.__class__, self.ndim))


@public
class Scalar(DataShape):
    ndim = 0


@public
class Columnar(DataShape):
    ndim = 1


@public
class Tabular(DataShape):
    ndim = 2


# for backward compat
DataShape.SCALAR = Scalar()
DataShape.COLUMNAR = Columnar()
DataShape.TABULAR = Tabular()

scalar = Scalar()
columnar = Columnar()
tabular = Tabular()


public(Any=DataShape)
