"""Operations for analytic window functions."""

from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations.core import Column, Scalar, Value

# Analytic 也是表达式 IR（中间表示）图中的一个重要基类
# Analytic 是 Ibis 中所有分析型窗口函数操作（Analytic Window Function Operations）的抽象基类。
# 窗口函数的抽象基底：在 SQL 和数据分析框架中，分析函数（如 row_number(), rank(), lead(), lag(), first_value() 等）通常需要配合窗口（OVER (PARTITION BY ... ORDER BY ...)）使用。
# Analytic 类作为所有这类操作节点的共同基类，在底层标记并规范它们的行为。
# 继承链与类型归属：它继承自 Value，表明每一个分析函数计算后都会产生一个具体的数据值（即可以在表达式中像列一样被引用或计算）。
# 数据形状（Data Shape）约束：它明确定义了该类操作计算后产出的数据维度形状，为上层分析和底层 SQL 编译提供类型检查支持。
@public
class Analytic(Value):
    """Base class for analytic window function operations."""
    # 类属性，赋值为 ds.columnar（来自于 ibis.expr.datashape 模块）。
    # 定该操作计算结果的数据形状（Data Shape）为列状/向量（Columnar）
    shape = ds.columnar


class ShiftBase(Analytic):
    """Base class for shift operations."""

    arg: Column[dt.Any]
    offset: Optional[Value[dt.Integer | dt.Interval]] = None
    default: Optional[Value] = None

    dtype = rlz.dtype_like("arg")


@public
class Lag(ShiftBase):
    """Shift a column forward."""


@public
class Lead(ShiftBase):
    """Shift a column backward."""


@public
class RankBase(Analytic):
    """Base class for ranking operations."""

    dtype = dt.int64


@public
class MinRank(RankBase):
    """Rank within an ordered partition."""


@public
class DenseRank(RankBase):
    """Rank within an ordered partition, consecutively."""


@public
class RowNumber(RankBase):
    """Compute the row number over a window, starting from 0."""


@public
class PercentRank(Analytic):
    """Compute the percentile rank over a window."""

    dtype = dt.double


@public
class CumeDist(Analytic):
    """Compute the cumulative distribution function of a column over a window."""

    dtype = dt.double


@public
class NTile(Analytic):
    """Compute the percentile of a column over a window."""

    buckets: Scalar[dt.Integer]

    dtype = dt.int64


@public
class NthValue(Analytic):
    """Retrieve the Nth element of a column over a window."""

    arg: Column[dt.Any]
    nth: Value[dt.Integer]

    dtype = rlz.dtype_like("arg")


public(AnalyticOp=Analytic)
