from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from ibis.util import PseudoHashable, indent

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    from ibis.expr.datatypes import DataType
    from ibis.expr.schema import Schema

C = TypeVar("C")
T = TypeVar("T")
S = TypeVar("S")

# TypeMapper[T] 类的核心作用包括：
# 类型双向桥接（Type Bridging）：
# 在 Ibis 标准类型（dt.DataType） 和 特定后端/库的原生类型对象 T（如 pyarrow.DataType、duckdb.typing.DuckDBPyType 或 SQL/C 类型对象）之间进行双向转换。
# 字符串与类型映射（SQL/Schema String Parsing）：
# 将后端特定的类型字符串（如数据库导出的 "VARCHAR(255)"、"TIMESTAMP WITH TIME ZONE" 等）解析为 Ibis 的 DataType，或者反向输出为符合目标后端语法的类型字符串。
# 标准化规范与接口契约（Interface Contract）：
# 作为泛型抽象基类（Generic[T]），它定义了所有后端类型映射器必须遵循的标准 API 接口，保证了不同后端实现在框架内部调用方式的高度统一。
# 1. 泛型类型属性：T 继承自 Python typing.Generic[T]。这里的 T 代表特定后端/数据格式原生的类型对象（Format-specific Type Object）。
class TypeMapper(Generic[T]):
    # `T` is the format-specific type object, e.g. pyarrow.DataType

    # 将 Ibis 统一的标准类型对象 dtype 转换为目标后端专用的原生类型对象 T。
    # dtype (DataType): 需要转换的 Ibis 标准数据类型（例如 dt.string、dt.int64）。
    # T: 转换后目标后端特定的原生类型对象。
    @classmethod
    def from_ibis(cls, dtype: DataType) -> T:
        """Convert an Ibis DataType to a format-specific type object.

        Parameters
        ----------
        dtype
            The Ibis DataType to convert.

        Returns
        -------
        Format-specific type object.

        """
        raise NotImplementedError

    # 基本作用：将目标后端原生的类型对象 typ 反向转换为 Ibis 标准的 DataType 对象。
    # typ (T): 待转换的目标后端原生类型对象（例如 pa.int64()）。
    # nullable (bool，默认值为 True): 指定生成的 Ibis DataType 是否允许包含空值（NULL）。
    # DataType: 转换后的 Ibis 标准数据类型对象。
    # 当从后端（如 DuckDB、Polars 或 Arrow 表）读取数据表 Schema 时，Ibis 调用此方法将后端的原生类型映射回统一的 Ibis 类型，供用户编写跨平台代码。
    @classmethod
    def to_ibis(cls, typ: T, nullable: bool = True) -> DataType:
        """Convert a format-specific type object to an Ibis DataType.

        Parameters
        ----------
        typ
            The format-specific type object to convert.
        nullable
            Whether the Ibis DataType should be nullable.

        Returns
        -------
        Ibis DataType.

        """
        raise NotImplementedError
    # 解析特定后端导出的文本/字符串形式的数据类型表示，将其转换为 Ibis 的 DataType。
    # text (str): 目标数据库/后端特有的类型描述文本（例如 "VARCHAR(255)"、"TIMESTAMP_NTZ"）。
    # nullable (bool，默认值为 True): 生成的 Ibis DataType 是否标记为可空。
    # 用于解析数据库 DDL、返回的元数据（如 JDBC/ODBC 驱动返回的类型名称字段）或用户输入的后端专属类型名称。
    @classmethod
    def from_string(cls, text: str, nullable: bool = True) -> DataType:
        """Convert a backend-specific string representation into an Ibis DataType.

        Parameters
        ----------
        text
            The backend-specific string representation to convert.
        nullable
            Whether the Ibis DataType should be nullable.

        Returns
        -------
        Ibis DataType.

        """
        raise NotImplementedError

    # 将 Ibis 的 DataType 转换为目标后端 SQL/DDL 或语法中可用的类型文本字符串。
    # dtype (DataType): 需要转换的 Ibis 标准数据类型。
    # str: 符合目标后端语法规范的类型描述字符串（例如 PostgreSQL 的 "text" 或 Snowflake 的 "NUMBER(38, 0)"）。
    @classmethod
    def to_string(cls, dtype: DataType) -> str:
        """Convert `dtype` into a backend-specific string representation.

        Parameters
        ----------
        dtype
            The Ibis DataType to convert.

        Returns
        -------
        Backend-specific string representation.

        """
        raise NotImplementedError


class SchemaMapper(Generic[S]):
    # `S` is the format-specific schema object, e.g. pyarrow.Schema

    @classmethod
    def from_ibis(cls, schema: Schema) -> S:
        """Convert an Ibis Schema to a format-specific schema object.

        Parameters
        ----------
        schema
            The Ibis Schema to convert.

        Returns
        -------
        Format-specific schema object.

        """
        raise NotImplementedError

    @classmethod
    def to_ibis(cls, obj: S) -> Schema:
        """Convert a format-specific schema object to an Ibis Schema.

        Parameters
        ----------
        obj
            The format-specific schema object to convert.

        Returns
        -------
        Ibis Schema.

        """
        raise NotImplementedError


class DataMapper(Generic[S, C, T]):
    # `S` is the format-specific scalar object, e.g. pyarrow.Scalar
    # `C` is the format-specific column object, e.g. pyarrow.Array
    # `T` is the format-specific table object, e.g. pyarrow.Table

    @classmethod
    def convert_scalar(cls, obj: S, dtype: DataType) -> S:
        """Convert a format-specific scalar to the given ibis datatype.

        Parameters
        ----------
        obj
            The format-specific scalar value to convert.
        dtype
            The Ibis datatype to convert to.

        Returns
        -------
        Format specific scalar corresponding to the given Ibis datatype.

        """
        raise NotImplementedError

    @classmethod
    def convert_column(cls, obj: C, dtype: DataType) -> C:
        """Convert a format-specific column to the given ibis datatype.

        Parameters
        ----------
        obj
            The format-specific column value to convert.
        dtype
            The Ibis datatype to convert to.

        Returns
        -------
        Format specific column corresponding to the given Ibis datatype.

        """
        raise NotImplementedError

    @classmethod
    def convert_table(cls, obj: T, schema: Schema) -> T:
        """Convert a format-specific table to the given ibis schema.

        Parameters
        ----------
        obj
            The format-specific table-like object to convert.
        schema
            The Ibis schema to convert to.

        Returns
        -------
        Format specific table-like object corresponding to the given Ibis schema.

        """
        raise NotImplementedError

    @classmethod
    def infer_scalar(cls, obj: S) -> DataType:
        """Infer the Ibis datatype of a format-specific scalar.

        Parameters
        ----------
        obj
            The format-specific scalar to infer the Ibis datatype of.

        Returns
        -------
        Ibis datatype corresponding to the given format-specific scalar.

        """
        raise NotImplementedError

    @classmethod
    def infer_column(cls, obj: C) -> DataType:
        """Infer the Ibis datatype of a format-specific column.

        Parameters
        ----------
        obj
            The format-specific column to infer the Ibis datatype of.

        Returns
        -------
        Ibis datatype corresponding to the given format-specific column.

        """
        raise NotImplementedError

    @classmethod
    def infer_table(cls, obj: T) -> Schema:
        """Infer the Ibis schema of a format-specific table.

        Parameters
        ----------
        obj
            The format-specific table to infer the Ibis schema of.

        Returns
        -------
        Ibis schema corresponding to the given format-specific table.

        """
        raise NotImplementedError


class TableProxy(PseudoHashable[T]):
    def __repr__(self) -> str:
        data_repr = indent(repr(self.obj), spaces=2)
        return f"{self.__class__.__name__}:\n{data_repr}"

    @abstractmethod
    def to_frame(self) -> pd.DataFrame:  # pragma: no cover
        """Convert this input to a pandas DataFrame."""

    @abstractmethod
    def to_pyarrow(self, schema: Schema) -> pa.Table:  # pragma: no cover
        """Convert this input to a PyArrow Table."""

    @abstractmethod
    def to_polars(self, schema: Schema) -> pl.DataFrame:  # pragma: no cover
        """Convert this input to a Polars DataFrame."""

    def to_pyarrow_bytes(self, schema: Schema) -> bytes:
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        data = self.to_pyarrow(schema=schema)
        out = pa.BufferOutputStream()
        with pa.RecordBatchFileWriter(out, data.schema) as writer:
            writer.write(data)
        return out.getvalue()
