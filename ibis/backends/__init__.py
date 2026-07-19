from __future__ import annotations

import abc
import atexit
import collections.abc
import contextlib
import functools
import importlib.metadata
import keyword
import re
import sys
import urllib.parse
import weakref
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, overload

import ibis
import ibis.common.exceptions as exc
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    import sqlglot as sg
    import torch


__all__ = ("BaseBackend", "connect")


class TablesAccessor(collections.abc.Mapping):
    """A mapping-like object for accessing tables off a backend.

    Tables may be accessed by name using either index or attribute access:

    Examples
    --------
    >>> con = ibis.sqlite.connect("example.db")
    >>> people = con.tables["people"]  # access via index
    >>> people = con.tables.people  # access via attribute
    """

    def __init__(self, backend: BaseBackend) -> None:
        self._backend = backend

    def __getitem__(self, name: str, /) -> ir.Table:
        try:
            return self._backend.table(name)
        except Exception as exc:
            raise KeyError(name) from exc

    def __getattr__(self, name: str, /) -> ir.Table:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._backend.table(name)
        except Exception as exc:
            raise AttributeError(name) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._backend.list_tables()))

    def __len__(self) -> int:
        return len(self._backend.list_tables())

    def __dir__(self) -> list[str]:
        o = set()
        o.update(dir(type(self)))
        o.update(
            name
            for name in self._backend.list_tables()
            if name.isidentifier() and not keyword.iskeyword(name)
        )
        return list(o)

    def __repr__(self) -> str:
        tables = self._backend.list_tables()
        rows = ["Tables", "------"]
        rows.extend(f"- {name}" for name in sorted(tables))
        return "\n".join(rows)

    def _ipython_key_completions_(self) -> list[str]:
        return self._backend.list_tables()

# _FileIOHandler 是一个基类/混合类（Mixin），专门用于为各种后端（Backend）提供文件系统交互与数据格式转换的标准接口
# 核心定位是 “数据进出 Ibis 的标准化枢纽”
# 格式统一化：它将 Ibis 表达式执行的结果转换为常用的 Python 数据生态格式（如 pandas DataFrame、Polars DataFrame、PyArrow 表、PyTorch Tensor）
# 文件读写能力：它定义了读取（read_...）和写入（to_...）常见数据格式（Parquet, CSV, JSON, Delta Lake）的通用协议。
# 后端解耦：通过依赖 pyarrow 作为中间桥梁，使得不同数据库后端无需重复实现这些复杂的转换逻辑，只需实现基础的执行逻辑即可。
class _FileIOHandler:
    # 静态私有方法。负责动态导入 pyarrow 库
    @staticmethod
    def _import_pyarrow():
        try:
            import pyarrow  # noqa: ICN001
        except ImportError:
            raise ModuleNotFoundError(
                "Exporting to arrow formats requires `pyarrow` but it is not installed"
            )
        else:
            import pyarrow_hotfix  # noqa: F401

            return pyarrow
    # 将 Ibis 表达式执行结果转为 pandas 的 DataFrame、Series 或标量。
    # 直接包装了后端的 execute 方法
    def to_pandas(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | Any:
        """Execute an Ibis expression and return a pandas `DataFrame`, `Series`, or scalar.

        ::: {.callout-note}
        This method is a wrapper around `execute`.
        :::

        Parameters
        ----------
        expr
            Ibis expression to execute.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        kwargs
            Keyword arguments
        """
        return self.execute(expr, params=params, limit=limit, **kwargs)
    # 以迭代器方式分批返回 pandas DataFrame
    # 调用 to_pyarrow_batches 获取分批后的 Arrow 数据，再利用 PandasData.convert_table 将每批次转换为 pandas 格式。适用于超大数据集的内存优化处理。
    def to_pandas_batches(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> Iterator[pd.DataFrame | pd.Series | Any]:
        """Execute an Ibis expression and return an iterator of pandas `DataFrame`s.

        Parameters
        ----------
        expr
            Ibis expression to execute.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        chunk_size
            Maximum number of rows in each returned `DataFrame` batch. This may have
            no effect depending on the backend.
        kwargs
            Keyword arguments

        Returns
        -------
        Iterator[pd.DataFrame]
            An iterator of pandas `DataFrame`s.

        """
        from ibis.formats.pandas import PandasData

        orig_expr = expr
        expr = expr.as_table()
        schema = expr.schema()
        yield from (
            orig_expr.__pandas_result__(
                PandasData.convert_table(batch.to_pandas(), schema)
            )
            for batch in self.to_pyarrow_batches(
                expr, params=params, limit=limit, chunk_size=chunk_size, **kwargs
            )
        )

    @overload
    def to_pyarrow(
        self,
        expr: ir.Table,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table: ...

    @overload
    def to_pyarrow(
        self,
        expr: ir.Column,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Array: ...

    @overload
    def to_pyarrow(
        self,
        expr: ir.Scalar,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Scalar: ...

    @util.experimental
    def to_pyarrow(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table | pa.Array | pa.Scalar:
        """Execute expression to a pyarrow object.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            Ibis expression to export to pyarrow
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        kwargs
            Keyword arguments

        Returns
        -------
        result
            If the passed expression is a Table, a pyarrow table is returned.
            If the passed expression is a Column, a pyarrow array is returned.
            If the passed expression is a Scalar, a pyarrow scalar is returned.
        """
        pa = self._import_pyarrow()
        self._run_pre_execute_hooks(expr)

        table_expr = expr.as_table()
        schema = table_expr.schema()
        arrow_schema = schema.to_pyarrow()
        with self.to_pyarrow_batches(
            table_expr, params=params, limit=limit, **kwargs
        ) as reader:
            table = pa.Table.from_batches(reader, schema=arrow_schema)

        return expr.__pyarrow_result__(
            table.rename_columns(list(table_expr.columns)).cast(arrow_schema)
        )

    @util.experimental
    def to_polars(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Execute expression and return results in as a polars DataFrame.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            Ibis expression to export to polars.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        kwargs
            Keyword arguments

        Returns
        -------
        dataframe
            A polars DataFrame holding the results of the executed expression.

        """
        import polars as pl

        table = self.to_pyarrow(expr.as_table(), params=params, limit=limit, **kwargs)
        return expr.__polars_result__(pl.from_arrow(table))

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        """Execute expression and return a RecordBatchReader.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            Ibis expression to export to pyarrow
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        chunk_size
            Maximum number of rows in each returned record batch.
        kwargs
            Keyword arguments

        Returns
        -------
        results
            RecordBatchReader

        """
        raise NotImplementedError

    @util.experimental
    def to_torch(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Execute an expression and return results as a dictionary of torch tensors.

        Parameters
        ----------
        expr
            Ibis expression to execute.
        params
            Parameters to substitute into the expression.
        limit
            An integer to effect a specific row limit. A value of `None` means no limit.
        kwargs
            Keyword arguments passed into the backend's `to_torch` implementation.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of torch tensors, keyed by column name.

        """
        import torch

        t = self.to_pyarrow(expr, params=params, limit=limit, **kwargs)
        # without .copy() the arrays are read-only and thus writing to them is
        # undefined behavior; we can't ignore this warning from torch because
        # we're going out of ibis and downstream code can do whatever it wants
        # with the data
        return {
            name: torch.from_numpy(t[name].to_numpy().copy()) for name in t.schema.names
        }

    def read_parquet(
        self, path: str | Path, /, *, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a parquet file as a table in the current backend.

        Parameters
        ----------
        path
            The data source.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to the backend loading function.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        raise NotImplementedError(
            f"{self.name} does not support direct registration of parquet data."
        )

    def read_csv(
        self, path: str | Path, /, *, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a CSV file as a table in the current backend.

        Parameters
        ----------
        path
            The data source. A string or Path to the CSV file.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to the backend loading function.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        raise NotImplementedError(
            f"{self.name} does not support direct registration of CSV data."
        )

    def read_json(
        self, path: str | Path, /, *, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a JSON file as a table in the current backend.

        Parameters
        ----------
        path
            The data source. A string or Path to the JSON file.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to the backend loading function.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        raise NotImplementedError(
            f"{self.name} does not support direct registration of JSON data."
        )

    def read_delta(
        self, path: str | Path, /, *, table_name: str | None = None, **kwargs: Any
    ):
        """Register a Delta Lake table in the current database.

        Parameters
        ----------
        path
            The data source. Must be a directory containing a Delta Lake table.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to the underlying backend or library.

        Returns
        -------
        ir.Table
            The just-registered table.
        """
        raise NotImplementedError(
            f"{self.name} does not support direct registration of DeltaLake tables."
        )

    @util.experimental
    def to_parquet(
        self,
        expr: ir.Table,
        /,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a parquet file.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to parquet.
        path
            The data source. A string or Path to the parquet file.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.parquet.ParquetWriter

        https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html

        """
        self._import_pyarrow()
        import pyarrow.parquet as pq

        with expr.to_pyarrow_batches(params=params) as batch_reader:
            with pq.ParquetWriter(path, batch_reader.schema, **kwargs) as writer:
                for batch in batch_reader:
                    writer.write_batch(batch)

    @util.experimental
    def to_parquet_dir(
        self,
        expr: ir.Table,
        /,
        directory: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a parquet file in a directory.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to parquet.
        directory
            The data source. A string or Path to the directory where the parquet file will be written.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.dataset.write_dataset

        https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html

        """
        self._import_pyarrow()
        import pyarrow.dataset as ds

        # by default write_dataset creates the directory
        with expr.to_pyarrow_batches(params=params) as batch_reader:
            ds.write_dataset(
                batch_reader, base_dir=directory, format="parquet", **kwargs
            )

    @util.experimental
    def to_csv(
        self,
        expr: ir.Table,
        /,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a CSV file.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to CSV.
        path
            The data source. A string or Path to the CSV file.
        params
            Mapping of scalar parameter expressions to value.
        kwargs
            Additional keyword arguments passed to pyarrow.csv.CSVWriter

        https://arrow.apache.org/docs/python/generated/pyarrow.csv.CSVWriter.html
        """
        self._import_pyarrow()
        import pyarrow.csv as pcsv

        with expr.to_pyarrow_batches(params=params) as batch_reader:
            with pcsv.CSVWriter(path, batch_reader.schema, **kwargs) as writer:
                for batch in batch_reader:
                    writer.write_batch(batch)

    @util.experimental
    def to_delta(
        self,
        expr: ir.Table,
        /,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a Delta Lake table.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to Delta Lake table.
        path
            The data source. A string or Path to the Delta Lake table.
        params
            Mapping of scalar parameter expressions to value.
        kwargs
            Additional keyword arguments passed to deltalake.writer.write_deltalake method

        """
        try:
            from deltalake.writer import write_deltalake
        except ImportError:
            raise ImportError(
                "The deltalake extra is required to use the "
                "to_delta method. You can install it using pip:\n\n"
                "pip install 'ibis-framework[deltalake]'\n"
            )

        with expr.to_pyarrow_batches(params=params) as batch_reader:
            write_deltalake(path, batch_reader, **kwargs)

    @util.experimental
    def to_json(
        self,
        expr: ir.Table,
        /,
        path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Write the results of `expr` to a json file of [{column -> value}, ...] objects.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to Delta Lake table.
        path
            The data source. A string or Path to the Delta Lake table.
        kwargs
            Additional, backend-specifc keyword arguments.
        """
        backend = expr._find_backend(use_default=True)
        raise NotImplementedError(
            f"{backend.__class__.__name__} does not support writing to JSON."
        )


class HasCurrentCatalog(abc.ABC):
    """Has a `current_catalog` property."""

    @property
    @abc.abstractmethod
    def current_catalog(self) -> str:
        """The name of the current catalog in the backend.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of the terminology the backend uses.

        See the
        [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
        for more info.

        Returns
        -------
        str
            The name of the current catalog.
        """


class HasCurrentDatabase(abc.ABC):
    """Has a `current_database` property."""

    @property
    @abc.abstractmethod
    def current_database(self) -> str:
        """The name of the current database in the backend.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of the terminology the backend uses.

        See the
        [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
        for more info.

        Returns
        -------
        str
            The name of the current database.
        """


class CanListCatalog(abc.ABC):
    @abc.abstractmethod
    def list_catalogs(self, *, like: str | None = None) -> list[str]:
        """List existing catalogs in the current connection.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of the terminology the backend uses.

        See the
        [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
        for more info.
        :::

        Parameters
        ----------
        like
            A pattern in Python's regex format to filter returned catalog names.

        Returns
        -------
        list[str]
            The catalog names that exist in the current connection, that match
            the `like` pattern if provided.
        """


class CanCreateCatalog(CanListCatalog):
    @abc.abstractmethod
    def create_catalog(self, name: str, /, *, force: bool = False) -> None:
        """Create a new catalog.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of the terminology the backend uses.

        See the
        [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
        for more info.
        :::

        Parameters
        ----------
        name
            Name of the new catalog.
        force
            If `False`, an exception is raised if the catalog already exists.
        """

    @abc.abstractmethod
    def drop_catalog(self, name: str, /, *, force: bool = False) -> None:
        """Drop a catalog with name `name`.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of the terminology the backend uses.

        See the
        [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
        for more info.
        :::

        Parameters
        ----------
        name
            Catalog to drop.
        force
            If `False`, an exception is raised if the catalog does not exist.
        """


class CanListDatabase(abc.ABC):
    @abc.abstractmethod
    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        """List existing databases in the current connection.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of the terminology the backend uses.

        See the
        [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
        for more info.
        :::

        Parameters
        ----------
        like
            A pattern in Python's regex format to filter returned database
            names.
        catalog
            The catalog to list databases from. If `None`, the current catalog
            is searched.

        Returns
        -------
        list[str]
            The database names that exist in the current connection, that match
            the `like` pattern if provided.
        """


class CanCreateDatabase(CanListDatabase):
    @abc.abstractmethod
    def create_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        """Create a database named `name` in `catalog`.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of the terminology the backend uses.

        See the
        [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
        for more info.
        :::

        Parameters
        ----------
        name
            Name of the database to create.
        catalog
            Name of the catalog in which to create the database. If `None`, the
            current catalog is used.
        force
            If `False`, an exception is raised if the database exists.

        """

    @abc.abstractmethod
    def drop_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        """Drop the database with `name` in `catalog`.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of the terminology the backend uses.

        See the
        [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
        for more info.
        :::

        Parameters
        ----------
        name
            Name of the schema to drop.
        catalog
            Name of the catalog to drop the database from.
            If `None`, the current catalog is used.
        force
            If `False`, an exception is raised if the database does not exist.

        """


class CacheEntry(NamedTuple):
    orig_op: ops.Relation
    cached_op_ref: weakref.ref[ops.Relation]
    finalizer: weakref.finalize

# 用于为后端（Backend）提供缓存管理能力。
# 核心定位是 “Ibis 计算结果的生命周期管理器”
class CacheHandler:
    """A mixin for handling `.cache()`/`CachedTable` operations."""

    def __init__(self):
        self._cache_name_to_entry = {}
        self._cache_op_to_entry = {}

    def _cached_table(self, table: ir.Table) -> ir.CachedTable:
        """Convert a Table to a CachedTable.

        Parameters
        ----------
        table
            Table expression to cache

        Returns
        -------
        Table
            Cached table
        """
        entry = self._cache_op_to_entry.get(table.op())
        if entry is None or (cached_op := entry.cached_op_ref()) is None:
            cached_op = self._create_cached_table(util.gen_name("cached"), table).op()
            entry = CacheEntry(
                table.op(),
                weakref.ref(cached_op),
                weakref.finalize(
                    cached_op, self._finalize_cached_table, cached_op.name
                ),
            )
            self._cache_op_to_entry[table.op()] = entry
            self._cache_name_to_entry[cached_op.name] = entry
        return ir.CachedTable(cached_op)

    def _finalize_cached_table(self, name: str) -> None:
        """Release a cached table given its name.

        This is a no-op if the cached table is already released.

        Parameters
        ----------
        name
            The name of the cached table.
        """
        if (entry := self._cache_name_to_entry.pop(name, None)) is not None:
            self._cache_op_to_entry.pop(entry.orig_op)
            entry.finalizer.detach()
            try:
                self._drop_cached_table(name)
            except Exception:
                # suppress exceptions during interpreter shutdown
                if not sys.is_finalizing():
                    raise

    def _create_cached_table(self, name: str, expr: ir.Table) -> ir.Table:
        return self.create_table(name, expr, schema=expr.schema(), temp=True)

    def _drop_cached_table(self, name: str) -> None:
        self.drop_table(name, force=True)

# BaseBackend 是所有数据库后端（如 DuckDB, PostgreSQL, BigQuery, pandas 等）的核心抽象基类。
# 它定义了 Ibis 统一查询接口的“契约”，确保用户无论底层连接的是什么数据库，都能使用相同的 Python API 进行数据操作
# Ibis “Write Once, Run Anywhere”（一次编写，随处运行）架构的基石
# 统一 API 规范：强制所有后端实现 table()、compile()、execute() 等标准方法
# 混合能力集成：继承了 _FileIOHandler（处理数据导入导出）和 CacheHandler（处理结果缓存），具备了完整的数据生命周期管理能力。
# 基础设施抽象：处理连接池生命周期、内存表（In-Memory Table）注册、UDF 注册以及 SQL 方言转换。
class BaseBackend(abc.ABC, _FileIOHandler, CacheHandler):
    """Base backend class.

    All Ibis backends must subclass this class and implement all the
    required methods.
    """
    # 子类定义的后端唯一名称（如 "duckdb"）
    name: ClassVar[str]
    # 标识该后端是否支持创建临时表
    supports_temporary_tables = False
    # 标识该后端是否支持将 Python 函数作为 UDF 在数据库中执行。
    supports_python_udfs = False
    # 初始化连接参数、内存表集合，并调用父类（Mixin）初始化
    def __init__(self, *args, **kwargs):
        # 存储连接数据库所需的参数，用于序列化（__getstate__）和重新连接（reconnect）
        self._con_args: tuple[Any] = args
        self._con_kwargs: dict[str, Any] = kwargs
        self._can_reconnect: bool = True
        # 用于追踪当前会话中注册的内存表，防止引用泄露。
        self._memtables = weakref.WeakSet()
        super().__init__()

    @property
    @abc.abstractmethod
    def dialect(self) -> sg.Dialect | None:
        """The sqlglot dialect for this backend, where applicable.

        Returns None if the backend is not a SQL backend.
        """

    def __getstate__(self):
        return dict(_con_args=self._con_args, _con_kwargs=self._con_kwargs)

    def __rich_repr__(self):
        yield "name", self.name

    def __hash__(self):
        return hash(self.db_identity)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.db_identity == other.db_identity
    # 计算属性，根据连接参数生成数据库的唯一标识（哈希值），用于去重判断。
    @functools.cached_property
    def db_identity(self) -> str:
        """Return the identity of the database.

        Multiple connections to the same
        database will return the same value for `db_identity`.

        The default implementation assumes connection parameters uniquely
        specify the database.

        Returns
        -------
        Hashable
            Database identity

        """
        parts = [self.__class__]
        parts.extend(self._con_args)
        parts.extend(f"{k}={v}" for k, v in self._con_kwargs.items())
        return "_".join(map(str, parts))
    # 类工厂方法
    # 创建并返回一个新的后端实例，同时自动触发连接
    # TODO(kszucs): this should be a classmethod returning with a new backend
    # instance which does instantiate the connection
    def connect(self, *args, **kwargs) -> BaseBackend:
        """Connect to the database.

        Parameters
        ----------
        *args
            Mandatory connection parameters, see the docstring of `do_connect`
            for details.
        **kwargs
            Extra connection parameters, see the docstring of `do_connect` for
            details.

        Notes
        -----
        This creates a new backend instance with saved `args` and `kwargs`,
        then calls `reconnect` and finally returns the newly created and
        connected backend instance.

        Returns
        -------
        BaseBackend
            An instance of the backend

        """
        new_backend = self.__class__(*args, **kwargs)
        new_backend.reconnect()
        return new_backend
    # 负责断开数据库连接
    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the backend."""
    # 用于在连接前对参数进行标准化或预处理
    @staticmethod
    def _convert_kwargs(kwargs: MutableMapping) -> None:
        """Manipulate keyword arguments to `.connect` method."""
    # 使用缓存的 _con_args 和 _con_kwargs 重新执行 do_connect
    # TODO(kszucs): should call self.connect(*self._con_args, **self._con_kwargs)
    def reconnect(self) -> None:
        """Reconnect to the database already configured with connect."""
        if self._can_reconnect:
            self.do_connect(*self._con_args, **self._con_kwargs)
        else:
            raise exc.IbisError("Cannot reconnect to unconfigured {self.name} backend")
    # 后端子类必须实现此逻辑以建立物理连接
    def do_connect(self, *args, **kwargs) -> None:
        """Connect to database specified by `args` and `kwargs`."""

    @staticmethod
    def _filter_with_like(values: Iterable[str], like: str | None = None) -> list[str]:
        """Filter names with a `like` pattern (regex).

        The methods `list_databases` and `list_tables` accept a `like`
        argument, which filters the returned tables with tables that match the
        provided pattern.

        We provide this method in the base backend, so backends can use it
        instead of reinventing the wheel.

        Parameters
        ----------
        values
            Iterable of strings to filter
        like
            Pattern to use for filtering names

        Returns
        -------
        list[str]
            Names filtered by the `like` pattern.

        """
        if like is None:
            return sorted(values)

        pattern = re.compile(like)
        return sorted(filter(pattern.findall, values))
    # 获取数据库中的表列表，支持 like 正则过滤
    @abc.abstractmethod
    def list_tables(
        self, *, like: str | None = None, database: tuple[str, str] | str | None = None
    ) -> list[str]:
        """The table names that match `like` in the given `database`.

        For some backends, the tables may be files in a directory,
        or other equivalent entities in a SQL database.

        Parameters
        ----------
        like
            A pattern in Python's regex format.
        database
            The database, or (catalog, database) from which to list tables.

            For backends that support a single-level table hierarchy,
            you can pass in a string like `"bar"`.
            For backends that support multi-level table hierarchies, you can
            pass in a dotted string path like `"catalog.database"` or a tuple of
            strings like `("catalog", "database")`.
            If not provided, the current database
            (and catalog, if applicable for this backend) is used.

            See the
            [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
            for more info.

        Returns
        -------
        list[str]
            The list of the table names that match the pattern `like`.

        Examples
        --------
        This example uses the DuckDB backend, but the list_tables API
        works the same for other backends.

        >>> import ibis
        >>> con = ibis.duckdb.connect()
        >>> foo = con.create_table("foo", schema=ibis.schema(dict(a="int")))
        >>> con.list_tables()
        ['foo']
        >>> bar = con.create_view("bar", foo)
        >>> con.list_tables()
        ['bar', 'foo']
        >>> con.create_database("my_database")
        >>> con.list_tables(database="my_database")
        []
        >>> con.raw_sql("CREATE TABLE my_database.baz (a INTEGER)")  # doctest: +ELLIPSIS
        <duckdb.duckdb.DuckDBPyConnection object at 0x...>
        >>> con.list_tables(database="my_database")
        ['baz']
        """
    # 返回一个 TablesAccessor，允许通过属性语法（con.tables.my_table）快速访问表。
    @abc.abstractmethod
    def table(
        self, name: str, /, *, database: tuple[str, str] | str | None = None
    ) -> ir.Table:
        """Construct a table expression from the corresponding table in the backend.

        Parameters
        ----------
        name
            Table name
        database
            The database, or (catalog, database) from which to get the table.

            For backends that support a single-level table hierarchy,
            you can pass in a string like `"bar"`.
            For backends that support multi-level table hierarchies, you can
            pass in a dotted string path like `"catalog.database"` or a tuple of
            strings like `("catalog", "database")`.
            If not provided, the current database
            (and catalog, if applicable for this backend) is used.

            See the
            [Table Hierarchy Concepts Guide](/concepts/backend-table-hierarchy.qmd)
            for more info.

        Returns
        -------
        Table
            Table expression

        Examples
        --------
        >>> import ibis
        >>> backend = ibis.duckdb.connect()

        Get the "foo" table from the current database
        (and catalog, if applicable for this backend):

        >>> backend.table("foo")  # doctest: +SKIP

        Get the "foo" table from the "bar" database
        (in DuckDB's language they would say the "bar" schema,
        in SQL this would be `"bar"."foo"`)

        >>> backend.table("foo", database="bar")  # doctest: +SKIP

        Get the "foo" table from the "bar" database, within the "baz" catalog
        (in DuckDB's language they would say the "bar" schema, and "baz" database,
        in SQL this would be `"baz"."bar"."foo"`)

        >>> backend.table("foo", database=("baz", "bar"))  # doctest: +SKIP
        """

    @property
    def tables(self):
        """An accessor for tables in the database.

        Tables may be accessed by name using either index or attribute access:

        Examples
        --------
        >>> con = ibis.sqlite.connect("example.db")
        >>> people = con.tables["people"]  # access via index
        >>> people = con.tables.people  # access via attribute

        """
        return TablesAccessor(self)
    # 返回后端引擎的版本字符串
    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Return the version of the backend engine.

        For database servers, return the server version.

        For others such as SQLite and pandas return the version of the
        underlying library or application.

        Returns
        -------
        str
            The backend version

        """
    # 将该后端的自定义配置项注册到 ibis.config 中
    @classmethod
    def register_options(cls) -> None:
        """Register custom backend options."""
        options = ibis.config.options
        backend_name = cls.name
        try:
            backend_options = cls.Options()
        except AttributeError:
            pass
        else:
            try:
                setattr(options, backend_name, backend_options)
            except ValueError as e:
                raise exc.BackendConfigurationNotRegistered(backend_name) from e
    # 检查表达式中是否包含 Python UDF，并尝试在后端注册。
    def _register_udfs(self, expr: ir.Expr) -> None:
        """Register UDFs contained in `expr` with the backend."""
        if self.supports_python_udfs:
            raise NotImplementedError(self.name)

    def _verify_in_memory_tables_are_unique(self, expr: ir.Expr) -> None:
        memtables = expr.op().find(ops.InMemoryTable)
        name_counts = Counter(op.name for op in memtables)

        if duplicate_names := sorted(
            name for name, count in name_counts.items() if count > 1
        ):
            raise exc.IbisError(f"Duplicate in-memory table names: {duplicate_names}")
        return memtables
    # 扫描表达式中的临时内存表，并确保它们已上传至后端数据库。
    def _register_in_memory_tables(self, expr: ir.Expr) -> None:
        for memtable in self._verify_in_memory_tables_are_unique(expr):
            if memtable not in self._memtables:
                self._register_in_memory_table(memtable)
                self._memtables.add(memtable)
                if (
                    finalizer := self._make_memtable_finalizer(memtable.name)
                ) is not None:

                    def finalize(finalizer=finalizer):
                        with contextlib.suppress(Exception):
                            finalizer()

                    atexit.register(finalize)
    # 子类需实现将内存数据（如 DataFrame）推送至数据库的逻辑。
    @abc.abstractmethod
    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        """Register an in-memory table associated with `op`."""
    # 生成清理临时表的闭包函数（与 atexit 配合使用）。
    @abc.abstractmethod
    def _make_memtable_finalizer(self, name: str) -> None | Callable[..., None]:
        """Make a finalizer for an in-memory table."""
    # 在执行前自动处理内存表注册和 UDF 注入。
    def _run_pre_execute_hooks(self, expr: ir.Expr) -> None:
        """Backend-specific hooks to run before an expression is executed."""
        self._register_udfs(expr)
        self._register_in_memory_tables(expr)
    # 将 Ibis 表达式转换为目标 SQL 字符串或 Polars LazyFrame
    @abc.abstractmethod
    def compile(
        self,
        expr: ir.Expr,
        /,
        *,
        limit: int | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
        **kwargs: Any,
    ) -> str | pl.LazyFrame:
        """Compile `expr` to a SQL string (for SQL backends) or a LazyFrame (for the polars backend).

        Parameters
        ----------
        expr
            An ibis expression to compile.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        kwargs
            Additional keyword arguments
        """
    # 执行 Ibis 表达式并将结果拉取到本地 Python 环境中（返回 pandas 对象或标量）。
    def execute(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | Any:
        """Execute an Ibis expression and return a pandas `DataFrame`, `Series`, or scalar.

        Parameters
        ----------
        expr
            Ibis expression to execute.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        kwargs
            Keyword arguments
        """
    # 根据表达式或 schema 创建新表
    @abc.abstractmethod
    def create_table(
        self,
        name: str,
        /,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new table.

        Parameters
        ----------
        name
            Name of the new table.
        obj
            An Ibis table expression or pandas table that will be used to
            extract the schema and the data of the new table. If not provided,
            `schema` must be given.
        schema
            The schema for the new table. Only one of `schema` or `obj` can be
            provided.
        database
            Name of the database where the table will be created, if not the
            default.
        temp
            Whether a table is temporary or not
        overwrite
            Whether to clobber existing data

        Returns
        -------
        Table
            The table that was created.
        """
    # 删除表
    @abc.abstractmethod
    def drop_table(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table.

        Parameters
        ----------
        name
            Name of the table to drop.
        database
            Name of the database where the table exists, if not the default.
        force
            If `False`, an exception is raised if the table does not exist.
        """
        raise NotImplementedError(
            f'Backend "{self.name}" does not implement "drop_table"'
        )
    # 重命名表
    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename an existing table.

        Parameters
        ----------
        old_name
            The old name of the table.
        new_name
            The new name of the table.

        """
        raise NotImplementedError(
            f'Backend "{self.name}" does not implement "rename_table"'
        )
    # 视图的创建与销毁
    @abc.abstractmethod
    def create_view(
        self,
        name: str,
        /,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new view from an expression.

        Parameters
        ----------
        name
            Name of the new view.
        obj
            An Ibis table expression that will be used to create the view.
        database
            Name of the database where the view will be created, if not
            provided the database's default is used.
        overwrite
            Whether to clobber an existing view with the same name

        Returns
        -------
        ir.Table
            The view that was created.
        """
    # 视图的创建与销毁
    @abc.abstractmethod
    def drop_view(
        self, name: str, /, *, database: str | None = None, force: bool = False
    ) -> None:
        """Drop a view.

        Parameters
        ----------
        name
            Name of the view to drop.
        database
            Name of the database where the view exists, if not the default.
        force
            If `False`, an exception is raised if the view does not exist.
        """
    # 检查该后端是否原生支持某种特定的 Ibis 操作（如数组索引、正则匹配）。
    @classmethod
    def has_operation(cls, operation: type[ops.Value], /) -> bool:
        """Return whether the backend implements support for `operation`.

        Parameters
        ----------
        operation
            A class corresponding to an operation.

        Returns
        -------
        bool
            Whether the backend implements the operation.

        Examples
        --------
        >>> import ibis
        >>> import ibis.expr.operations as ops
        >>> ibis.sqlite.has_operation(ops.ArrayIndex)
        False
        >>> ibis.postgres.has_operation(ops.ArrayIndex)
        True
        """
        raise NotImplementedError(
            f"{cls.name} backend has not implemented `has_operation` API"
        )
    # 利用 sqlglot 将一种 SQL 方言转换为当前后端的方言。
    def _transpile_sql(self, query: str, *, dialect: str | None = None) -> str:
        # only transpile if dialect was passed
        if dialect is None:
            return query

        import sqlglot as sg

        # only transpile if the backend dialect doesn't match the input dialect
        name = self.name
        if (output_dialect := self.dialect) is None:
            raise NotImplementedError(f"No known sqlglot dialect for backend {name}")

        if dialect != output_dialect:
            (query,) = sg.transpile(query, read=dialect, write=output_dialect)
        return query

# 设计目的是为 Ibis 的集成测试或示例演示提供一种标准化的、高效的数据摄入机制，支持从本地文件直接读取数据到数据库后端。
class BaseExampleLoader(abc.ABC):
    __slots__ = ()
    # 模板方法（Template Method）模式的应用，通过判断文件后缀自动分发加载任务
    @abc.abstractmethod
    def _load_example(self, *, path: str | Path, table_name: str) -> ir.Table:
        # Read directly into these backends. This helps reduce memory
        # usage, making the larger example datasets easier to work with.
        if path.endswith(".parquet"):
            return self._load_parquet(path=path, table_name=table_name)
        else:
            return self._load_csv(path=path, table_name=table_name)


class ExampleLoader(BaseExampleLoader):
    __slots__ = ()

    def _load_example(self, *, path: str | Path, table_name: str) -> ir.Table:
        # Read directly into these backends. This helps reduce memory
        # usage, making the larger example datasets easier to work with.
        if path.endswith(".parquet"):
            return self._load_parquet(path=path, table_name=table_name)
        else:
            return self._load_csv(path=path, table_name=table_name)

    @abc.abstractmethod
    def _load_parquet(self, *, path: str | Path, table_name: str) -> ir.Table:
        """Load an example Apache Parquet file."""

    @abc.abstractmethod
    def _load_csv(self, *, path: str | Path, table_name: str) -> ir.Table:
        """Load an example CSV file."""


class NoExampleLoader(BaseExampleLoader):
    __slots__ = ()

    def _load_example(self, *, path: str | Path, table_name: str) -> ir.Table:
        raise NotImplementedError(f"{self.name} does not support loading example data.")


class DirectExampleLoader(ExampleLoader):
    __slots__ = ()

    def _load_parquet(self, *, path: str | Path, table_name: str) -> ir.Table:
        return self.read_parquet(path, table_name=table_name)

    def _load_csv(self, *, path: str | Path, table_name: str) -> ir.Table:
        return self.read_csv(path, table_name=table_name)


class DirectPyArrowExampleLoader(DirectExampleLoader):
    __slots__ = ()

    overwrite_example: bool = False
    temporary_example: bool = False

    def _load_csv(self, *, path: str | Path, table_name: str) -> ir.Table:
        import pyarrow as pa
        import pyarrow.csv

        # The convert options lets pyarrow treat empty strings as null for
        # string columns, but not quoted empty strings.
        table = pyarrow.csv.read_csv(
            path,
            convert_options=pyarrow.csv.ConvertOptions(
                strings_can_be_null=True, quoted_strings_can_be_null=False
            ),
        )

        # All null columns are inferred as null-type, but not all
        # backends support null-type columns. Cast to an all-null
        # string column instead.
        for i, field in enumerate(table.schema):
            if pyarrow.types.is_null(field.type):
                table = table.set_column(i, field.name, table[i].cast(pa.string()))

        return self.create_table(
            table_name,
            obj=table,
            temp=self.temporary_example,
            overwrite=self.overwrite_example,
        )


class PyArrowExampleLoader(ExampleLoader):
    __slots__ = ()

    overwrite_example: bool = False
    temporary_example: bool = True

    def _load_parquet(self, *, path: str | Path, table_name: str) -> ir.Table:
        import pyarrow_hotfix  # noqa: F401, I001
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return self.create_table(
            table_name,
            obj=table,
            temp=self.temporary_example,
            overwrite=self.overwrite_example,
        )

    def _load_csv(self, *, path: str | Path, table_name: str) -> ir.Table:
        import pyarrow_hotfix  # noqa: F401, I001
        import pyarrow as pa
        import pyarrow.csv

        # The convert options lets pyarrow treat empty strings as null for
        # string columns, but not quoted empty strings.
        table = pyarrow.csv.read_csv(
            path,
            convert_options=pyarrow.csv.ConvertOptions(
                strings_can_be_null=True, quoted_strings_can_be_null=False
            ),
        )

        # All null columns are inferred as null-type, but not all
        # backends support null-type columns. Cast to an all-null
        # string column instead.
        for i, field in enumerate(table.schema):
            if pyarrow.types.is_null(field.type):
                table = table.set_column(i, field.name, table[i].cast(pa.string()))

        return self.create_table(
            table_name,
            obj=table,
            temp=self.temporary_example,
            overwrite=self.overwrite_example,
        )


@functools.cache
def _get_backend_names(*, exclude: tuple[str] = ()) -> frozenset[str]:
    """Return the set of known backend names.

    Parameters
    ----------
    exclude
        Exclude these backend names from the result

    Notes
    -----
    This function returns a frozenset to prevent cache pollution.

    If a `set` is used, then any in-place modifications to the set
    are visible to every caller of this function.

    """

    entrypoints = importlib.metadata.entry_points(group="ibis.backends")
    return frozenset(ep.name for ep in entrypoints).difference(exclude)


def connect(resource: Path | str, /, **kwargs: Any) -> BaseBackend:
    """Connect to `resource`, inferring the backend automatically.

    The general pattern for `ibis.connect` is

    ```python
    con = ibis.connect("backend://connection-parameters")
    ```

    With many backends that looks like

    ```python
    con = ibis.connect("backend://user:password@host:port/database")
    ```

    See the connection syntax for each backend for details about URL connection
    requirements.

    Parameters
    ----------
    resource
        A URL or path to the resource to be connected to.
    kwargs
        Backend specific keyword arguments

    Examples
    --------
    Connect to an in-memory DuckDB database:

    >>> import ibis
    >>> con = ibis.connect("duckdb://")

    Connect to an on-disk SQLite database:

    >>> con = ibis.connect("sqlite://relative.db")
    >>> con = ibis.connect(
    ...     "sqlite:///absolute/path/to/data.db"
    ... )  # quartodoc: +SKIP # doctest: +SKIP

    Connect to a PostgreSQL server:

    >>> con = ibis.connect(
    ...     "postgres://user:password@hostname:5432"
    ... )  # quartodoc: +SKIP # doctest: +SKIP

    Connect to BigQuery:

    >>> con = ibis.connect(
    ...     "bigquery://my-project/my-dataset"
    ... )  # quartodoc: +SKIP # doctest: +SKIP

    """
    url = resource = str(resource)

    if re.match("[A-Za-z]:", url):
        # windows path with drive, treat it as a file
        url = f"file://{url}"

    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme or "file"

    orig_kwargs = kwargs.copy()
    kwargs = dict(urllib.parse.parse_qsl(parsed.query))

    # convert single parameter lists value to single values
    for name, value in kwargs.items():
        if len(value) == 1:
            kwargs[name] = value[0]

    # Merge explicit kwargs with query string, explicit kwargs
    # taking precedence
    kwargs.update(orig_kwargs)

    if scheme == "file":
        path = parsed.netloc + parsed.path
        if path.endswith(".duckdb"):
            return ibis.duckdb.connect(path, **kwargs)
        elif path.endswith((".sqlite", ".db")):
            return ibis.sqlite.connect(path, **kwargs)
        elif path.endswith((".csv", ".csv.gz")):
            # Load csv/csv.gz files with duckdb by default
            con = ibis.duckdb.connect(**kwargs)
            con.read_csv(path)
            return con
        elif path.endswith(".parquet"):
            # Load parquet files with duckdb by default
            con = ibis.duckdb.connect(**kwargs)
            con.read_parquet(path)
            return con
        else:
            raise ValueError(f"Don't know how to connect to {resource!r}")

    # Treat `postgres://` and `postgresql://` the same
    scheme = scheme.replace("postgresql", "postgres")

    try:
        backend = getattr(ibis, scheme)
    except AttributeError:
        raise ValueError(f"Don't know how to connect to {resource!r}") from None

    return backend._from_url(parsed, **kwargs)


class UrlFromPath:
    __slots__ = ()

    def _from_url(self, url: ParseResult, **kwargs: Any) -> BaseBackend:
        """Connect to a backend using a URL `url`.

        Parameters
        ----------
        url
            URL with which to connect to a backend.
        kwargs
            Additional keyword arguments

        Returns
        -------
        BaseBackend
            A backend instance

        """
        netloc = url.netloc
        parts = list(filter(None, (netloc, url.path[bool(netloc) :])))
        database = Path(*parts) if parts and parts != [":memory:"] else ":memory:"
        if (strdatabase := str(database)).startswith("md:") or strdatabase.startswith(
            "motherduck:"
        ):
            database = strdatabase
        elif isinstance(database, Path):
            database = database.absolute()

        self._convert_kwargs(kwargs)
        return self.connect(database=database, **kwargs)


class NoUrl:
    __slots__ = ()

    name: str

    def _from_url(self, url: ParseResult, **kwargs) -> BaseBackend:
        """Connect to the backend with empty url.

        Parameters
        ----------
        url : str
            The URL with which to connect to the backend. This parameter is not used
            in this method but is kept for consistency.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        BaseBackend
            A backend instance

        """
        return self.connect(**kwargs)


class SupportsTempTables:
    __slots__ = ()

    supports_temporary_tables = True

    def _make_memtable_finalizer(self, name: str) -> None:
        """No-op because temporary tables are automatically cleaned up."""
