from __future__ import annotations

import abc
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import BaseBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import pandas as pd
    import pyarrow as pa

    from ibis.backends.sql.compilers.base import SQLGlotCompiler
    from ibis.expr.api import IntoMemtable
    from ibis.expr.schema import IntoSchema

# SQLBackend 是所有基于 SQL 的数据库后端（如 PostgreSQL, MySQL, SQLite, DuckDB 等）的通用基类。它继承自 BaseBackend，并利用 SQLGlot 库实现了跨数据库的统一编译和执行逻辑。
# SQLBackend 的核心任务是将 Ibis 的高级抽象表达式（IR）翻译并执行为特定方言的 SQL：
# 统一编译引擎：通过 SQLGlotCompiler 将 Ibis 的算子图转换为 SQLGlot 的抽象语法树（AST），再渲染为目标 SQL。
# 标准化 SQL 操作：为所有 SQL 后端提供标准的 CREATE VIEW、DROP TABLE、INSERT、UPSERT 等 DDL 和 DML 操作模板。
# 跨方言兼容：通过 _transpile_sql 和 dialect 属性，能够处理不同数据库在语法上的微小差异。
class SQLBackend(BaseBackend):
    # typing.ClassVar 用于显式地标记一个变量为类变量（Class Variable），而非实例变量。
    # 定义该后端使用的 SQLGlot 编译器实例，负责具体的算子到 SQL 的转换逻辑。
    compiler: ClassVar[SQLGlotCompiler]
    # 后端的名称标识符
    name: ClassVar[str]
    # 指定了该后端支持的工厂方法名称，用于 Ibis 动态加载。
    _top_level_methods = ("from_connection",)
    # 返回当前后端对应的 sqlglot.Dialect，用于 SQL 生成时的语法校验和格式化。
    @property
    def dialect(self) -> sg.Dialect:
        """Return the SQL dialect used by the backend."""
        return self.compiler.dialect
    # 检查编译器是否定义了对应算子的 visit_ 方法，从而判断该后端是否支持特定操作。
    @classmethod
    def has_operation(cls, operation: type[ops.Value], /) -> bool:
        """Return whether the backend supports the given operation.

        Parameters
        ----------
        operation
            Operation type, a Python class object.
        """
        compiler = cls.compiler
        if operation in compiler.extra_supported_ops:
            return True
        method = getattr(compiler, f"visit_{operation.__name__}", None)
        return method not in (
            None,
            compiler.visit_Undefined,
            compiler.visit_Unsupported,
        )
    # 负责数据反序列化的关键环节。
    # 核心任务是将数据库游标（Cursor）中原始的行数据流，转化为结构化、类型对齐的 Pandas DataFrame
    # self: 指向当前的 SQL 后端实例。
    # cursor: 数据库游标对象（符合 Python DB-API 2.0 标准），它是获取查询结果的迭代器。
    # schema: sch.Schema: Ibis 的模式定义对象。它包含了列名列表 (names) 和 Ibis 定义的强数据类型，是后续进行类型转换的标准。
    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        from ibis.formats.pandas import PandasData

        try:
            # 从游标创建初步 DataFrame
            df = pd.DataFrame.from_records(
                cursor, columns=schema.names, coerce_float=True
            )
        except Exception:
            # clean up the cursor if we fail to create the DataFrame
            #
            # in the sqlite case failing to close the cursor results in
            # artificially locked tables
            cursor.close()
            raise
        df = PandasData.convert_table(df, schema)
        return df
    # 用户从数据库中引入一张现有表的入口
    # 通过查询数据库的元数据，构建一个代表物理表的 Ibis 表达式对象 (ir.Table)
    # name: str: 目标表的名称（例如 'users'）。
    def table(
        self, name: str, /, *, database: tuple[str, str] | str | None = None
    ) -> ir.Table:
        # 第一阶段：解析命名空间 (Location Resolution)
        # 将用户传入的 database 参数标准化为 SQLGLOT 的 Table 对象，
        # 从而自动处理不同数据库（如 MySQL 和 Postgres）对“数据库/Schema/Catalog”称呼的差异
        table_loc = self._to_sqlglot_table(database)

        catalog = table_loc.catalog or None
        database = table_loc.db or None
        # 第二阶段：获取元数据 (Schema Discovery)
        # 键的 IO 操作。
        # Ibis 会向数据库发送元数据查询请求（例如在 Postgres 中执行 SELECT ... FROM information_schema.columns），获取该表的列名、数据类型、可空性等定义。
        table_schema = self.get_schema(name, catalog=catalog, database=database)
        return ops.DatabaseTable(
            name,
            schema=table_schema,
            source=self,
            namespace=ops.Namespace(catalog=catalog, database=database),
        ).to_expr()
    # 将 Ibis 表达式转换为目标方言的 SQL 字符串。内部调用 compiler.to_sqlglot 并进行错误处理
    def compile(
        self,
        expr: ir.Expr,
        /,
        *,
        limit: int | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
        pretty: bool = False,
    ) -> str:
        """Compile an expression to a SQL string.

        Parameters
        ----------
        expr
            An ibis expression to compile.
        limit
            An integer to effect a specific row limit. A value of `None` means no limit.
        params
            Mapping of scalar parameter expressions to value.
        pretty
            Pretty print the SQL query during compilation.

        Returns
        -------
        str
            Compiled expression
        """
        query = self.compiler.to_sqlglot(expr, limit=limit, params=params)
        try:
            sql = query.sql(
                dialect=self.dialect,
                pretty=pretty,
                copy=False,
                unsupported_level=sg.ErrorLevel.RAISE,
            )
        except sg.UnsupportedError as e:
            raise exc.UnsupportedOperationError(
                f"Operation not supported in {self.name} backend: {e}\n\nexpression:\n{expr}\n\nsqlglot expression:\n{query}"
            ) from e
        self._log(sql)
        return sql

    def _log(self, sql: str) -> None:
        """Log `sql`.

        This method can be implemented by subclasses. Logging occurs when
        `ibis.options.verbose` is `True`.
        """
        from ibis import util

        util.log(sql)
    # 将原始 SQL 语句“提升”为 Ibis 表达式的入口。
    # 允许用户直接通过 SQL 定义数据源，并利用 Ibis 的后续 API（如 .filter(), .select()）对该 SQL 的结果集进行进一步的链式操作。
    # query: str: 用户编写的原始 SQL 查询字符串（例如 "SELECT * FROM users WHERE age > 18"）。
    # schema: IntoSchema | None: 可选参数。显式提供 SQL 结果集的模式定义。若省略，Ibis 会尝试自动推断。
    # dialect: str | None: 可选参数。指定 SQL 的方言。如果查询是用另一种方言编写的（例如在 Postgres 后端执行 BigQuery 语法的 SQL），Ibis 可以通过此参数进行转换。
    def sql(
        self,
        query: str,
        /,
        *,
        schema: IntoSchema | None = None,
        dialect: str | None = None,
    ) -> ir.Table:
        """Create an Ibis table expression from a SQL query.

        Parameters
        ----------
        query
            A SQL query string
        schema
            The schema of the query. If not provided, Ibis will try to infer
            the schema of the query.
        dialect
            The SQL dialect of the query. If not provided, the backend's dialect
            is assumed. This argument can be useful when the query is written
            in a different dialect from the backend.

        Returns
        -------
        ir.Table
            The table expression representing the query
        """
        # 第一阶段：SQL 方言标准化 (Transpilation)
        # 使用 sqlglot 库将输入的 SQL 转换为当前后端能够理解的标准 SQL。如果用户指定了 dialect，它会执行跨方言转换；如果没有指定，它会检查并确保 SQL 符合当前后端的语法规范，防止执行无效的 SQL。
        query = self._transpile_sql(query, dialect=dialect)
        # 如果用户没有手动传入 schema，Ibis 必须知道 SQL 返回了什么列及其类型，才能进行后续的类型检查。
        if schema is None:
            schema = self._get_schema_using_query(query)
        # 它将 query 字符串和解析好的 schema 封装在一起，告诉 Ibis：“这是一个虚拟表，它的数据由这段 SQL 产生”。
        return ops.SQLQueryResult(query, ibis.schema(schema), self).to_expr()

    @abc.abstractmethod
    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a backend-specific SQL string.

        Parameters
        ----------
        query
            Backend-specific SQL string

        Returns
        -------
        Schema
            The schema inferred from `query`
        """
    # 用于处理 SQL 视图注册与模式推断的辅助方法。
    # 主要作用是：在数据库中定义一个临时视图或公用表表达式（CTE），并获取该视图返回结果的列名与数据类型定义。
    # name: str: 视图或临时表的名称（例如 'tmp_view_123'）。
    # table: ir.Table: 该查询所依赖的基础 Ibis 表表达式（用于构建上下文环境）。
    # query: str: 原始的 SQL 查询字符串，该查询通常引用了 name 定义的视图或需要被转换。
    def _get_sql_string_view_schema(
        self, *, name: str, table: ir.Table, query: str
    ) -> sch.Schema:
        # 用户提供的原始 query 与 Ibis 的表达式树 (table) 逻辑合并
        sql = self.compiler.add_query_to_expr(name=name, table=table, query=query)
        # 通过数据库的“元数据查询”功能（如执行 EXPLAIN、PREPARE 或通过驱动程序的 cursor.description）来获取上述拼接后的 SQL 执行结果的列结构。
        return self._get_schema_using_query(sql)
    # 负责 UDF（用户自定义函数）生命周期管理的核心方法。
    # 作用是在执行 SQL 查询前，将 Ibis 中定义的 Python 函数逻辑“翻译”并注册到目标数据库中，确保数据库能够识别并调用这些自定义逻辑。
    def _register_udfs(self, expr: ir.Expr) -> None:
        udf_sources = []
        compiler = self.compiler
        # 深度扫描整个查询表达式树，找出所有 ScalarUDF 类型的节点。这意味着无论 UDF 嵌套在 select 还是 filter 中，都能被发现。
        for udf_node in expr.op().find(ops.ScalarUDF):
            # 不同的输入类型（如 Python 函数、PyArrow 逻辑）需要不同的数据库创建语句（如 CREATE FUNCTION ... LANGUAGE PYTHON），
            # 此逻辑将具体的编译细节委托给后端对应的 compiler 处理，保持了扩展性。
            compile_func = getattr(
                compiler, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
            )
            # 批量执行注册
            # 通过 ";\n".join(udf_sources) 将所有 UDF 定义合并为一个大的 SQL 脚本执行。
            if sql := compile_func(udf_node):
                udf_sources.append(sql)
        if udf_sources:
            # define every udf in one execution to avoid the overhead of db
            # round trips per udf
            with self._safe_raw_sql(";\n".join(udf_sources)):
                pass
    # 将 Ibis 的逻辑表达持久化为数据库对象的方法。它通过将 Ibis 表达式编译为 CREATE VIEW 语句，在数据库中创建一个虚拟表，并返回对应的 Ibis 表引用
    # name: str: 视图的名称。
    # obj: ir.Table: 要保存为视图的 Ibis 表表达式（查询逻辑）。
    # overwrite: bool: 如果为 True，则生成 CREATE OR REPLACE VIEW 语句，替换同名视图。
    def create_view(
        self,
        name: str,
        /,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a view from an Ibis expression.

        Parameters
        ----------
        name
            The name of the view to create.
        obj
            The Ibis expression to create the view from.
        database
            The database that the view should be created in.
        overwrite
            If `True`, replace an existing view with the same name.

        Returns
        -------
        ir.Table
            A table expression representing the view.
        """
        # 将用户输入的 database 字符串转换为 SQLGlot 的标准 (catalog, db) 元组。这保证了跨数据库方言（如 Postgres 的 schema 与 BigQuery 的 project.dataset）在生成的 SQL 中具有一致的引用格式。
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)
        # 构建 SQL 抽象语法树 (AST)
        src = sge.Create(
            this=sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted),
            kind="VIEW",
            replace=overwrite,
            # 调用编译器的 compile 方法将 Ibis IR 转换成具体的 SQL SELECT 语句，作为视图的定义体。
            expression=self.compile(obj),
        )
        # 如果视图依赖于临时的内存表（如 Pandas 内存数据），此方法会将这些数据预先上传或注册到数据库中，确保视图在创建时能找到数据源。
        self._register_in_memory_tables(obj)
        with self._safe_raw_sql(src):
            pass
        # 重新从数据库的元数据中读取刚创建的视图结构
        return self.table(name, database=(catalog, db))

    def drop_view(
        self, name: str, /, *, database: str | None = None, force: bool = False
    ) -> None:
        """Drop a view from the backend.

        Parameters
        ----------
        name
            The name of the view to drop.
        database
            The database that the view is located in.
        force
            If `True`, do not raise an error if the view does not exist.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        src = sge.Drop(
            this=sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted),
            kind="VIEW",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass
    # 负责将 Ibis 的抽象语法树（AST）编译为 SQL，并在目标数据库中运行，最后将返回的原始数据转换为 Pandas 格式。
    # self: 指向当前的后端实例（如 DuckDBBackend, PostgreSQLBackend 等）
    # expr: ir.Expr: 用户定义的 Ibis 表达式（可以是 Table 也可以是 Value，如 t.a + 1）。这是执行的核心目标
    # params: Mapping[ir.Scalar, Any] | None: 参数映射表。允许用户在执行时动态替换表达式中的占位符（ibis.param）
    # limit: int | str | None: 结果集行数限制。如果设为整数，SQL 编译时会自动添加 LIMIT 子句。
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

        Returns
        -------
        DataFrame | Series | scalar
            The result of the expression execution.
        """
        # 执行查询前的检查钩子（如验证连接是否有效、处理特定的懒加载逻辑）。
        self._run_pre_execute_hooks(expr)
        # 核心归一化步骤。
        # 无论用户传入的是单列表达式（Value）还是整表（Table），统一将其转换为表结构，以便后续生成统一的 SELECT 语句。
        table = expr.as_table()
        # 调用后端的编译器，将 Ibis 的 IR 树转化为目标数据库的 SQL 字符串。
        # 它会处理 params 的绑定（SQL 注入防御）以及 limit 的添加。
        sql = self.compile(table, params=params, limit=limit, **kwargs)
        # 获取结果集的模式定义（列名、数据类型）。这对于后续从数据库游标（Cursor）中正确反序列化数据至关重要。
        schema = table.schema()

        # TODO(kszucs): these methods should be abstractmethods or this default
        # implementation should be removed
        # self._safe_raw_sql(sql): 一个上下文管理器，负责发送 SQL 到数据库并返回一个游标（Cursor）。它处理了底层连接的安全性（如确保连接开启、异常捕获
        with self._safe_raw_sql(sql) as cur:
            # self._fetch_from_cursor(cur, schema): 关键的数据转换层。它从游标中提取二进制/原始数据，并根据之前获取的 schema，利用底层库（如 pyarrow 或 dbapi）将其转化为高效的 Python 数据结构。
            result = self._fetch_from_cursor(cur, schema)
        return expr.__pandas_result__(result)

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: tuple[str, str] | str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table from the backend.

        Parameters
        ----------
        name
            The name of the table to drop
        database
            The database that the table is located in.
        force
            If `True`, do not raise an error if the table does not exist.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        drop_stmt = sge.Drop(
            kind="TABLE",
            this=sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted),
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass
    # Ibis 后端中用于大规模数据分块读取（Streaming/Chunking）的核心生成器方法
    # 作用是避免一次性将海量查询结果载入内存，而是通过游标按批次拉取数据，是实现高效内存管理的基石。
    def _cursor_batches(
        self,
        # expr: ir.Expr: 需要执行的 Ibis 查询表达式。
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1 << 20,
    ) -> Iterable[list]:
        # 在执行查询前运行预处理逻辑（如验证连接状态、环境清理或自动注册 UDF），确保后续 SQL 执行环境就绪。
        self._run_pre_execute_hooks(expr)
        # 编译与执行
        with self._safe_raw_sql(
            self.compile(expr, limit=limit, params=params)
        ) as cursor:
            # 分块迭代 (Chunking Logic)
            while batch := cursor.fetchmany(chunk_size):
                yield batch

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        """Execute expression and return an iterator of PyArrow record batches.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            Ibis expression to export to pyarrow
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        chunk_size
            Maximum number of rows in each returned record batch.

        Returns
        -------
        RecordBatchReader
            Collection of pyarrow `RecordBatch`s.
        """
        pa = self._import_pyarrow()

        schema = expr.as_table().schema()
        array_type = schema.as_struct().to_pyarrow()
        arrays = (
            pa.array(map(tuple, batch), type=array_type)
            for batch in self._cursor_batches(
                expr, params=params, limit=limit, chunk_size=chunk_size
            )
        )
        batches = map(pa.RecordBatch.from_struct_array, arrays)

        return pa.ipc.RecordBatchReader.from_batches(schema.to_pyarrow(), batches)
    # Ibis 后端中用于将数据写入现有数据库表的核心方法
    # 支持将 Ibis 表达式（查询结果）或内存数据（如列表、字典、Pandas DataFrame）插入到目标表中。
    # name: str: 目标表的名称。
    # obj: ir.Table | IntoMemtable: 插入的数据源。
    def insert(
        self,
        name: str,
        /,
        obj: ir.Table | IntoMemtable,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Insert data into a table.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        name
            The name of the table to which data will be inserted
        obj
            The source data or expression to insert
        database
            Name of the attached database that the table is located in.

            For backends that support multi-level table hierarchies, you can
            pass in a dotted string path like `"catalog.database"` or a tuple of
            strings like `("catalog", "database")`.
        overwrite
            If `True` then replace existing contents of table
        """
        # 第一阶段：命名空间解析 (Location Resolution)
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)
        # 第二阶段：覆盖模式处理 (Overwrite Handling)
        if overwrite:
            self.truncate_table(name, database=(catalog, db))
        # 第三阶段：数据源归一化 (Data Normalization)
        # 如果 obj 不是 Ibis 的表表达式（即用户传入了原始的 Pandas/Arrow 数据），调用 ibis.memtable() 将其包装为 Ibis 可识别的虚拟内存表。
        # 确保了后端逻辑可以统一处理 SQL 表源和内存数据源。
        if not isinstance(obj, ir.Table):
            obj = ibis.memtable(obj)
        # 触发预处理逻辑，例如检查数据类型兼容性，或如果插入的是内存表，自动执行临时表上传操作（将 Python 内存数据发送到数据库服务端）。
        self._run_pre_execute_hooks(obj)
        # 构建与执行 SQL (Build & Execute)
        query = self._build_insert_from_table(
            target=name, source=obj, db=db, catalog=catalog
        )
        # 根据数据库方言生成对应的插入 SQL（例如 INSERT INTO ... SELECT ... 或 INSERT INTO ... VALUES (...)）。
        with self._safe_raw_sql(query):
            pass
    # Ibis 后端中用于自动映射插入列（Column Mapping）的核心逻辑。
    # 作用是决定在执行 INSERT 语句时，SQL 中应该显式列出哪些字段。
    # 在执行 INSERT INTO target (...) SELECT ... 时，Ibis 需要确定括号中字段的顺序。该方法通过对比“源数据模式（Source Schema）”与“目标表模式（Target Schema）”来做出智能决策：
    # 按名称映射：如果源列是目标列的子集，则按名称匹配插入。
    # 按位置映射：如果不匹配（例如列名不一致或源表无列名），则回退到按位置（Positional）顺序插入。
    def _get_columns_to_insert(
        self, *, target: str, source, db: str | None = None, catalog: str | None = None
    ):
        # Compare the columns between the target table and the object to be inserted
        # If source is a subset of target, use source columns for insert list
        # Otherwise, assume auto-generated column names and use positional ordering.
        # 第一阶段：获取目标表架构
        target_cols = self.get_schema(target, catalog=catalog, database=db).keys()
        # 情况 A (Subset)：如果 source_cols 是 target_cols 的子集，说明用户明确提供了列名且匹配成功。方法返回 source_cols，生成的 SQL 会类似于 INSERT INTO target (col_a, col_b) SELECT ...。
        # 情况 B (Mismatch/Full)：如果不是子集（例如列名不一致），则放弃列名匹配，返回 target_cols。此时生成的 SQL 会采取“全量/位置”映射方式，类似于 INSERT INTO target SELECT * FROM ...，由数据库引擎根据字段顺序进行隐式匹配。
        return (
            source_cols
            if (source_cols := source.schema().keys()) <= target_cols
            else target_cols
        )

    def _build_insert_from_table(
        self, *, target: str, source, db: str | None = None, catalog: str | None = None
    ):
        compiler = self.compiler
        quoted = compiler.quoted

        columns = self._get_columns_to_insert(
            target=target, source=source, db=db, catalog=catalog
        )

        query = sge.insert(
            expression=self.compile(source),
            into=sg.table(target, db=db, catalog=catalog, quoted=quoted),
            columns=[sg.to_identifier(col, quoted=quoted) for col in columns],
            dialect=compiler.dialect,
        )
        return query

    def _build_insert_template(
        self,
        name,
        *,
        schema: sch.Schema,
        catalog: str | None = None,
        columns: bool = False,
        placeholder: str = "?",
    ) -> str:
        """Builds an INSERT INTO table VALUES query string with placeholders.

        Parameters
        ----------
        name
            Name of the table to insert into
        schema
            Ibis schema of the table to insert into
        catalog
            Catalog name of the table to insert into
        columns
            Whether to render the columns to insert into
        placeholder
            Placeholder string.

        Returns
        -------
        str
            The query string
        """
        quoted = self.compiler.quoted
        return sge.insert(
            sge.Values(
                expressions=[
                    sge.Tuple(
                        expressions=[
                            sge.Var(this=placeholder.format(i=i, name=name))
                            for i, name in enumerate(schema.keys())
                        ]
                    )
                ]
            ),
            into=sg.table(name, catalog=catalog, quoted=quoted),
            columns=(
                map(partial(sg.to_identifier, quoted=quoted), schema.keys())
                if columns
                else None
            ),
        ).sql(self.dialect)

    def upsert(
        self,
        name: str,
        /,
        obj: ir.Table | IntoMemtable,
        on: str,
        *,
        database: str | None = None,
    ) -> None:
        """Upsert data into a table.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        name
            The name of the table to which data will be upserted
        obj
            The source data or expression to upsert
        on
            Column name to join on
        database
            Name of the attached database that the table is located in.

            For backends that support multi-level table hierarchies, you can
            pass in a dotted string path like `"catalog.database"` or a tuple of
            strings like `("catalog", "database")`.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        if not isinstance(obj, ir.Table):
            obj = ibis.memtable(obj)

        self._run_pre_execute_hooks(obj)

        query = self._build_upsert_from_table(
            target=name, source=obj, on=on, db=db, catalog=catalog
        )

        with self._safe_raw_sql(query):
            pass
    # Ibis 编译流水线中负责生成标准 INSERT INTO ... SELECT 语句的核心构建方法。它将 Ibis 的查询表达式（source）转化为一段跨数据库兼容的 SQL 插入语句。
    def _build_upsert_from_table(
        self,
        *,
        target: str,
        source,
        on: str,
        db: str | None = None,
        catalog: str | None = None,
    ):
        compiler = self.compiler
        quoted = compiler.quoted

        columns = self._get_columns_to_insert(
            target=target, source=source, db=db, catalog=catalog
        )

        source_alias = util.gen_name("source")
        target_alias = util.gen_name("target")
        query = sge.merge(
            sge.When(
                matched=True,
                then=sge.Update(
                    expressions=[
                        sg.column(col, quoted=quoted).eq(
                            sg.column(col, table=source_alias, quoted=quoted)
                        )
                        for col in columns
                        if col != on
                    ]
                ),
            ),
            sge.When(
                matched=False,
                then=sge.Insert(
                    this=sge.Tuple(
                        expressions=[sg.column(col, quoted=quoted) for col in columns]
                    ),
                    expression=sge.Tuple(
                        expressions=[
                            sg.column(col, table=source_alias, quoted=quoted)
                            for col in columns
                        ]
                    ),
                ),
            ),
            into=sg.table(target, db=db, catalog=catalog, quoted=quoted).as_(
                sg.to_identifier(target_alias, quoted=quoted), table=True
            ),
            using=f"({self.compile(source)}) AS {sg.to_identifier(source_alias, quoted=quoted)}",
            on=sge.Paren(
                this=sg.column(on, table=target_alias, quoted=quoted).eq(
                    sg.column(on, table=source_alias, quoted=quoted)
                )
            ),
            dialect=compiler.dialect,
        )
        return query

    def truncate_table(
        self, name: str, /, *, database: str | tuple[str, str] | None = None
    ) -> None:
        """Delete all rows from a table.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        name
            Table name
        database
            Name of the attached database that the table is located in.

            For backends that support multi-level table hierarchies, you can
            pass in a dotted string path like `"catalog.database"` or a tuple of
            strings like `("catalog", "database")`.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        ident = sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted).sql(
            self.dialect
        )
        with self._safe_raw_sql(f"TRUNCATE TABLE {ident}"):
            pass

    @util.experimental
    @classmethod
    def from_connection(cls, con: Any, /, **kwargs: Any) -> BaseBackend:
        """Create an Ibis client from an existing connection.

        Parameters
        ----------
        con
            An existing connection.
        **kwargs
            Extra arguments to be applied to the newly-created backend.
        """
        raise NotImplementedError(
            f"{cls.name} backend cannot be constructed from an existing connection"
        )

    def disconnect(self):
        """Disconnect from the backend."""
        # This is part of the Python DB-API specification so should work for
        # _most_ sqlglot backends
        self.con.close()

    def _to_catalog_db_tuple(self, table_loc: sge.Table):
        if (sg_cat := table_loc.args["catalog"]) is not None:
            sg_cat.args["quoted"] = False
            sg_cat = sg_cat.sql(self.dialect)
        if (sg_db := table_loc.args["db"]) is not None:
            sg_db.args["quoted"] = False
            sg_db = sg_db.sql(self.dialect)

        return sg_cat, sg_db

    # Ibis 后端中负责数据库命名空间标准化（Normalization）的核心工具方法。
    # 由于不同数据库（如 MySQL 的 db.table 与 BigQuery 的 project.dataset.table）对层级结构的定义各不相同，Ibis 统一使用 sqlglot.expressions.Table 来规范化这些路径。
    # 回值 (sge.Table): 一个包含标准化 catalog 和 db 属性的 SQLGlot 表对象。
    def _to_sqlglot_table(self, database: None | str | tuple[str, str]) -> sge.Table:
        # 获取当前方言的引用规则（是否需要双引号）和方言类型，确保生成的标识符符合目标数据库的语法。
        quoted = self.compiler.quoted
        dialect = self.dialect
        # 分支 A：空值处理
        # 如果未指定，返回空的表对象，后续 SQL 编译时会使用数据库的默认当前上下文。
        if database is None:
            # Create "table" with empty catalog and db
            sgt = sge.Table(catalog=None, db=None)
        # 分支 B：元组格式处理 (('cat', 'db'))
        elif isinstance(database, (list, tuple)):
            if len(database) > 2:
                raise ValueError(
                    "Only database hierarchies of two or fewer levels are supported."
                    "\nYou can specify ('catalog', 'database')."
                )
            elif len(database) == 2:
                catalog, database = database
            elif len(database) == 1:
                database = database[0]
                catalog = None
            else:
                raise ValueError(
                    f"Malformed database tuple {database} provided"
                    "\nPlease specify one of:"
                    '\n("catalog", "database")'
                    '\n("database",)'
                )
            sgt = sge.Table(
                catalog=sg.to_identifier(catalog, quoted=quoted),
                db=sg.to_identifier(database, quoted=quoted),
            )
        elif isinstance(database, str):
            # There is no definition of a sqlglot catalog.database hierarchy outside
            # of the standard table expression.
            # sqlglot parsing of the string will assume that it's a Table
            # so we unpack the arguments into a new sqlglot object, switching
            # table (this) -> database (db) and database (db) -> catalog
            sgt = sg.parse_one(
                ".".join(
                    sg.to_identifier(part, quoted=quoted).sql(dialect)
                    for part in database.split(".")
                ),
                into=sge.Table,
                dialect=dialect,
            )
            if sgt.args["catalog"] is not None:
                raise exc.IbisInputError(
                    f"Overspecified table hierarchy provided: `{sgt.sql(dialect)}`"
                )
            catalog = sgt.args["db"]
            db = sgt.args["this"]
            sgt = sge.Table(catalog=catalog, db=db)
        else:
            raise ValueError(
                """Invalid database hierarchy format.  Please use either dotted
                strings ('catalog.database') or tuples ('catalog', 'database')."""
            )

        return sgt

    def _register_builtin_udf(self, udf_node: ops.ScalarUDF) -> None:
        """No-op."""

    def _register_python_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(
            f"Python UDFs are not supported in the {self.dialect} backend"
        )

    def _register_pyarrow_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(
            f"PyArrow UDFs are not supported in the {self.dialect} backend"
        )

    def _register_pandas_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(
            f"pandas UDFs are not supported in the {self.dialect} backend"
        )

    def _make_memtable_finalizer(self, name: str) -> Callable[..., None]:
        this = sg.table(name, quoted=self.compiler.quoted)
        drop_stmt = sge.Drop(kind="TABLE", this=this, exists=True)
        drop_sql = drop_stmt.sql(self.dialect)

        def finalizer(drop_sql=drop_sql, con=self.con) -> None:
            with con.cursor() as cursor:
                cursor.execute(drop_sql)

        return finalizer
