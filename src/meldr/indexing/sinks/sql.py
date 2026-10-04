"""Table sink for all databases and warehouses with a SQLAlchemy dialect.

Postgres, MySQL, SQL Server, Oracle, and SQLite use SQLAlchemy's built-in dialects. Warehouses
use their own dialect package (snowflake-sqlalchemy, sqlalchemy-bigquery, sqlalchemy-redshift,
databricks-sqlalchemy, trino, ...). An upsert deletes the keys of the batch and inserts its rows in
one transaction, which every dialect supports. The sink makes missing tables and adds new columns.
To control types, partitions, or cluster keys, make the tables yourself.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import polars as pl

from meldr._sql import make_engine, rows_frame
from meldr.core.errors import MissingExtraError, SinkError
from meldr.indexing.sinks.base import (
    SEQUENCE,
    WriteResult,
    fresh_rows,
    keep_latest,
    key_groups,
    removable,
    safe_identifier,
)
from meldr.observability import safe_log
from meldr.secrets import SecretRef

try:
    import sqlalchemy as sa
except ImportError as e:
    raise MissingExtraError("sql", "The SQL table sink") from e

# Keys for each IN (...) list. These dialects accept tens of thousands of bound parameters. Oracle
# caps IN lists at 1000 and SQL Server at 2100 parameters, so other dialects use the safe default.
_IN_LIST = {"postgresql": 20_000, "mysql": 20_000, "mariadb": 20_000, "sqlite": 20_000}
_IN_LIST_DEFAULT = 1000


def _column_type(dtype: pl.DataType, is_key: bool) -> Any:
    if is_key:
        return sa.String(255)
    if dtype.is_integer():
        return sa.BigInteger()
    if dtype.is_float():
        return sa.Float()
    if isinstance(dtype, pl.Decimal):
        return sa.Numeric(dtype.precision, dtype.scale)
    if isinstance(dtype, pl.Datetime):
        return sa.DateTime(timezone=dtype.time_zone is not None)
    simple = {pl.Boolean: sa.Boolean, pl.Date: sa.Date, pl.Time: sa.Time, pl.Binary: sa.LargeBinary}
    return simple.get(type(dtype), sa.Text)()


def _portable(df: pl.DataFrame) -> pl.DataFrame:
    """Lists and structs become JSON text so every database can store them."""
    nested = [n for n, t in df.schema.items() if isinstance(t, pl.List | pl.Array | pl.Struct)]
    return df.with_columns(
        pl.Series(n, [None if v is None else json.dumps(v, default=str) for v in df[n].to_list()])
        for n in nested
    )


class SQLTableSink:
    """Harmonized tables in a SQL database or warehouse (one writer for each table at a time)."""

    def __init__(
        self, url: str | SecretRef | None = None, *, schema: str | None = None, engine: Any = None
    ) -> None:
        if engine is None:
            if url is None:
                raise SinkError("The sql table sink has no `url`. Set `url` to a secret reference.")
            engine = make_engine(url, "The SQL table sink")
        self.engine = engine
        self.schema = schema
        self._tables: dict[str, sa.Table] = {}

    def _reflect(self, conn: sa.Connection, name: str) -> sa.Table | None:
        if name not in self._tables:
            if not sa.inspect(conn).has_table(name, schema=self.schema):
                return None
            self._tables[name] = sa.Table(
                name, sa.MetaData(), schema=self.schema, autoload_with=conn
            )
        return self._tables[name]

    def _ensure(
        self, conn: sa.Connection, name: str, df: pl.DataFrame, key: Sequence[str]
    ) -> sa.Table:
        table = self._reflect(conn, name)
        if table is None:
            columns = [
                sa.Column(safe_identifier(c), _column_type(t, c in key), primary_key=c in key)
                for c, t in df.schema.items()
            ]
            table = self._tables[name] = sa.Table(name, sa.MetaData(), *columns, schema=self.schema)
            table.create(conn)
            return table
        missing = [c for c in df.columns if c not in table.c]
        if not missing:
            return table
        add = "ADD" if conn.dialect.name == "mssql" else "ADD COLUMN"
        target = conn.dialect.identifier_preparer.format_table(table)
        for c in missing:
            column = sa.Column(safe_identifier(c), _column_type(df.schema[c], False))
            ddl = sa.schema.CreateColumn(column).compile(dialect=conn.dialect)
            conn.execute(sa.text(f"ALTER TABLE {target} {add} {ddl}"))
        del self._tables[name]
        return self._ensure(conn, name, df, key)

    @staticmethod
    def _where(
        conn: sa.Connection, table: sa.Table, keys: pl.DataFrame, key: Sequence[str]
    ) -> Iterator[Any]:
        size = _IN_LIST.get(conn.dialect.name, _IN_LIST_DEFAULT)
        for fixed, values in key_groups(keys, key):
            for i in range(0, len(values), size):
                yield sa.and_(
                    *(table.c[k] == v for k, v in fixed.items()),
                    table.c[key[-1]].in_(values[i : i + size]),
                )

    def _stored(
        self, conn: sa.Connection, table: sa.Table, keys: pl.DataFrame, key: Sequence[str]
    ) -> pl.DataFrame:
        cols = [table.c[c] for c in (*key, SEQUENCE) if c in table.c]
        select = sa.select(*cols)
        rows = [
            r
            for cond in self._where(conn, table, keys, key)
            for r in conn.execute(select.where(cond))
        ]
        return rows_frame(rows, [c.name for c in cols])

    def _delete(
        self, conn: sa.Connection, table: sa.Table, keys: pl.DataFrame, key: Sequence[str]
    ) -> None:
        for cond in self._where(conn, table, keys, key):
            conn.execute(table.delete().where(cond))

    def write(self, entity: str, df: pl.DataFrame, *, key: Sequence[str]) -> WriteResult:
        name = safe_identifier(entity)
        rows = _portable(keep_latest(df, key))
        if rows.height == 0:
            return WriteResult(rows_written=0, target=f"sql:{name}")
        try:
            with self.engine.begin() as conn:
                table = self._ensure(conn, name, rows, key)
                if SEQUENCE in rows.columns and SEQUENCE in table.c:
                    rows = fresh_rows(rows, self._stored(conn, table, rows, key), key)
                self._delete(conn, table, rows, key)
                if rows.height:
                    conn.execute(table.insert(), rows.to_dicts())
        except sa.exc.SQLAlchemyError as e:
            self._tables.pop(name, None)
            raise SinkError(f"SQL write to {name} failed ({type(e).__name__}).") from e
        return WriteResult(rows_written=rows.height, target=f"sql:{name}")

    def delete(self, entity: str, keys: pl.DataFrame) -> None:
        name = safe_identifier(entity)
        key = [c for c in keys.columns if c != SEQUENCE]
        with self.engine.begin() as conn:
            table = self._reflect(conn, name)
            if table is not None and keys.height:
                victims = removable(keys, self._stored(conn, table, keys, key), key)
                self._delete(conn, table, victims, key)

    def read(self, entity: str) -> pl.DataFrame:
        name = safe_identifier(entity)
        with self.engine.connect() as conn:
            table = self._reflect(conn, name)
            if table is None:
                return pl.DataFrame()
            result = conn.execute(sa.select(table))
            return rows_frame(result, list(result.keys()))

    def describe(self, entity: str, *, table: str, columns: Mapping[str, str]) -> None:
        """Table and column comments, on dialects that support them (best effort)."""
        name = safe_identifier(entity)
        try:
            with self.engine.begin() as conn:
                target = self._reflect(conn, name)
                if target is None or not conn.dialect.supports_comments:
                    return
                target.comment = table
                conn.execute(sa.schema.SetTableComment(target))
                for column, doc in columns.items():
                    if column in target.c:
                        target.c[column].comment = doc
                        conn.execute(sa.schema.SetColumnComment(target.c[column]))
        except sa.exc.SQLAlchemyError as e:
            safe_log("sink.describe_skipped", sink="sql", table=name, error=type(e).__name__)

    def close(self) -> None:
        """Close the SQLAlchemy engine."""
        self.engine.dispose()
