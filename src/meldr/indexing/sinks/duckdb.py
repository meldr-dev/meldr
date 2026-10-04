"""DuckDB table sink: one table for each entity, with upserts by key."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

import duckdb
import polars as pl

from meldr.connectors.sql import check_read_only
from meldr.core.errors import ConnectorError, SinkError
from meldr.indexing.sinks.base import (
    SEQUENCE,
    WriteResult,
    keep_latest,
    safe_identifier,
    sink_failure,
    sql_literal,
)


def _q(name: str) -> str:
    return '"' + safe_identifier(name) + '"'


class DuckDBSink:
    """Harmonized tables in a single DuckDB file (default `{root}/tables/meldr.duckdb`).

    External access is disabled: queries cannot read other files, URLs, or attach databases.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._conn = duckdb.connect(self.path, config={"enable_external_access": False})
        self._lock = threading.Lock()

    def _exists(self, table: str) -> bool:
        row = self._conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
        return bool(row and row[0])

    def _columns(self, table: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? "
            "ORDER BY ordinal_position",
            [table],
        ).fetchall()
        return [r[0] for r in rows]

    def write(self, entity: str, df: pl.DataFrame, *, key: Sequence[str]) -> WriteResult:
        table = safe_identifier(entity)
        if df.height == 0:
            return WriteResult(rows_written=0, target=f"duckdb:{table}")
        incoming = keep_latest(df, key)
        cols = ", ".join(_q(c) for c in incoming.columns)
        insert = f"INTO {_q(table)} ({cols}) SELECT {cols} FROM meldr_incoming"  # noqa: S608
        if SEQUENCE in incoming.columns:
            updates = ", ".join(
                f"{_q(c)} = excluded.{_q(c)}" for c in incoming.columns if c not in key
            )
            current = f"{_q(table)}.{_q(SEQUENCE)}"
            sql = (
                f"INSERT {insert} ON CONFLICT ({', '.join(_q(k) for k in key)}) DO UPDATE SET "
                f"{updates} WHERE {current} IS NULL OR excluded.{_q(SEQUENCE)} >= {current}"
            )
        else:
            sql = f"INSERT OR REPLACE {insert}"
        with self._lock:
            self._conn.register("meldr_incoming", incoming.to_arrow())
            try:
                self._ensure_table(table, key)
                self._conn.execute(sql)
            except duckdb.Error as e:
                if "lock" in str(e).casefold():
                    raise SinkError(
                        "DuckDB is locked. Close the other writer or use one sink instance."
                    ) from e
                raise sink_failure(f"DuckDB write to {table}", e) from e
            finally:
                self._conn.unregister("meldr_incoming")
        return WriteResult(rows_written=incoming.height, target=f"duckdb:{table}")

    def _ensure_table(self, table: str, key: Sequence[str]) -> None:
        described = self._conn.execute("DESCRIBE SELECT * FROM meldr_incoming").fetchall()
        if not self._exists(table):
            defs = [f"{_q(name)} {dtype}" for name, dtype, *_ in described]
            defs.append(f"PRIMARY KEY ({', '.join(_q(k) for k in key)})")
            self._conn.execute(f"CREATE TABLE {_q(table)} ({', '.join(defs)})")
            return
        existing = set(self._columns(table))
        for name, dtype, *_ in described:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE {_q(table)} ADD COLUMN {_q(name)} {dtype}")

    def delete(self, entity: str, keys: pl.DataFrame) -> None:
        table = safe_identifier(entity)
        if keys.height == 0:
            return
        match = [f"{_q(table)}.{_q(c)} = meldr_keys.{_q(c)}" for c in keys.columns if c != SEQUENCE]
        with self._lock:
            if not self._exists(table):
                return
            if SEQUENCE in keys.columns and SEQUENCE in self._columns(table):
                current = f"{_q(table)}.{_q(SEQUENCE)}"
                match.append(f"({current} IS NULL OR {current} <= meldr_keys.{_q(SEQUENCE)})")
            self._conn.register("meldr_keys", keys.to_arrow())
            try:
                self._conn.execute(
                    f"DELETE FROM {_q(table)} USING meldr_keys WHERE {' AND '.join(match)}"  # noqa: S608
                )
            except duckdb.Error as e:
                raise sink_failure(f"Removing rows from DuckDB table {table}", e) from e
            finally:
                self._conn.unregister("meldr_keys")

    def describe(self, entity: str, *, table: str, columns: Mapping[str, str]) -> None:
        """Store descriptions as table and column comments (`duckdb_columns().comment`)."""
        name = safe_identifier(entity)
        with self._lock:
            if not self._exists(name):
                return
            present = set(self._columns(name))
            self._conn.execute(f"COMMENT ON TABLE {_q(name)} IS {sql_literal(table)}")
            for column, doc in columns.items():
                if column in present:
                    self._conn.execute(
                        f"COMMENT ON COLUMN {_q(name)}.{_q(column)} IS {sql_literal(doc)}"
                    )

    def read(self, entity: str) -> pl.DataFrame:
        table = safe_identifier(entity)
        with self._lock:
            if not self._exists(table):
                return pl.DataFrame()
            return self._conn.execute(f"SELECT * FROM {_q(table)}").pl()  # noqa: S608

    def sql(self, query: str) -> pl.DataFrame:
        """Run one read-only SELECT/WITH query against the harmonized tables."""
        try:
            checked = check_read_only(query)
        except ConnectorError as e:
            raise SinkError(str(e)) from e
        with self._lock:
            try:
                return self._conn.execute(checked).pl()
            except duckdb.Error as e:
                raise SinkError(f"DuckDB query failed: {e}") from e

    def close(self) -> None:
        self._conn.close()
