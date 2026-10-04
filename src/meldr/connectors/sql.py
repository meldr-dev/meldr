"""Read-only SQL connector (SQLAlchemy 2).

Always use a database role with read-only grants. The checks in this module are only defense in
depth.

The connector stages the results to local Parquet before it yields the first batch. Because of
this, the source transaction lasts only as long as the read. It does not last through the
mapping, LLM, embedding, and write work that follows.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from meldr._sql import make_engine, rows_frame
from meldr.core.errors import ConnectorError
from meldr.core.types import Modality, RawBatch
from meldr.secrets import SecretRef

if TYPE_CHECKING:
    from meldr.config import SourceConfig

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|attach|detach|"
    r"pragma|copy|call|exec|execute|vacuum|reindex|lock|set|into)\b",
    re.IGNORECASE,
)
_LEADING = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_TAG_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def check_read_only(query: str) -> str:
    """Accept only a single SELECT/WITH statement with no keywords that change data."""
    no_comments = re.sub(r"--[^\n]*|/\*.*?\*/", " ", query, flags=re.DOTALL)
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", no_comments)
    body = no_strings.strip().rstrip(";").strip()
    if ";" in body:
        raise ConnectorError("The query has more than one SQL statement. Use only one statement.")
    if not _LEADING.match(body):
        raise ConnectorError(
            "The query is not a SELECT or WITH query. Use only read-only SELECT or WITH queries."
        )
    m = _FORBIDDEN.search(body)
    if m:
        raise ConnectorError(
            f"Read-only queries cannot contain the keyword '{m.group(1).upper()}'. Remove it."
        )
    return query.strip().rstrip(";")


class SQLConnector:
    """Reads a table (through reflection) or a single read-only query in batches."""

    modality = Modality.TABULAR

    def __init__(
        self,
        source_id: str,
        url: str | SecretRef,
        *,
        batch_size: int = 50_000,
        table: str | None = None,
        query: str | None = None,
        engine: Any = None,
    ) -> None:
        self.source_id = source_id
        self.batch_size = batch_size
        self.default_table = table
        self.default_query = query
        self._url = url
        self._engine = engine

    def _get_engine(self) -> Any:
        if self._engine is None:
            self._engine = make_engine(self._url, "The SQL connector")
        return self._engine

    def pull(self, **params: Any) -> Iterator[RawBatch]:
        """Params: `table` (optionally `schema`) or `query`."""
        import sqlalchemy as sa

        table = params.get("table", self.default_table)
        query = params.get("query", self.default_query)
        if bool(table) == bool(query):
            raise ConnectorError("SQL pull() must have `table` or `query`, but not the two.")
        engine = self._get_engine()
        meta: dict[str, Any] = {}
        if table:
            try:
                tbl = sa.Table(
                    table, sa.MetaData(), schema=params.get("schema"), autoload_with=engine
                )
            except sa.exc.NoSuchTableError as e:
                raise ConnectorError(f"Cannot find table '{table}'.") from e
            stmt: Any = sa.select(tbl).prefix_with(self._tag)
            meta["entity_hint"] = table
            meta["descriptions"] = {c.name: c.comment for c in tbl.columns if c.comment}
        else:
            stmt = sa.text(f"{self._tag} {check_read_only(str(query))}")
        with tempfile.TemporaryDirectory(prefix="meldr-sql-") as staging:
            for part in self._stage(engine, stmt, Path(staging)):
                yield RawBatch.from_polars(pl.read_parquet(part), self.source_id, **meta)

    @property
    def _tag(self) -> str:
        """Query comment that names the caller in `pg_stat_activity` and slow-query logs."""
        return f"/* meldr source={_TAG_UNSAFE.sub('_', self.source_id)} */"

    def _stage(self, engine: Any, stmt: Any, directory: Path) -> list[Path]:
        """Stream the result to Parquet parts. The connection closes before the first yield."""
        parts: list[Path] = []
        with engine.connect() as conn:
            self._read_only(conn)
            result = conn.execution_options(stream_results=True).execute(stmt)
            columns = list(result.keys())
            while rows := result.fetchmany(self.batch_size):
                parts.append(directory / f"part-{len(parts):06d}.parquet")
                rows_frame(rows, columns).write_parquet(parts[-1])
            conn.rollback()
        return parts

    @staticmethod
    def _read_only(conn: Any) -> None:
        import sqlalchemy as sa

        dialect = conn.dialect.name
        stmts = {
            "sqlite": "PRAGMA query_only = ON",
            "postgresql": "SET TRANSACTION READ ONLY",
            "mysql": "START TRANSACTION READ ONLY",
            "mariadb": "START TRANSACTION READ ONLY",
        }
        if dialect in stmts:
            conn.execute(sa.text(stmts[dialect]))


def from_config(source_id: str, cfg: SourceConfig, batch_size: int) -> SQLConnector:
    if cfg.url is None:
        raise ConnectorError(
            f"sources.{source_id}: the sql connector must have `url` (a secret ref)."
        )
    return SQLConnector(source_id, cfg.url, batch_size=batch_size, table=cfg.table, query=cfg.query)
