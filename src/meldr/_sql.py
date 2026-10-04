"""SQLAlchemy helpers for the SQL connector and the SQL table sink (the sql extra is necessary)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import polars as pl

from meldr.core.errors import MissingExtraError
from meldr.secrets import SecretRef

_DIALECT_EXTRAS = {
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "databricks": "databricks",
}


def make_engine(url: str | SecretRef, feature: str) -> Any:
    try:
        import sqlalchemy as sa
    except ImportError as e:
        raise MissingExtraError("sql", feature) from e
    resolved = url.reveal() if isinstance(url, SecretRef) else url
    try:
        engine = sa.create_engine(resolved, pool_pre_ping=True)
        if resolved.lower().startswith("sqlite"):

            def on_connect(dbapi_connection: Any, _: Any) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA busy_timeout=5000")
                if ":memory:" not in resolved:
                    cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

            sa.event.listen(engine, "connect", on_connect)
        return engine
    except sa.exc.NoSuchModuleError as e:
        scheme = resolved.partition(":")[0].lower()
        extra = _DIALECT_EXTRAS.get(scheme)
        if extra is None:
            raise
        raise MissingExtraError(extra, feature) from e


def rows_frame(rows: Iterable[Sequence[Any]], columns: Sequence[str]) -> pl.DataFrame:
    """Result rows as a DataFrame. Driver objects (UUID, INET, ...) become strings."""
    df = pl.DataFrame(
        [tuple(r) for r in rows], schema=list(columns), orient="row", infer_schema_length=None
    )
    objects = [n for n, t in df.schema.items() if t == pl.Object]
    return df.with_columns(pl.col(objects).map_elements(str, return_dtype=pl.String))
