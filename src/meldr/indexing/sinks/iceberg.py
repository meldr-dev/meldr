"""Apache Iceberg table sink that works with all PyIceberg catalogs (REST, Glue, Hive, SQL, ...).

Each write is one Iceberg transaction (schema update, delete of replaced keys, append), so
Spark, Trino, Snowflake, BigQuery, DuckDB, and Polars readers see all of a batch or none of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from meldr.core.errors import MissingExtraError, SinkError
from meldr.indexing.sinks.base import (
    SEQUENCE,
    WriteResult,
    fresh_rows,
    keep_latest,
    key_groups,
    removable,
    safe_identifier,
    sink_failure,
)

try:
    from pyiceberg.catalog import Catalog, load_catalog
    from pyiceberg.exceptions import CommitFailedException, NoSuchTableError
    from pyiceberg.expressions import And, BooleanExpression, EqualTo, In
    from pyiceberg.table import Table
except ImportError as e:
    raise MissingExtraError("iceberg", "The Iceberg table sink") from e

# PyIceberg's pydantic models show type checkers a field-based __init__, not the runtime one.
_In: Any = In
_EqualTo: Any = EqualTo


def _filters(keys: pl.DataFrame, key: Sequence[str]) -> list[BooleanExpression]:
    out: list[BooleanExpression] = []
    for fixed, values in key_groups(keys, key):
        expr: BooleanExpression = _In(key[-1], values)
        for k, v in fixed.items():
            expr = And(_EqualTo(k, v), expr)
        out.append(expr)
    return out


class IcebergSink:
    """Iceberg tables `{namespace}.{entity}`.

    Upserts replace rows by key, and the newer sequence wins.
    """

    def __init__(
        self,
        catalog: str | Catalog = "default",
        *,
        namespace: str = "meldr",
        properties: Mapping[str, Any] | None = None,
    ) -> None:
        self._catalog = (
            load_catalog(catalog, **dict(properties or {})) if isinstance(catalog, str) else catalog
        )
        self.namespace = safe_identifier(namespace)
        self._catalog.create_namespace_if_not_exists(self.namespace)

    def _load(self, entity: str) -> Table | None:
        try:
            return self._catalog.load_table((self.namespace, safe_identifier(entity)))
        except NoSuchTableError:
            return None

    @staticmethod
    def _stored(table: Table, keys: pl.DataFrame, key: Sequence[str]) -> pl.DataFrame:
        names = set(table.schema().column_names)
        fields = tuple(c for c in (*key, SEQUENCE) if c in names)
        frames = [
            pl.DataFrame(table.scan(row_filter=f, selected_fields=fields).to_arrow())
            for f in _filters(keys, key)
        ]
        return pl.concat(frames) if frames else pl.DataFrame()

    def write(self, entity: str, df: pl.DataFrame, *, key: Sequence[str]) -> WriteResult:
        target = f"iceberg:{self.namespace}.{entity}"
        if df.height == 0:
            return WriteResult(rows_written=0, target=target)
        incoming = keep_latest(df, key)
        schema = incoming.head(0).to_arrow().schema
        table = self._load(entity) or self._catalog.create_table(
            (self.namespace, safe_identifier(entity)), schema=schema
        )
        stored = self._stored(table, incoming, key)
        incoming = fresh_rows(incoming, stored, key)
        replaced = removable(incoming.select(key), stored, key)
        try:
            with table.transaction() as tx:
                if set(incoming.columns) - set(table.schema().column_names):
                    with tx.update_schema() as update:
                        update.union_by_name(schema)
                for f in _filters(replaced, key):
                    tx.delete(delete_filter=f)
                if incoming.height:
                    tx.append(incoming.to_arrow())
        except CommitFailedException as e:
            raise SinkError(
                f"A different writer committed to {target} first. "
                "Run the write again (writes are idempotent)."
            ) from e
        except (ValueError, TypeError, OSError) as e:  # schema mismatches surface as ValueError
            raise sink_failure(f"Iceberg write to {target}", e) from e
        return WriteResult(rows_written=incoming.height, target=target)

    def delete(self, entity: str, keys: pl.DataFrame) -> None:
        table = self._load(entity)
        if table is None or keys.height == 0:
            return
        key = [c for c in keys.columns if c != SEQUENCE]
        victims = removable(keys, self._stored(table, keys, key), key)
        if victims.height:
            with table.transaction() as tx:
                for f in _filters(victims, key):
                    tx.delete(delete_filter=f)

    def read(self, entity: str) -> pl.DataFrame:
        table = self._load(entity)
        return pl.DataFrame(table.scan().to_arrow()) if table is not None else pl.DataFrame()

    def describe(self, entity: str, *, table: str, columns: Mapping[str, str]) -> None:
        """Set the column docs and the table `comment` property, but skip unchanged values."""
        t = self._load(entity)
        if t is None:
            return
        schema = t.schema()
        names = set(schema.column_names)
        docs = {c: d for c, d in columns.items() if c in names and schema.find_field(c).doc != d}
        if not docs and t.properties.get("comment") == table:
            return
        with t.transaction() as tx:
            if docs:
                with tx.update_schema() as update:
                    for column, doc in docs.items():
                        update.update_column(column, doc=doc)
            tx.set_properties(comment=table)
