"""Delta Lake table sink: one Delta table for each entity in a local path or object-store URI.

Each upsert is one atomic MERGE, readable by Spark, Databricks, Trino, DuckDB, Polars, and
warehouses with Delta support. `storage_options` holds object-store settings (s3://, gs://,
abfss://).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from meldr.core.errors import MissingExtraError
from meldr.indexing.sinks.base import (
    SEQUENCE,
    WriteResult,
    keep_latest,
    safe_identifier,
    sink_failure,
)

try:
    from deltalake import DeltaTable, Field
    from deltalake.exceptions import TableNotFoundError
except ImportError as e:
    raise MissingExtraError("delta", "The Delta Lake table sink") from e


def _col(alias: str, name: str) -> str:
    return f'{alias}."{safe_identifier(name)}"'


_T, _S = _col("t", SEQUENCE), _col("s", SEQUENCE)
_NOT_OLDER = f"{_T} IS NULL OR {_S} >= {_T}"  # update only when the source row is not older
_NOT_NEWER = f"{_T} IS NULL OR {_T} <= {_S}"  # delete only rows not newer than the delete


class DeltaSink:
    """`{root}/{entity}` Delta tables: upserts merge on the key. A newer `_meldr_sequence` wins."""

    def __init__(
        self, root: str | Path, *, storage_options: Mapping[str, str] | None = None
    ) -> None:
        self.root = str(root).rstrip("/")
        self._options = dict(storage_options) if storage_options else None

    def _uri(self, entity: str) -> str:
        return f"{self.root}/{safe_identifier(entity)}"

    def _table(self, entity: str) -> DeltaTable | None:
        try:
            return DeltaTable(self._uri(entity), storage_options=self._options)
        except TableNotFoundError:
            return None

    def _merge(self, df: pl.DataFrame, entity: str, key: Sequence[str]) -> Any:
        on = " AND ".join(f"{_col('t', k)} = {_col('s', k)}" for k in key)
        return df.write_delta(
            self._uri(entity),
            mode="merge",
            storage_options=self._options,
            delta_merge_options={"predicate": on, "source_alias": "s", "target_alias": "t"},
        )

    @staticmethod
    def _columns(table: DeltaTable) -> set[str]:
        return set(table.schema().to_arrow().names)

    def write(self, entity: str, df: pl.DataFrame, *, key: Sequence[str]) -> WriteResult:
        target = f"delta:{self._uri(entity)}"
        if df.height == 0:
            return WriteResult(rows_written=0, target=target)
        incoming = keep_latest(df, key)
        try:
            table = self._table(entity)
            if table is not None:
                # MERGE with schema evolution leaks new-column values into rows the
                # sequence guard skipped (deltalake 1.6), so evolve the schema first.
                new = [
                    f
                    for f in incoming.head(0).to_arrow().schema
                    if f.name not in self._columns(table)
                ]
                if new:
                    table.alter.add_columns([Field.from_arrow(f) for f in new])
            if table is None:
                incoming.write_delta(
                    self._uri(entity), mode="append", storage_options=self._options
                )
            else:
                guard = _NOT_OLDER if SEQUENCE in incoming.columns else None
                merge = self._merge(incoming, entity, key)
                merge.when_matched_update_all(
                    predicate=guard
                ).when_not_matched_insert_all().execute()
        except Exception as e:  # deltalake and arrow raise many unrelated error types
            raise sink_failure(f"Delta write to {entity}", e) from e
        return WriteResult(rows_written=incoming.height, target=target)

    def delete(self, entity: str, keys: pl.DataFrame) -> None:
        table = self._table(entity)
        if table is None or keys.height == 0:
            return
        key = [c for c in keys.columns if c != SEQUENCE]
        guarded = SEQUENCE in keys.columns and SEQUENCE in self._columns(table)
        guard = _NOT_NEWER if guarded else None
        self._merge(keep_latest(keys, key), entity, key).when_matched_delete(guard).execute()

    def read(self, entity: str) -> pl.DataFrame:
        if self._table(entity) is None:
            return pl.DataFrame()
        return pl.read_delta(self._uri(entity), storage_options=self._options)

    def describe(self, entity: str, *, table: str, columns: Mapping[str, str]) -> None:
        """Set the table description and column `comment` metadata, but skip unchanged values."""
        dt = self._table(entity)
        if dt is None:
            return
        if dt.metadata().description != table:
            dt.alter.set_table_description(table)
        fields = {f.name: f.metadata for f in dt.schema().fields}
        for column, doc in columns.items():
            if column in fields and fields[column].get("comment") != doc:
                dt.alter.set_column_metadata(column, {"comment": doc})
