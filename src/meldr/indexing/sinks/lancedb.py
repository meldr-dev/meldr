"""LanceDB vector sink (optional extra `lancedb`)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from meldr.core.errors import MissingExtraError, SinkError
from meldr.indexing.sinks.base import (
    OVERFETCH,
    IndexItem,
    SearchHit,
    WriteResult,
    dumps_meta,
    matches,
    safe_identifier,
    scalar_filters,
    sql_literal,
)

# Metadata keys stored as filterable string columns (others live only in metadata_json).
FILTER_KEYS = ("entity", "source_id", "record_id", "spec_id", "kind")


class LanceDBSink:
    """Embedded LanceDB at `{root}/vectors`.

    The sink pushes filters on FILTER_KEYS down to Lance and applies the other filters in Python.
    """

    def __init__(self, path: str | Path) -> None:
        try:
            import lancedb
        except ImportError as e:
            raise MissingExtraError("lancedb", "The LanceDB vector sink") from e
        Path(path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(path))

    @staticmethod
    def _row(item: IndexItem) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": item.id,
            "vector": item.vector,
            "text": item.text,
            "metadata_json": dumps_meta(item.metadata),
        }
        for k in FILTER_KEYS:
            v = item.metadata.get(k)
            row[f"m_{k}"] = None if v is None else str(v)
        return row

    def _open(self, name: str) -> Any | None:
        try:
            return self._db.open_table(name)
        except ValueError:  # a missing table makes lancedb raise ValueError
            return None

    def upsert(self, collection: str, items: Sequence[IndexItem]) -> WriteResult:
        name = safe_identifier(collection)
        if not items:
            return WriteResult(rows_written=0, target=f"lancedb:{name}")
        rows = [self._row(i) for i in items]
        try:
            table = self._open(name)
            if table is None:
                import pyarrow as pa

                dim = len(items[0].vector)
                schema = pa.schema(
                    [
                        pa.field("id", pa.string()),
                        pa.field("vector", pa.list_(pa.float32(), dim)),
                        pa.field("text", pa.string()),
                        pa.field("metadata_json", pa.string()),
                        *[pa.field(f"m_{k}", pa.string()) for k in FILTER_KEYS],
                    ]
                )
                self._db.create_table(name, data=rows, schema=schema)
            else:
                (
                    table.merge_insert("id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(rows)
                )
        except Exception as e:
            raise SinkError(f"LanceDB upsert into {name} failed: {type(e).__name__}") from e
        return WriteResult(rows_written=len(rows), target=f"lancedb:{name}")

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        table = self._open(safe_identifier(collection))
        if table is None:
            return []
        query = table.search(list(vector)).distance_type("cosine")
        pushed = {key: v for key, v in scalar_filters(filters).items() if key in FILTER_KEYS}
        if pushed:
            query = query.where(
                " AND ".join(f"m_{key} = {sql_literal(v)}" for key, v in pushed.items())
            )
        rest = {key: v for key, v in (filters or {}).items() if key not in pushed}
        rows = query.limit(k * OVERFETCH if rest else k).to_list()
        hits: list[SearchHit] = []
        for r in rows:
            meta = json.loads(r["metadata_json"])
            if not matches(meta, rest):
                continue
            hits.append(
                SearchHit(
                    id=r["id"], score=1.0 - float(r["_distance"]), text=r["text"], metadata=meta
                )
            )
        return hits[:k]
