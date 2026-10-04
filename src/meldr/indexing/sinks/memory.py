"""In-process vector sink (default for tests and small pilots)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from meldr._math import cosine
from meldr.indexing.sinks.base import IndexItem, SearchHit, WriteResult, matches


class MemoryVectorSink:
    """Brute-force cosine search of all items in a dict."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, IndexItem]] = {}

    def upsert(self, collection: str, items: Sequence[IndexItem]) -> WriteResult:
        coll = self._data.setdefault(collection, {})
        for item in items:
            coll[item.id] = item
        return WriteResult(rows_written=len(items), target=f"memory:{collection}")

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        items = [i for i in self._data.get(collection, {}).values() if matches(i.metadata, filters)]
        scored = sorted(((cosine(vector, i.vector), i) for i in items), key=lambda p: -p[0])
        return [
            SearchHit(id=i.id, score=score, text=i.text, metadata=i.metadata)
            for score, i in scored[:k]
        ]
