"""Qdrant vector sink (optional extra `qdrant`)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from meldr.core.errors import MissingExtraError
from meldr.indexing.sinks.base import (
    IndexItem,
    SearchHit,
    WriteResult,
    matches,
    safe_identifier,
)

_NS = uuid.UUID("6f1c6f0e-8f0a-4b8e-9d7e-6a1f3c2b5d4e")
_POINTS_PER_REQUEST = 256  # keeps 1536-dim batches far below Qdrant's 32 MB request limit


class QdrantSink:
    """Points keyed by uuid5(id), with the original id and metadata in the payload."""

    def __init__(self, url: str | None = None, *, client: Any = None) -> None:
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise MissingExtraError("qdrant", "The Qdrant vector sink") from e
            client = QdrantClient(url=url) if url else QdrantClient(":memory:")
        self._client = client

    def _ensure(self, name: str, dim: int) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(name):
            self._client.create_collection(
                name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )

    def upsert(self, collection: str, items: Sequence[IndexItem]) -> WriteResult:
        from qdrant_client import models

        name = safe_identifier(collection)
        if items:
            self._ensure(name, len(items[0].vector))
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(_NS, i.id)),
                    vector=i.vector,
                    payload={"_id": i.id, "_text": i.text, **i.metadata},
                )
                for i in items
            ]
            for start in range(0, len(points), _POINTS_PER_REQUEST):
                self._client.upsert(name, points=points[start : start + _POINTS_PER_REQUEST])
        return WriteResult(rows_written=len(items), target=f"qdrant:{name}")

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        from qdrant_client import models

        name = safe_identifier(collection)
        if not self._client.collection_exists(name):
            return []
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key=key,
                match=models.MatchAny(any=list(v))
                if isinstance(v, list | tuple | set)
                else models.MatchValue(value=v),
            )
            for key, v in (filters or {}).items()
        ]
        result = self._client.query_points(
            name,
            query=list(vector),
            limit=k,
            query_filter=models.Filter(must=conditions) if conditions else None,
            with_payload=True,
        )
        hits = []
        for p in result.points:
            payload = dict(p.payload or {})
            item_id = str(payload.pop("_id", p.id))
            text = str(payload.pop("_text", ""))
            if matches(payload, filters):
                hits.append(
                    SearchHit(id=item_id, score=float(p.score), text=text, metadata=payload)
                )
        return hits
