"""Chroma vector sink for local embedded search."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from meldr.core.errors import MissingExtraError, SinkError
from meldr.indexing.sinks.base import (
    IndexItem,
    SearchHit,
    WriteResult,
    matches,
    safe_identifier,
)

_META_JSON = "__meldr_metadata"


class ChromaSink:
    """Embedded Chroma storage with cosine search."""

    def __init__(self, path: str | Path | None = None) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise MissingExtraError("chromadb", "the Chroma vector sink") from e
        self._client = (
            chromadb.PersistentClient(path=str(path)) if path is not None else chromadb.Client()
        )

    @staticmethod
    def _name(collection: str) -> str:
        return f"hz_{safe_identifier(collection)}"

    def _collection(self, name: str) -> Any | None:
        try:
            return self._client.get_collection(name)
        except Exception as e:
            if type(e).__name__ == "NotFoundError":
                return None
            raise

    @staticmethod
    def _metadata(item: IndexItem) -> dict[str, str | int | float | bool]:
        data: dict[str, str | int | float | bool] = {
            _META_JSON: json.dumps(item.metadata, sort_keys=True, default=str)
        }
        for key, value in item.metadata.items():
            if value is not None:
                data[f"m_{key}"] = value
        return data

    @staticmethod
    def _where(filters: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        pushed: list[dict[str, Any]] = []
        rest: dict[str, Any] = {}
        for key, value in (filters or {}).items():
            values = list(value) if isinstance(value, list | tuple | set) else None
            if value is None or (values is not None and any(v is None for v in values)):
                rest[key] = value
            elif values is not None:
                pushed.append({f"m_{key}": {"$in": values}})
            else:
                pushed.append({f"m_{key}": value})
        if not pushed:
            return None, rest
        return (pushed[0] if len(pushed) == 1 else {"$and": pushed}), rest

    def upsert(self, collection: str, items: Sequence[IndexItem]) -> WriteResult:
        name = self._name(collection)
        if not items:
            return WriteResult(rows_written=0, target=f"chroma:{collection}")
        try:
            target: Any = self._collection(name)
            if target is None:
                target = self._client.get_or_create_collection(
                    name=name,
                    configuration={"hnsw": {"space": "cosine"}},
                )
            target.upsert(
                ids=[item.id for item in items],
                embeddings=[item.vector for item in items],
                documents=[item.text for item in items],
                metadatas=[self._metadata(item) for item in items],
            )
        except Exception as e:
            raise SinkError(f"Chroma upsert into {collection} failed ({type(e).__name__}).") from e
        return WriteResult(rows_written=len(items), target=f"chroma:{collection}")

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        name = self._name(collection)
        target = self._collection(name)
        if target is None or k <= 0:
            return []
        where, rest = self._where(filters)
        try:
            result = target.query(
                query_embeddings=[list(vector)],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise SinkError(f"Chroma search in {collection} failed ({type(e).__name__}).") from e
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[SearchHit] = []
        for item_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            raw = dict(metadata or {})
            saved = raw.pop(_META_JSON, "{}")
            restored = json.loads(str(saved))
            if not matches(restored, rest):
                continue
            hits.append(
                SearchHit(
                    id=str(item_id),
                    score=1.0 - float(distance),
                    text=str(text or ""),
                    metadata=restored,
                )
            )
        return hits
