"""Postgres + pgvector sink (optional extra `pgvector`)."""

from __future__ import annotations

import json
from collections.abc import Sequence
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
)


def _vec(v: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


class PgVectorSink:
    """One table for each collection: id text pk, embedding vector(dim), text, metadata jsonb."""

    def __init__(self, dsn: str | None = None, *, connection: Any = None) -> None:
        if connection is None:
            if not dsn:
                raise SinkError("The pgvector sink has no DSN. Set sinks.vectors.url.")
            try:
                import psycopg
            except ImportError as e:
                raise MissingExtraError("pgvector", "The pgvector sink") from e
            connection = psycopg.connect(dsn, autocommit=True)
        self._conn = connection
        self._ready: set[str] = set()

    def _ensure(self, name: str, dim: int) -> None:
        if name in self._ready:
            return
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f'CREATE TABLE IF NOT EXISTS "{name}" (id text PRIMARY KEY, '
                f"embedding vector({int(dim)}), text text, metadata jsonb)"
            )
        self._ready.add(name)

    def upsert(self, collection: str, items: Sequence[IndexItem]) -> WriteResult:
        name = safe_identifier(collection)
        if items:
            self._ensure(name, len(items[0].vector))
            with self._conn.cursor() as cur:
                cur.executemany(
                    f'INSERT INTO "{name}" (id, embedding, text, metadata) '  # noqa: S608
                    "VALUES (%s, %s::vector, %s, %s::jsonb) ON CONFLICT (id) DO UPDATE SET "
                    "embedding = EXCLUDED.embedding, text = EXCLUDED.text, "
                    "metadata = EXCLUDED.metadata",
                    [(i.id, _vec(i.vector), i.text, dumps_meta(i.metadata)) for i in items],
                )
        return WriteResult(rows_written=len(items), target=f"pgvector:{name}")

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        name = safe_identifier(collection)
        scalar = scalar_filters(filters)
        limit = k if len(scalar) == len(filters or {}) else k * OVERFETCH
        with self._conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f'"{name}"',))  # quoted: names keep their case
            row = cur.fetchone()
            if not row or row[0] is None:
                return []
            cur.execute(
                f'SELECT id, text, metadata, 1 - (embedding <=> %s::vector) FROM "{name}" '  # noqa: S608
                "WHERE metadata @> %s::jsonb ORDER BY embedding <=> %s::vector LIMIT %s",
                (_vec(vector), dumps_meta(scalar), _vec(vector), limit),
            )
            rows = cur.fetchall()
        hits = [
            SearchHit(
                id=r[0],
                text=r[1],
                score=float(r[3]),
                metadata=r[2] if isinstance(r[2], dict) else json.loads(r[2]),
            )
            for r in rows
        ]
        return [h for h in hits if matches(h.metadata, filters)][:k]
