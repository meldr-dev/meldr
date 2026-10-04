"""Embed harmonized records and content segments into a vector sink."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import polars as pl
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from meldr._retry import RetryAfterWait
from meldr.core.canonical import CanonicalEntity
from meldr.core.errors import LLMTransientError
from meldr.core.segments import ContentSegment
from meldr.indexing.render import render
from meldr.indexing.sinks.base import (
    RECORD_KEY,
    IndexItem,
    MetaValue,
    SearchHit,
    VectorSink,
    keep_latest,
)
from meldr.llm.base import Embedder
from meldr.llm.cache import VectorCache
from meldr.observability import otel_span

SEGMENTS = "segments"
LINEAGE_META = (
    "_meldr_source_id",
    "_meldr_record_id",
    "_meldr_spec_id",
    "_meldr_spec_version",
    "_meldr_run_id",
)
SLICE_ROWS = 2048  # rows to render, embed, and upsert at a time (limits memory for each batch)


def records_collection(entity: str) -> str:
    return f"records__{entity}"


def _scalar(v: Any) -> MetaValue:
    if v is None or isinstance(v, bool | int | float | str):
        return v
    if isinstance(v, datetime | date):
        return v.isoformat()
    return json.dumps(v, default=str)


class Indexer:
    """Batched, retried embedding with a text-hash cache."""

    def __init__(
        self,
        embedder: Embedder,
        sink: VectorSink,
        *,
        cache: VectorCache | None = None,
        batch_size: int = 256,
        max_concurrency: int = 4,
        max_retries: int = 4,
    ) -> None:
        self.embedder = embedder
        self.sink = sink
        self.cache = cache
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries

    def _key(self, text: str) -> str:
        model = getattr(self.embedder, "model", self.embedder.name)
        dim = getattr(self.embedder, "dim", "")
        return hashlib.sha256(f"{self.embedder.name}|{model}|{dim}|{text}".encode()).hexdigest()

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(LLMTransientError),
            stop=stop_after_attempt(self.max_retries + 1),
            wait=RetryAfterWait(base=1.0),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self.embedder.embed(texts)
        raise AssertionError("unreachable")  # pragma: no cover

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed with caching, and send distinct misses in batches, with a concurrency limit."""
        keys = [self._key(t) for t in texts]
        known = await asyncio.to_thread(self.cache.get_many, keys) if self.cache else {}
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in known}
        pending = list(missing.items())
        sem = asyncio.Semaphore(self.max_concurrency)

        async def run(chunk: list[tuple[str, str]]) -> dict[str, list[float]]:
            async with sem:
                vectors = await self._embed_batch([t for _, t in chunk])
            return {k: v for (k, _), v in zip(chunk, vectors, strict=True)}

        chunks = [pending[i : i + self.batch_size] for i in range(0, len(pending), self.batch_size)]
        fresh: dict[str, list[float]] = {}
        for part in await asyncio.gather(*(run(c) for c in chunks)):
            fresh.update(part)
        if self.cache and fresh:
            await asyncio.to_thread(self.cache.set_many, fresh)
        return [known[k] if k in known else fresh[k] for k in keys]

    async def search(
        self,
        query: str,
        *,
        collections: Sequence[str],
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[SearchHit]:
        """Embed `query`, search each collection, and merge by score."""
        [vector] = await self.embed([query])
        groups = await asyncio.gather(
            *(
                asyncio.to_thread(self.sink.search, c, vector, k=k, filters=filters)
                for c in collections
            )
        )
        hits = [h for group in groups for h in group]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

    async def index_records(self, entity: CanonicalEntity, df: pl.DataFrame) -> int:
        """Index one item for each row.

        The item has the rendered text, with scalar canonical fields + lineage as metadata.
        """
        names = entity.field_names()
        if all(c in df.columns for c in RECORD_KEY):
            df = keep_latest(df, RECORD_KEY)  # one vector for each record: the batch's newest row
        for part in df.iter_slices(SLICE_ROWS):
            rows = part.to_dicts()
            texts = [render(entity.embedding_template, r, names) for r in rows]
            vectors = await self.embed(texts)
            items = []
            for r, text, vec in zip(rows, texts, vectors, strict=True):
                meta: dict[str, MetaValue] = {n: _scalar(r.get(n)) for n in names}
                for col in LINEAGE_META:
                    if col in r:
                        meta[col.removeprefix("_meldr_")] = _scalar(r[col])
                meta["entity"] = entity.name
                meta["kind"] = "record"
                record_id = f"{entity.name}:{r.get('_meldr_source_id')}:{r.get('_meldr_record_id')}"
                items.append(IndexItem(id=record_id, vector=vec, text=text, metadata=meta))
            with otel_span(
                "meldr.sink.write",
                sink=type(self.sink).__name__,
                kind="vector",
                collection=records_collection(entity.name),
                rows=len(items),
            ):
                await asyncio.to_thread(self.sink.upsert, records_collection(entity.name), items)
        return df.height

    async def index_segments(
        self,
        segments: Sequence[ContentSegment],
        links: Mapping[str, Sequence[str]] | None = None,
    ) -> int:
        """Segments with locators, where `links` maps segment_id -> extracted record ids."""
        if not segments:
            return 0
        texts = [f"{s.title}\n{s.text}" if s.title else s.text for s in segments]
        vectors = await self.embed(texts)
        items = []
        for s, text, vec in zip(segments, texts, vectors, strict=True):
            meta: dict[str, MetaValue] = {
                "source_id": s.source_id,
                "record_id": s.record_id,
                "segment_id": s.segment_id,
                "modality": str(s.modality),
                "kind": "segment",
                "title": s.title,
            }
            for k, v in s.locator.model_dump(exclude_none=True).items():
                meta[k] = v
            ids = list((links or {}).get(s.segment_id, []))
            meta["extracted_record_ids"] = ",".join(ids) if ids else None
            items.append(IndexItem(id=s.segment_id, vector=vec, text=text, metadata=meta))
        with otel_span(
            "meldr.sink.write",
            sink=type(self.sink).__name__,
            kind="vector",
            collection=SEGMENTS,
            rows=len(items),
        ):
            await asyncio.to_thread(self.sink.upsert, SEGMENTS, items)
        return len(items)
