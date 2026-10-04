"""One behavioural contract for every embeddable vector sink: idempotent upserts, ranking,
scalar and any-of filters, k, and missing collections."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from meldr.indexing.sinks.base import IndexItem, VectorSink


def _memory(tmp: Path) -> VectorSink:
    from meldr.indexing.sinks.memory import MemoryVectorSink

    return MemoryVectorSink()


def _lancedb(tmp: Path) -> VectorSink:
    pytest.importorskip("lancedb")
    from meldr.indexing.sinks.lancedb import LanceDBSink

    return LanceDBSink(tmp / "vectors")


def _qdrant(tmp: Path) -> VectorSink:
    pytest.importorskip("qdrant_client")
    from meldr.indexing.sinks.qdrant import QdrantSink

    return QdrantSink()


def _chroma(tmp: Path) -> VectorSink:
    pytest.importorskip("chromadb")
    from meldr.indexing.sinks.chroma import ChromaSink

    return ChromaSink(tmp / "chroma")


SINKS: dict[str, Callable[[Path], VectorSink]] = {
    "chroma": _chroma,
    "memory": _memory,
    "lancedb": _lancedb,
    "qdrant": _qdrant,
}


@pytest.fixture(params=sorted(SINKS))
def sink(request: pytest.FixtureRequest, tmp_path: Path) -> VectorSink:
    return SINKS[request.param](tmp_path)


def item(i: str, vector: list[float], source: str) -> IndexItem:
    return IndexItem(id=i, vector=vector, text=f"text {i}", metadata={"source_id": source})


def test_upsert_search_and_filters(sink: VectorSink) -> None:
    items = [
        item("a", [1.0, 0.0], "x"),
        item("b", [0.9, 0.1], "y"),
        item("c", [0.0, 1.0], "z"),
    ]
    sink.upsert("records", items)
    sink.upsert("records", items)
    assert [h.id for h in sink.search("records", [1.0, 0.0], k=3)] == ["a", "b", "c"]
    assert [h.id for h in sink.search("records", [1.0, 0.0], k=1)] == ["a"]
    assert [h.id for h in sink.search("records", [1, 0], k=5, filters={"source_id": "y"})] == ["b"]
    any_of = sink.search("records", [1, 0], k=2, filters={"source_id": ["y", "z"]})
    assert [h.id for h in any_of] == ["b", "c"]
    assert sink.search("missing", [1.0, 0.0], k=3) == []


def test_ranking_is_cosine_for_unnormalized_vectors(sink: VectorSink) -> None:
    far_same_direction = item("a", [5.0, 0.0], "x")
    near_other_direction = item("b", [0.7, 0.7], "x")
    sink.upsert("records", [far_same_direction, near_other_direction])
    hits = sink.search("records", [1.0, 0.0], k=2)
    assert [h.id for h in hits] == ["a", "b"] and abs(hits[0].score - 1.0) < 1e-6
