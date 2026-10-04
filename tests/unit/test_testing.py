"""Tests for the public extension checks."""

from __future__ import annotations

from pathlib import Path

from meldr import check_table_sink, check_vector_sink
from meldr.indexing.sinks.duckdb import DuckDBSink
from meldr.indexing.sinks.memory import MemoryVectorSink


def test_public_table_sink_check() -> None:
    check_table_sink(lambda root: DuckDBSink(root / "data.duckdb"))


def test_public_vector_sink_check() -> None:
    check_vector_sink(lambda root: MemoryVectorSink())


def test_check_factories_accept_a_root(tmp_path: Path) -> None:
    check_table_sink(lambda root: DuckDBSink(root / "data.duckdb"), root=tmp_path / "table")
    check_vector_sink(lambda root: MemoryVectorSink(), root=tmp_path / "vector")
