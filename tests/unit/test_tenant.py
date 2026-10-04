from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from meldr import Meldr
from meldr._compat import UTC
from meldr.indexing.sinks.base import IndexItem
from meldr.indexing.sinks.duckdb import DuckDBSink
from meldr.indexing.sinks.memory import MemoryVectorSink
from meldr.lineage.store import SQLiteLineageStore, SQLLineageStore
from meldr.llm.cache import SQLiteResponseCache, SQLResponseCache, SQLVectorCache
from meldr.mapping.store import FileMappingStore, SQLMappingStore
from meldr.observability import RunReport
from tests.helpers import make_config, make_spec


def _url(path: Path) -> str:
    return f"sqlite:///{path}"


def _report(run_id: str) -> RunReport:
    return RunReport(
        run_id=run_id,
        source_id="source",
        target="Entity",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_file_mapping_store_scopes_tenants(tmp_path: Path) -> None:
    spec = make_spec([])
    first = FileMappingStore(tmp_path, tenant="first")
    second = FileMappingStore(tmp_path, tenant="second")

    assert first.save(spec).version == 1
    assert second.latest("s", "E") is None
    assert second.save(spec).version == 1
    assert first.list() == [(spec.spec_id, 1, spec.status)]
    assert second.list() == [(spec.spec_id, 1, spec.status)]


def test_sql_stores_scope_tenants(tmp_path: Path) -> None:
    url = _url(tmp_path / "state.sqlite")
    spec = make_spec([])
    first = SQLMappingStore(url, tenant="first")
    second = SQLMappingStore(url, tenant="second")
    first.save(spec)
    second.save(spec)

    assert first.latest("s", "E") is not None
    assert first.latest("s", "E").version == 1  # type: ignore[union-attr]
    assert second.latest("s", "E") is not None
    assert second.latest("s", "E").version == 1  # type: ignore[union-attr]

    first_lineage = SQLLineageStore(url, tenant="first")
    second_lineage = SQLLineageStore(url, tenant="second")
    report = _report("same-run")
    first_lineage.record_run(report)
    assert first_lineage.get(report.run_id) == report
    assert second_lineage.get(report.run_id) is None

    first_llm = SQLResponseCache(url, tenant="first")
    second_llm = SQLResponseCache(url, tenant="second")
    first_llm.set("same-key", "first")
    second_llm.set("same-key", "second")
    assert first_llm.get("same-key") == "first"
    assert second_llm.get("same-key") == "second"

    first_vectors = SQLVectorCache(url, tenant="first")
    second_vectors = SQLVectorCache(url, tenant="second")
    first_vectors.set_many({"same-key": [1.0]})
    second_vectors.set_many({"same-key": [2.0]})
    assert first_vectors.get_many(["same-key"]) == {"same-key": [1.0]}
    assert second_vectors.get_many(["same-key"]) == {"same-key": [2.0]}


def test_sqlite_lineage_and_caches_scope_tenants(tmp_path: Path) -> None:
    first_lineage = SQLiteLineageStore(tmp_path / "lineage.sqlite", tenant="first")
    second_lineage = SQLiteLineageStore(tmp_path / "lineage.sqlite", tenant="second")
    report = _report("same-run")
    first_lineage.record_run(report)
    assert first_lineage.get(report.run_id) == report
    assert second_lineage.get(report.run_id) is None

    first_cache = SQLiteResponseCache(tmp_path / "cache.sqlite", tenant="first")
    second_cache = SQLiteResponseCache(tmp_path / "cache.sqlite", tenant="second")
    first_cache.set("same-key", "first")
    second_cache.set("same-key", "second")
    assert first_cache.get("same-key") == "first"
    assert second_cache.get("same-key") == "second"


def test_meldr_tenant_namespaces_shared_sinks(tmp_path: Path) -> None:
    table_sink = DuckDBSink(tmp_path / "tables.duckdb")
    vector_sink = MemoryVectorSink()
    first = Meldr(
        make_config(tmp_path / "first-state"),
        tenant="first",
        table_sink=table_sink,
        vector_sink=vector_sink,
    )
    second = Meldr(
        make_config(tmp_path / "second-state"),
        tenant="second",
        table_sink=table_sink,
        vector_sink=vector_sink,
    )
    frame = pl.DataFrame({"id": ["1"], "value": [1]})

    first.table_sink.write("Entity", frame, key=["id"])
    second.table_sink.write("Entity", frame, key=["id"])
    assert first.table_sink.read("Entity").height == 1
    assert second.table_sink.read("Entity").height == 1
    assert table_sink.read("Entity").height == 0

    item = IndexItem(
        id="record",
        vector=[1.0, 0.0],
        text="record",
        metadata={"entity": "Entity"},
    )
    first.vector_sink.upsert("records__Entity", [item])
    assert first.vector_sink.search("records__Entity", [1.0, 0.0], k=1)
    assert not second.vector_sink.search("records__Entity", [1.0, 0.0], k=1)

    first.close()
    second.close()
