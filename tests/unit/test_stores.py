from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from meldr import Meldr, MeldrConfig
from meldr._compat import UTC
from meldr.core.errors import ConfigError, SinkError, SpecConflictError
from meldr.core.mapping import FieldMapping, FieldStatus, Origin
from meldr.indexing.sinks.duckdb import DuckDBSink
from meldr.lineage.store import SQLLineageStore
from meldr.llm.cache import SQLResponseCache, SQLVectorCache
from meldr.mapping.store import SQLMappingStore
from meldr.observability import RunReport
from tests.helpers import make_config, make_spec


def _url(path: Path) -> str:
    return f"sqlite:///{path}"


def _spec() -> Any:
    return make_spec(
        [
            FieldMapping(
                target="a",
                sources=["a"],
                confidence=0.95,
                origin=Origin.HUMAN,
                status=FieldStatus.APPROVED,
            )
        ]
    )


def test_sql_mapping_store_round_trip_and_compare_and_set(tmp_path: Path) -> None:
    store = SQLMappingStore(_url(tmp_path / "state.sqlite"))
    first = store.save(_spec(), expected_latest=None)

    assert first.version == 1
    assert store.get(first.spec_id, 1).spec_id == first.spec_id
    assert store.latest("s", "E") == first
    assert store.history(first.spec_id) == [1]
    assert store.list("s") == [(first.spec_id, 1, first.status)]

    same = store.save(first, expected_latest=1)
    assert same.version == 1

    changed_field = first.fields[0].model_copy(update={"rationale": "changed"})
    changed = first.model_copy(update={"fields": [changed_field]})
    second = store.save(changed, expected_latest=1)
    assert second.version == 2
    with pytest.raises(SpecConflictError):
        store.save(changed, expected_latest=1)


def test_sql_lineage_and_caches_use_one_database(tmp_path: Path) -> None:
    url = _url(tmp_path / "state.sqlite")
    lineage = SQLLineageStore(url)
    report = RunReport(
        run_id="run-1",
        source_id="source",
        target="Entity",
        started_at=datetime.now(UTC),
    )
    lineage.record_run(report)
    assert lineage.get("run-1") == report
    assert lineage.runs()[0] == report

    responses = SQLResponseCache(url)
    responses.set("request", '{"value": 1}')
    assert responses.get("request") == '{"value": 1}'

    vectors = SQLVectorCache(url)
    vectors.set_many({"text": [0.25, 0.5]})
    assert vectors.get_many(["text"]) == {"text": [0.25, 0.5]}


def test_meldr_accepts_storage_overrides(tmp_path: Path) -> None:
    lineage = SQLLineageStore(_url(tmp_path / "state.sqlite"))
    responses = SQLResponseCache(_url(tmp_path / "responses.sqlite"))
    vectors = SQLVectorCache(_url(tmp_path / "vectors.sqlite"))
    mr = Meldr(
        make_config(tmp_path / "local"),
        lineage=lineage,
        llm_cache=responses,
        embedding_cache=vectors,
    )

    assert mr.lineage is lineage
    assert mr._llm_cache is responses
    assert mr.indexer.cache is vectors


def test_sql_store_config_requires_url() -> None:
    with pytest.raises(ConfigError, match=r"stores\.url"):
        MeldrConfig.from_dict({"stores": {"type": "sql"}})


def test_duckdb_lock_has_a_clear_error(tmp_path: Path, monkeypatch: Any) -> None:
    sink = DuckDBSink(tmp_path / "tables.duckdb")

    def locked(*args: Any, **kwargs: Any) -> None:
        raise duckdb.Error("database is locked")

    monkeypatch.setattr(sink, "_ensure_table", locked)
    with pytest.raises(SinkError, match="locked"):
        sink.write("Entity", pl.DataFrame({"id": ["1"]}), key=["id"])
