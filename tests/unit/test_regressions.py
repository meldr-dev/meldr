"""One test per correctness bug found in review, so each stays fixed."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
import respx

from meldr import CanonicalEntity, CanonicalField, DataType, Meldr, RawBatch
from meldr._async import run_sync
from meldr.api import apply_mapping, propose_mapping
from meldr.config import PaginationConfig
from meldr.connectors.files import FileConnector
from meldr.connectors.rest import RestConnector
from meldr.core.errors import SinkError
from meldr.core.mapping import TransformStep
from meldr.execution.validator import validate
from meldr.indexing.sinks.duckdb import DuckDBSink
from meldr.mapping import review
from meldr.transforms.ops import clean_number
from meldr.transforms.registry import default_registry
from tests.helpers import make_config


def _entity(*fields: CanonicalField, key: str = "id", **extra: Any) -> CanonicalEntity:
    return CanonicalEntity(
        name="E",
        version="1.0.0",
        fields=[CanonicalField(name=key, dtype=DataType.STRING), *fields],
        primary_key=[key],
        **extra,
    )


def _apply(
    df: pl.DataFrame, entity: CanonicalEntity, target: str, steps: list[dict[str, Any]]
) -> Any:
    spec = review.approve(propose_mapping(df, entity, source_id="s"), user="t", above=0.0)
    fields = [
        f.model_copy(update={"steps": [TransformStep(**s) for s in steps]})
        if f.target == target
        else f
        for f in spec.fields
    ]
    return apply_mapping(df, spec.model_copy(update={"fields": fields}), entity)


def test_failed_parse_after_nullify_is_lost() -> None:
    df = pl.DataFrame({"id": ["1", "2", "3"], "amount": ["10", "N/A", "12O0"]})
    entity = _entity(CanonicalField(name="amount", dtype=DataType.FLOAT))
    out = _apply(
        df,
        entity,
        "amount",
        [{"op": "nullify", "args": {"values": ["N/A"]}}, {"op": "parse_number"}],
    )
    assert out.error_counts == {"value_lost:amount": 1}
    assert out.data["amount"].to_list() == [10.0, None]


def test_misplaced_thousands_separator_is_not_a_number() -> None:
    values = pl.DataFrame({"v": ["1,5", "1,234.5", "12,345,678", "3000"]})
    assert values.select(clean_number(pl.col("v")))["v"].to_list() == [
        None,
        1234.5,
        12345678.0,
        3000.0,
    ]


def test_same_file_names_in_different_folders_stay_distinct(tmp_path: Path) -> None:
    for folder, text in (("a", "Acme"), ("b", "Beta")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "report.txt").write_text(text)
    [batch] = FileConnector("docs", tmp_path, records_mode=True).pull()
    assert sorted(r.record_id for r in batch.records) == ["a/report.txt", "b/report.txt"]
    with pytest.raises(Exception, match="same name"):
        RawBatch.from_files([tmp_path / "a/report.txt", tmp_path / "b/report.txt"], "docs")


@respx.mock
def test_offset_paging_survives_a_capped_limit() -> None:
    data = [{"id": i} for i in range(7)]

    def page(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        return httpx.Response(200, json=data[offset : offset + 2])  # server caps limit at 2

    respx.get("https://api.example.com/items").mock(side_effect=page)
    rest = RestConnector(
        "api",
        "https://api.example.com",
        endpoint="items",
        pagination=PaginationConfig(type="offset", page_size=5),
    )
    assert [r.data["id"] for b in rest.pull() for r in b.records] == list(range(7))  # type: ignore[index]
    with pytest.raises(ValueError, match="next_path"):
        PaginationConfig(type="cursor")


def test_json_arrays_and_array_vocabularies(mr: Meldr) -> None:
    rows = [
        {"org_id": "V1", "legal_name": "Acme", "tags": ["a", "b"]},
        {"org_id": None, "legal_name": "B"},
    ]
    result = mr.meld(RawBatch.from_dicts(rows, "api"), "Organization")
    assert result.data.height == 1 and result.quarantine.height == 1
    colors = CanonicalField(name="colors", dtype=DataType.ARRAY, vocabulary=["red", "blue"])
    checked = validate(pl.DataFrame({"colors": [["red"], ["green"]]}), _entity(colors))
    assert checked.error_counts == {"value_not_in_vocabulary:colors": 1}


def test_quarantine_table_older_than_sequence_field(tmp_path: Path) -> None:
    sink = DuckDBSink(tmp_path / "h.duckdb")
    key = ["_meldr_source_id", "_meldr_record_id"]
    sink.write("Q", pl.DataFrame({"_meldr_source_id": ["s"], "_meldr_record_id": ["1"]}), key=key)
    sink.delete(
        "Q",
        pl.DataFrame(
            {"_meldr_source_id": ["s"], "_meldr_record_id": ["1"], "_meldr_sequence": [1]}
        ),
    )
    assert sink.read("Q").height == 0
    sink.write(
        "T", pl.DataFrame({"_meldr_source_id": ["s"], "_meldr_record_id": ["1"], "v": [1]}), key=key
    )
    with pytest.raises(SinkError) as err:
        sink.write(
            "T",
            pl.DataFrame(
                {"_meldr_source_id": ["s"], "_meldr_record_id": ["2"], "v": ["jane@corp.com"]}
            ),
            key=key,
        )
    assert "jane" not in str(err.value)


def test_extraction_edge_cases(tmp_path: Path, mr: Meldr) -> None:
    (tmp_path / "empty.txt").write_text("")
    res = mr.extract(RawBatch.from_files([tmp_path / "empty.txt"], "docs"), "Organization")
    assert res.data.height == 0 and "no_segments:empty.txt" in res.report.warnings
    for name in ("jan.txt", "feb.txt"):
        (tmp_path / name).write_text("Legal name: Acme Corp\nOrg ID: A-1\n")
    for name in ("jan.txt", "feb.txt"):
        out = mr.extract(RawBatch.from_files([tmp_path / name], "docs"), "Organization")
        assert [r.record_id for r in out.records] == [name]
    explained = mr.explain("Organization", "jan.txt", "legal_name")
    assert explained.origin == "extraction" and explained.value == "Acme Corp"


def test_empty_first_batch_does_not_fix_the_spec(mr: Meldr, erp_df: pl.DataFrame) -> None:
    mr.meld(RawBatch.from_polars(erp_df.head(0), "erp"), "Organization")
    assert mr.store.latest("erp", "Organization") is None
    result = mr.meld(RawBatch.from_polars(erp_df, "erp"), "Organization")
    steps = {f.target: [s.op for s in f.steps] for f in result.spec.fields}
    assert steps["founded_on"] == ["parse_date"]


def test_ambiguous_local_time_loses_one_row() -> None:
    df = pl.DataFrame({"id": ["1", "2"], "at": ["2024-11-03 01:30:00", "2024-11-04 09:00:00"]})
    entity = _entity(CanonicalField(name="at", dtype=DataType.DATETIME))
    args = {"formats": ["%Y-%m-%d %H:%M:%S"], "assume_tz": "America/New_York"}
    out = _apply(df, entity, "at", [{"op": "parse_datetime", "args": args}])
    assert out.data["at"].to_list() == [datetime.fromisoformat("2024-11-04T14:00:00+00:00")]
    assert out.error_counts == {"value_lost:at": 1}


def test_valid_version_in_the_same_batch_clears_quarantine(tmp_path: Path) -> None:
    doc = CanonicalEntity(
        name="Doc",
        version="1.0.0",
        primary_key=["doc_id"],
        sequence_field="rev",
        fields=[
            CanonicalField(name="doc_id", dtype=DataType.STRING),
            CanonicalField(name="rev", dtype=DataType.INTEGER),
            CanonicalField(name="title", dtype=DataType.STRING, required=True),
        ],
    )
    mr = Meldr(make_config(tmp_path / "state"))
    mr.registry.register(doc)
    df = pl.DataFrame({"doc_id": ["D1", "D1"], "rev": [3, 5], "title": [None, "fixed"]})
    mr.meld(RawBatch.from_polars(df, "s"), "Doc")
    assert mr.table_sink.read("Doc")["title"].to_list() == ["fixed"]
    assert mr.table_sink.read("Doc__quarantine").height == 0


def test_embedder_sees_no_raw_samples_under_a_masked_policy(org: CanonicalEntity) -> None:
    from meldr import DataPolicy, FakeProvider, LLMGateway
    from meldr.llm.embedders.fake import FakeEmbedder
    from meldr.mapping.engine import MappingEngine
    from meldr.pipeline.oracle import heuristic_oracle
    from meldr.profiling.profiler import profile_table

    seen: list[str] = []

    class Spy(FakeEmbedder):
        async def embed(self, texts: Any) -> list[list[float]]:
            seen.extend(texts)
            return await super().embed(texts)

    vendor = FakeProvider(mode="oracle", oracle=heuristic_oracle, is_local=False)
    gateway = LLMGateway(vendor, policy=DataPolicy.default_for(is_local=False))
    df = pl.DataFrame({"contact": ["jane.doe@corp.com", "bob@corp.com"]})
    engine = MappingEngine(default_registry(), gateway=gateway, embedder=Spy())
    run_sync(engine.propose(profile_table(df, "s"), org, df))
    assert seen and not any("corp.com" in t for t in seen)
