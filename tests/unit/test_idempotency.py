"""Re-running anything must converge to the same state."""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from meldr import Meldr, RawBatch
from meldr.api import apply_mapping, propose_mapping
from meldr.config import PaginationConfig
from meldr.connectors.files import FileConnector
from meldr.connectors.rest import RestConnector
from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import DataType
from meldr.eval.tables import entity_from_table
from meldr.execution.record_id import namespace_for
from meldr.indexing.sinks.parquet import ParquetSink
from meldr.mapping import review
from meldr.mapping.store import FileMappingStore


def _vendors(name_of_v2: str | None) -> RawBatch:
    return RawBatch.from_polars(
        pl.DataFrame({"VNDR_ID": ["V1", "V2"], "VNDR_NM": ["Acme Corp", name_of_v2]}), "erp"
    )


def test_quarantine_converges_across_reruns(mr: Meldr) -> None:
    for _ in range(2):
        res = mr.meld(_vendors(None), "Organization")
    assert res.report.rows_quarantined == 1
    assert mr.table_sink.read("Organization__quarantine").height == 1  # upserted, not appended

    mr.meld(_vendors("Beta LLC"), "Organization")  # the record is fixed at the source
    assert mr.table_sink.read("Organization__quarantine").height == 0
    assert sorted(mr.table_sink.read("Organization")["legal_name"]) == ["Acme Corp", "Beta LLC"]

    mr.meld(_vendors(None), "Organization")  # broken again: the last good row is kept
    assert sorted(mr.table_sink.read("Organization")["legal_name"]) == ["Acme Corp", "Beta LLC"]
    assert mr.table_sink.read("Organization__quarantine").height == 1


def test_csv_is_read_id_safe_and_streamed(tmp_path: Path) -> None:
    path = tmp_path / "sites.csv"
    path.write_text("zip,n\n02134,1\n00501,2\n10001,3\n")
    batches = list(FileConnector("f", path, batch_size=2).pull())
    assert [b.table.height for b in batches] == [2, 1]  # type: ignore[union-attr]
    assert batches[0].table["zip"].to_list() == ["02134", "00501"]  # type: ignore[index]


@respx.mock
def test_rest_pull_is_lazy() -> None:
    route = respx.get("https://api.example.com/items")
    route.side_effect = [httpx.Response(200, json=[{"id": 1}]), httpx.Response(200, json=[])]
    rest = RestConnector(
        "api", "https://api.example.com", endpoint="items", pagination=PaginationConfig(type="page")
    )
    pages = rest.pull()
    assert next(pages).records[0].data == {"id": 1}
    assert route.call_count == 1  # the second page is only fetched when asked for
    assert list(pages) == []


def test_parquet_read_returns_latest_row_per_key(tmp_path: Path) -> None:
    sink = ParquetSink(tmp_path)
    key = ["_meldr_source_id", "_meldr_record_id"]

    def rows(value: int, at: str) -> pl.DataFrame:
        return pl.DataFrame(
            {"_meldr_source_id": ["s"], "_meldr_record_id": ["1"], "v": [value]}
        ).with_columns(pl.lit(at).str.to_datetime(time_zone="UTC").alias("_meldr_ingested_at"))

    sink.write("Org", rows(1, "2026-01-01"), key=key)
    sink.write("Org", rows(2, "2026-01-02"), key=key)
    sink.write("Org", rows(2, "2026-01-02"), key=key)
    assert sink.read("Org")["v"].to_list() == [2]
    sink.delete("Org", rows(2, "2026-01-02").select(key))
    assert sink.read("Org").height == 0

    sequenced = ParquetSink(tmp_path / "seq")
    newer = rows(2, "2026-01-01").with_columns(pl.lit(20).alias("_meldr_sequence"))
    older_written_later = rows(1, "2026-01-05").with_columns(pl.lit(10).alias("_meldr_sequence"))
    sequenced.write("Org", newer, key=key)
    sequenced.write("Org", older_written_later, key=key)
    assert sequenced.read("Org")["v"].to_list() == [2]


def test_parquet_sink_reads_and_writes_memory_uri() -> None:
    pytest.importorskip("fsspec")
    sink = ParquetSink(f"memory://parquet-{uuid.uuid4().hex}")
    rows = pl.DataFrame({"_meldr_source_id": ["s"], "_meldr_record_id": ["1"], "v": [1]})
    sink.write("Org", rows, key=["_meldr_source_id", "_meldr_record_id"])
    assert sink.read("Org")["v"].to_list() == [1]


def test_functional_api_is_deterministic(erp_df: pl.DataFrame, org: CanonicalEntity) -> None:
    spec = propose_mapping(erp_df, org, source_id="erp")
    approved = review.approve(spec, user="t", above=0.0)
    first = apply_mapping(erp_df, approved, org, run_id="r")
    second = apply_mapping(erp_df, approved, org, run_id="r")
    lineage_time = "_meldr_ingested_at"
    assert first.data.drop(lineage_time).equals(second.data.drop(lineage_time))
    assert approved.generator.transforms_version == "1"
    stale = approved.model_copy(
        update={"generator": approved.generator.model_copy(update={"transforms_version": "0"})}
    )
    assert "transforms_version_changed:0->1" in apply_mapping(erp_df, stale, org).warnings


def test_store_never_overwrites_a_version(
    tmp_path: Path, erp_df: pl.DataFrame, org: CanonicalEntity
) -> None:
    store = FileMappingStore(tmp_path)
    spec = propose_mapping(erp_df, org, source_id="erp")
    assert store.save(spec).version == 1
    taken = tmp_path / "specs" / "erp" / "Organization" / "v0002.yaml"
    taken.write_text("written by another process\n")
    assert store.save(spec).version == 3
    assert taken.read_text() == "written by another process\n"
    assert not list(taken.parent.glob(".tmp-*"))


def test_replaying_older_data_never_regresses(org: CanonicalEntity, mr: Meldr) -> None:
    event = org.model_copy(
        update={
            "name": "Vendor",
            "fields": [*org.fields, CanonicalField(name="updated_at", dtype=DataType.DATE)],
            "sequence_field": "updated_at",
        }
    )
    mr.registry.register(event)

    def batch(name: str | None, updated: str) -> RawBatch:
        df = pl.DataFrame({"VNDR_ID": ["V1"], "VNDR_NM": [name], "updated_at": [updated]})
        return RawBatch.from_polars(df, "erp")

    mr.meld(batch("Acme v2", "2026-02-01"), "Vendor")
    mr.meld(batch("Acme v1", "2026-01-01"), "Vendor")  # late replay of an older extract
    assert mr.table_sink.read("Vendor")["legal_name"].to_list() == ["Acme v2"]

    mr.meld(batch(None, "2026-03-01"), "Vendor")  # newer but invalid
    mr.meld(batch("Acme v2", "2026-02-01"), "Vendor")  # older valid replay
    assert mr.table_sink.read("Vendor__quarantine").height == 1  # the newer failure stands
    mr.meld(batch("Acme v4", "2026-04-01"), "Vendor")
    assert mr.table_sink.read("Vendor__quarantine").height == 0
    assert mr.table_sink.read("Vendor")["legal_name"].to_list() == ["Acme v4"]


def test_keyless_ids_hash_the_source_row() -> None:
    entity = entity_from_table(pl.DataFrame({"a": ["x"], "b": [1]}), "E")
    spec = review.approve(
        propose_mapping(pl.DataFrame({"a": ["x"], "b": [1]}), entity, source_id="s"),
        user="t",
        above=0.0,
    )
    out = apply_mapping(pl.DataFrame({"a": ["x", "y"], "b": [1, None]}), spec, entity)
    ns = namespace_for("s", "E")
    expected = [
        str(uuid.uuid5(ns, 'row\x01{"a":"x","b":"1"}')),
        str(uuid.uuid5(ns, 'row\x01{"a":"y","b":null}')),
    ]
    assert out.data["_meldr_record_id"].to_list() == expected
