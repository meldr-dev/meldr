"""Phase "Done when" checks, end to end on synthetic data (no network)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from meldr import (
    DriftBlockedError,
    LLMGateway,
    Meldr,
    PolicyViolation,
    RawBatch,
)
from meldr.connectors.rest import RestConnector
from meldr.connectors.sql import SQLConnector
from meldr.llm.policy import DataClass, DataPolicy
from meldr.llm.providers.fake import FakeProvider
from meldr.mapping import review
from meldr.pipeline.oracle import heuristic_oracle
from tests.helpers import make_config

pytestmark = pytest.mark.integration


def _sources(tmp: Path) -> list[RawBatch]:
    csv = tmp / "crm.csv"
    pl.DataFrame(
        {
            "customer_number": ["C1", "C2"],
            "company_name": ["Acme Inc", "Beta LLC"],
            "type": ["customer", "partner"],
            "annual_revenue": [1200.5, 99.0],
            "founded": ["2001-03-14", "1999-12-25"],
            "country_code": ["US", "CA"],
        }
    ).write_csv(csv)
    pq = tmp / "ap.parquet"
    pl.DataFrame(
        {
            "vendorId": ["V9"],
            "vendorName": ["Gamma Co"],
            "revenueCents": [1000],
            "estDate": ["14.03.2001"],
            "ctry": ["GB"],
        }
    ).write_parquet(pq)
    db = tmp / "erp.db"
    db.unlink(missing_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE vendors (VNDR_ID TEXT, VNDR_NM TEXT, VNDR_TYP_CD TEXT, "
            "REV_CENTS INTEGER, EST_DT TEXT)"
        )
        conn.executemany(
            "INSERT INTO vendors VALUES (?,?,?,?,?)",
            [("E1", "Delta", "S", 500, "03/14/2001"), ("E2", "Echo", "V", 700, "12/25/1999")],
        )
    return [
        RawBatch.from_polars(pl.read_csv(csv), "crm"),
        RawBatch.from_polars(pl.read_parquet(pq), "ap"),
        next(SQLConnector("erp", f"sqlite:///{db}").pull(table="vendors")),
    ]


def test_phase2_tabular_demo(tmp_path: Path, mr: Meldr) -> None:
    first = [mr.meld(b, "Organization") for b in _sources(tmp_path)]
    assert all(r.report.llm_calls > 0 for r in first)
    table = mr.table_sink.read("Organization")
    assert set(table["_meldr_source_id"].unique()) == {"crm", "ap", "erp"}
    erp_rows = table.filter(pl.col("_meldr_source_id") == "erp").sort("org_id")
    assert erp_rows["annual_revenue"].to_list() == [5.0, 7.0]

    second = [mr.meld(b, "Organization") for b in _sources(tmp_path)]
    assert all(r.report.llm_calls == 0 and r.report.spec_cache_hit for r in second)

    erp = second[2]
    pending = [f.target for f in erp.pending_review]
    assert "org_type" in pending
    approved = mr.store.save(review.approve(erp.spec, user="test", targets=pending))
    rerun = mr.meld(_sources(tmp_path)[2], "Organization")
    assert rerun.spec.version == approved.version
    assert rerun.data["org_type"].to_list() == ["subcontractor", "vendor"]


@respx.mock
def test_phase3_api_sql_drift(tmp_path: Path) -> None:
    fake = FakeProvider(mode="oracle", oracle=heuristic_oracle)
    mr = Meldr(make_config(tmp_path / "state"), llm=LLMGateway(fake))
    respx.get("https://api.example.com/v2/vendors").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "A1",
                        "profile": {"legal_name": "Api Co", "contact": {"email": "x@y.io"}},
                        "country": "US",
                    }
                ]
            },
        )
    )
    api = next(
        RestConnector(
            "vendors_api", "https://api.example.com/v2", endpoint="vendors", records_path="data"
        ).pull()
    )
    res = mr.meld(api, "Organization")
    assert res.data["legal_name"].to_list() == ["Api Co"]

    base = pl.DataFrame({"org_id": ["1"], "legal_name": ["Acme"], "city": ["Denver"]})
    mr.meld(RawBatch.from_polars(base, "sql"), "Organization")
    unmapped_before = mr.store.latest("sql", "Organization").unmapped_targets  # type: ignore[union-attr]
    calls = len(fake.calls)
    grown = base.with_columns(pl.lit("US").alias("country"))
    mr.meld(RawBatch.from_polars(grown, "sql"), "Organization")
    asked = [
        t["name"] for c in fake.calls[calls:] for t in (c.payload() or {}).get("target_fields", [])
    ]
    assert asked and set(asked) <= set(unmapped_before)

    with pytest.raises(DriftBlockedError, match="legal_name"):
        mr.meld(RawBatch.from_polars(grown.drop("legal_name"), "sql"), "Organization")


DOC = """# Project Riverside Clinic

Project ID: P-100
Name: Riverside Clinic
Status: active
Client: Acme Health Inc
City: Denver
Budget: $1,250,000
Start date: 2025-03-01

The general contractor mentioned in the kickoff meeting is Beta Builders.
"""


def test_phase4_documents_search_and_policy(tmp_path: Path, mr: Meldr) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "riverside.md").write_text(DOC)
    res = mr.extract(RawBatch.from_files([docs / "riverside.md"], "project_docs"), "Project")
    assert res.data["project_id"].to_list() == ["P-100"]
    assert res.data["budget"].to_list() == [1_250_000.0]
    rec = res.records[0]
    assert rec.evidence and all(e.verified for e in rec.evidence.values())
    hits = mr.search("Riverside clinic budget")
    assert any(h.metadata.get("kind") == "segment" for h in hits)

    vendor = FakeProvider(mode="oracle", oracle=heuristic_oracle, is_local=False)
    hz2 = Meldr(
        make_config(tmp_path / "state2"),
        llm=LLMGateway(vendor, policy=DataPolicy.default_for(is_local=False)),
    )
    with pytest.raises(PolicyViolation, match="content"):
        hz2.extract(RawBatch.from_files([docs / "riverside.md"], "project_docs"), "Project")
    assert vendor.calls == []
    assert DataClass.CONTENT not in hz2.llm.policy.allowed  # type: ignore[union-attr]


def test_explain_resolve_discover(tmp_path: Path, mr: Meldr) -> None:
    batches = _sources(tmp_path)
    for b in batches:
        mr.meld(b, "Organization")
    exp = mr.explain("Organization", "C1", "legal_name")
    assert exp.source_paths == ["company_name"] and exp.run is not None
    res = mr.resolve("Organization", fields=["legal_name"], block_on=["country"])
    assert "_meldr_entity_id" in res.data.columns
    draft = mr.discover_schema(batches[:2], "Vendor")
    assert draft.name == "Vendor" and draft.fields
    assert json.loads(mr.lineage.runs(limit=1)[0].model_dump_json())["source_id"]
