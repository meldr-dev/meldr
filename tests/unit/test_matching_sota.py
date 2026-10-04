"""Techniques from the schema-matching literature (see docs/DECISIONS.md for sources)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from meldr._async import run_sync
from meldr.api import apply_mapping, propose_mapping
from meldr.config import MappingConfig
from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import DataType
from meldr.core.errors import ConfigError
from meldr.eval.runner import builtin_factories, run_table_eval
from meldr.eval.tables import entity_from_table
from meldr.llm.gateway import LLMGateway
from meldr.llm.providers.fake import FakeProvider
from meldr.mapping import heuristic, review
from meldr.mapping.adjudicator import vote
from meldr.mapping.candidates import generate
from meldr.mapping.engine import MappingEngine
from meldr.mapping.memory import ApprovedMappingMatcher
from meldr.mapping.store import FileMappingStore
from meldr.mapping.values import value_map_request
from meldr.profiling.patterns import detect
from meldr.profiling.profiler import profile_table
from meldr.transforms.registry import default_registry
from tests.helpers import decision

OPS = default_registry()

SOURCE = pl.DataFrame(
    {
        "total_changes": ["916", "2086", "2252", "14"],
        "changes": ["6752000.5", "-10555000", "73735148.87", "120000"],
        "report_date": ["2023-01-01", "2022-09-01", "2022-04-01", "2021-01-01"],
    }
)
TARGET = pl.DataFrame(
    {
        "total_schedule_changes": ["1477", "1215", "60", "300"],
        "total_budget_changes": ["-19049921.84", "25273000", "0.76", "4338327.57"],
        "date_reported_as_of": ["2023-01-01T00:00:00.000", "2017-09-01T00:00:00.000", "", ""],
    }
)


def _cands() -> dict[str, list[Any]]:
    return run_sync(generate(profile_table(SOURCE, "s"), entity_from_table(TARGET, "T")))


def test_instance_signal_resolves_name_traps() -> None:
    cands = _cands()
    assert cands["total_budget_changes"][0].path == "changes"  # millions, not counts
    assert cands["total_schedule_changes"][0].path == "total_changes"


def test_dates_are_not_phone_numbers() -> None:
    assert "phone" not in detect(SOURCE["report_date"])


def test_ambiguous_near_ties_are_left_to_review(org: CanonicalEntity) -> None:
    df = pl.DataFrame({"org_name_a": ["x", "y"], "org_name_b": ["x", "y"]})
    cands = run_sync(generate(profile_table(df, "s"), org))
    proposals = heuristic.propose(profile_table(df, "s"), org, cands)
    assert "legal_name" not in proposals
    assert {c.path for c in cands["legal_name"][:2]} == {"org_name_a", "org_name_b"}


def test_spec_keeps_ranked_candidates(erp_df: pl.DataFrame, org: CanonicalEntity) -> None:
    spec = propose_mapping(erp_df, org, source_id="erp")
    assert spec.candidates["legal_name"][0].path == "VNDR_NM"
    assert spec.candidates["phone"]  # unmapped targets keep candidates for reviewers


def _decisions(org: CanonicalEntity, legal: str, conf: float = 0.95) -> dict[str, Any]:
    rest = [
        decision(f.name, [], conf=0.2, rationale="") for f in org.fields if f.name != "legal_name"
    ]
    return {"decisions": [decision("legal_name", [legal], conf=conf), *rest]}


def test_self_consistency_majority_vote(org: CanonicalEntity) -> None:
    df = pl.DataFrame({"company": ["A Co", "B Co"], "brand": ["A", "B"]})
    fake = FakeProvider(
        responses=[
            _decisions(org, "company"),
            _decisions(org, "brand"),
            _decisions(org, "company"),
        ]
    )
    engine = MappingEngine(OPS, MappingConfig(self_consistency=3), gateway=LLMGateway(fake))
    proposal = run_sync(engine.propose(profile_table(df, "s"), org, df))
    legal = next(f for f in proposal.fields if f.target == "legal_name")
    assert legal.sources == ["company"] and "agreement 2/3" in legal.rationale
    assert len(fake.calls) == 3
    orders = {tuple(d["path"] for d in (c.payload() or {})["source_fields"]) for c in fake.calls}
    assert len(orders) > 1  # prompts vary the field order


def test_vote_scales_confidence_by_agreement(org: CanonicalEntity) -> None:
    from meldr.mapping.adjudicator import LLMMappingResponse

    runs = [
        {d.target: d for d in LLMMappingResponse.model_validate(_decisions(org, s)).decisions}
        for s in ("a", "a", "b", "a")
    ]
    assert vote(runs, ["legal_name"])["legal_name"].confidence == pytest.approx(0.95 * 3 / 4)


def test_shared_source_goes_to_review(org: CanonicalEntity) -> None:
    df = pl.DataFrame({"contact": ["a@b.co", "c@d.co"]})
    decisions = _decisions(org, "contact")
    decisions["decisions"][0]["target"] = "email"
    decisions["decisions"].append(
        {
            "target": "legal_name",
            "sources": ["contact"],
            "steps": [],
            "confidence": 0.99,
            "rationale": "r",
        }
    )
    decisions["decisions"] = [
        d for d in decisions["decisions"] if not (d["target"] == "email" and not d["sources"])
    ]
    engine = MappingEngine(
        OPS, MappingConfig(), gateway=LLMGateway(FakeProvider(responses=[decisions]))
    )
    fields = run_sync(engine.propose(profile_table(df, "s"), org, df)).fields
    assert all(
        f.review_reason == "shared_source:contact" for f in fields if f.sources == ["contact"]
    )


def test_approved_mappings_are_learned(tmp_path: Path, org: CanonicalEntity) -> None:
    store = FileMappingStore(tmp_path)
    first = pl.DataFrame({"VNDR_ZZ": ["Acme", "Beta"], "id": ["1", "2"]})
    spec = propose_mapping(first, org, source_id="erp")
    spec = review.edit(spec, "legal_name", "sources: [VNDR_ZZ]", user="ana", ops=OPS)
    store.save(spec)
    second = profile_table(pl.DataFrame({"vndr_zz": ["Gamma"]}), "crm")
    learned = ApprovedMappingMatcher(store)(second, org)
    assert [c.path for c in learned["legal_name"]] == ["vndr_zz"]


def test_large_vocabularies_are_shortlisted() -> None:
    vocab = [f"diagnosis {i}" for i in range(100)] + ["adenocarcinoma"]
    request = value_map_request(["adenocarcinoma nos"], vocab)
    assert "vocabulary" not in request
    [item] = request["values"]  # type: ignore[misc]
    assert item["candidates"][0] == "adenocarcinoma" and len(item["candidates"]) == 10
    assert value_map_request(["x"], ["a", "b"]) == {"vocabulary": ["a", "b"], "values": ["x"]}


def test_json_schema_round_trip_and_ranges(org: CanonicalEntity) -> None:
    ranged = org.model_copy(
        update={
            "fields": [
                f.model_copy(update={"minimum": 0}) if f.name == "annual_revenue" else f
                for f in org.fields
            ]
        }
    )
    schema = ranged.to_json_schema()
    assert schema["properties"]["founded_on"] == {
        "type": "string",
        "format": "date",
        "description": "Date the organization was founded",
        "x-synonyms": [
            "founded",
            "founding_date",
            "established",
            "established_date",
            "incorporation_date",
            "est_date",
        ],
    }
    back = CanonicalEntity.from_json_schema(schema)
    assert back == ranged
    with pytest.raises(ConfigError):
        CanonicalEntity.from_json_schema({"type": "object"})

    df = pl.DataFrame({"org_id": ["1", "2"], "legal_name": ["A", "B"], "rev": ["-5", "7"]})
    spec = review.approve(propose_mapping(df, ranged, source_id="s"), user="t", above=0.0)
    spec = review.edit(
        spec, "annual_revenue", "sources: [rev]\nsteps: [{op: parse_number}]", user="t", ops=OPS
    )
    out = apply_mapping(df, spec, ranged)
    assert out.error_counts == {"out_of_range:annual_revenue": 1}


def test_table_benchmark_suite(tmp_path: Path) -> None:
    case = tmp_path / "case1"
    case.mkdir()
    SOURCE.write_csv(case / "source_table.csv")
    TARGET.write_csv(case / "target_table.csv")
    (case / "ground_truth.json").write_text(
        json.dumps(
            {
                "matches": [
                    {"source_column": "total_changes", "target_column": "total_schedule_changes"},
                    {"source_column": "changes", "target_column": "total_budget_changes"},
                    {"source_column": "report_date", "target_column": "date_reported_as_of"},
                ]
            }
        )
    )
    report = run_sync(run_table_eval(tmp_path, {"heuristic": builtin_factories()["heuristic"]}))
    [result] = report.results
    assert report.suite == "tables" and result.hit_at_1 == 1.0 and result.value_accuracy is None
    assert CanonicalField(name="x", dtype=DataType.STRING).examples == []
