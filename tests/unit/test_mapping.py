from __future__ import annotations

import uuid
from pathlib import Path

import polars as pl
import pytest

from meldr.config import MappingConfig
from meldr.core.canonical import CanonicalEntity
from meldr.core.errors import SpecValidationError
from meldr.core.mapping import FieldStatus, Origin, SpecStatus
from meldr.execution.executor import Lineage, execute
from meldr.execution.record_id import namespace_for, record_id
from meldr.execution.validator import validate
from meldr.llm.gateway import LLMGateway
from meldr.llm.providers.fake import FakeProvider
from meldr.mapping import drift, review
from meldr.mapping.candidates import generate
from meldr.mapping.engine import MappingEngine, build_spec
from meldr.mapping.normalize import normalize
from meldr.mapping.store import FileMappingStore
from meldr.mapping.values import match_values
from meldr.profiling.profiler import profile_table
from meldr.transforms.registry import default_registry
from tests.helpers import decision

OPS = default_registry()


def test_normalize() -> None:
    assert normalize("VNDR_NM") == "vendor name"
    assert normalize("customerName") == "customer name"
    assert normalize("tbl_REV_CENTS") == "revenue cents"


def test_match_values() -> None:
    vm = match_values(
        ["S", "V", "cust", "Vendors", "zz"],
        ["customer", "vendor", "subcontractor", "partner", "other"],
    )
    assert vm.mapping == {
        "S": "subcontractor",
        "V": "vendor",
        "cust": "customer",
        "Vendors": "vendor",
    }
    assert vm.residue == ["zz"]


async def test_candidates_rank_exact_names(erp_df: pl.DataFrame, org: CanonicalEntity) -> None:
    cands = await generate(profile_table(erp_df, "erp"), org)
    assert cands["legal_name"][0].path == "VNDR_NM"
    assert cands["annual_revenue"][0].path == "REV_CENTS"


async def test_heuristic_only_never_auto_approves(
    erp_df: pl.DataFrame, org: CanonicalEntity
) -> None:
    engine = MappingEngine(OPS, MappingConfig())
    p = await engine.propose(profile_table(erp_df, "erp"), org, erp_df)
    assert p.fields and all(f.status == FieldStatus.PROPOSED for f in p.fields)
    assert {f.origin for f in p.fields} == {Origin.HEURISTIC}
    steps = {f.target: [s.op for s in f.steps] for f in p.fields}
    assert steps["annual_revenue"] == ["scale"] and steps["founded_on"] == ["parse_date"]


async def test_adjudicator_repairs_invalid_decisions(org: CanonicalEntity) -> None:
    df = pl.DataFrame({"org_id": ["1", "2"], "company": ["A Co", "B Co"]})
    rest = [decision(f.name, []) for f in org.fields if f.name not in ("org_id", "legal_name")]
    bad = {
        "decisions": [
            decision("org_id", ["org_id", "company"]),  # wrong arity
            decision("legal_name", ["missing_col"]),  # unknown source path
            *rest,
        ]
    }
    good = {
        "decisions": [
            decision("org_id", ["org_id"]),
            decision("legal_name", ["company"], [{"op": "trim"}]),
            *rest,
        ]
    }
    fake = FakeProvider(responses=[bad, good])
    engine = MappingEngine(OPS, MappingConfig(), gateway=LLMGateway(fake))
    p = await engine.propose(profile_table(df, "src"), org, df)
    by = {f.target: f for f in p.fields}
    assert by["legal_name"].sources == ["company"] and by["org_id"].sources == ["org_id"]
    assert "unknown source path" in fake.calls[1].last_user_text
    assert by["org_id"].status == FieldStatus.APPROVED


async def test_llm_failure_falls_back_to_heuristics(
    erp_df: pl.DataFrame, org: CanonicalEntity
) -> None:
    engine = MappingEngine(
        OPS, MappingConfig(), gateway=LLMGateway(FakeProvider(responses=[]), max_retries=0)
    )
    p = await engine.propose(profile_table(erp_df, "erp"), org, erp_df)
    assert p.warnings == ["llm_unavailable:LLMError"]
    assert p.fields


async def test_self_correction_escalation_and_value_residue(org: CanonicalEntity) -> None:
    df = pl.DataFrame(
        {
            "org_id": ["1", "2"],
            "company": ["A", "B"],
            "kind": ["V", "ZZ"],
            "est": ["2001-03-14", "1999-12-25"],
        }
    )
    others = [
        decision(f.name, [])
        for f in org.fields
        if f.name not in ("org_id", "legal_name", "founded_on", "org_type")
    ]
    first = {
        "decisions": [
            decision("org_id", ["org_id"]),
            decision("legal_name", ["company"], conf=0.6),
            decision("org_type", ["kind"]),
            decision(
                "founded_on", ["est"], [{"op": "parse_date", "args": {"formats": ["%m/%d/%Y"]}}]
            ),
            *others,
        ]
    }
    residue = {"mapping": {"ZZ": "partner"}, "confidence": 0.8}
    fixed = {
        "decisions": [
            decision(
                "founded_on", ["est"], [{"op": "parse_date", "args": {"formats": ["%Y-%m-%d"]}}]
            )
        ]
    }
    cheap = FakeProvider(responses=[first, residue, fixed])
    strong = FakeProvider(responses=[{"decisions": [decision("legal_name", ["company"])]}])
    engine = MappingEngine(
        OPS, MappingConfig(), gateway=LLMGateway(cheap), escalation=LLMGateway(strong)
    )
    p = await engine.propose(profile_table(df, "src"), org, df)
    by = {f.target: f for f in p.fields}
    assert by["founded_on"].steps[0].args["formats"] == ["%Y-%m-%d"]
    assert by["founded_on"].dryrun.loss_rate == 0  # type: ignore[union-attr]
    assert by["org_type"].steps[-1].args["mapping"] == {"V": "vendor", "ZZ": "partner"}
    assert by["legal_name"].rationale.startswith("[escalated]")
    assert len(strong.calls) == 1


async def test_store_versions_review_and_drift(
    tmp_path: Path, erp_df: pl.DataFrame, org: CanonicalEntity
) -> None:
    store = FileMappingStore(tmp_path)
    schema = profile_table(erp_df, "erp")
    proposal = await MappingEngine(OPS, MappingConfig()).propose(schema, org, erp_df)
    v1 = store.save(build_spec(schema, org, proposal))
    assert (
        v1.version == 1 and store.find_approved("erp", "Organization", schema.fingerprint) is None
    )

    v2 = store.save(review.approve(v1, user="ana", above=0.0))
    assert v2.version == 2 and v2.parent_version == 1 and v2.approved_by == "ana"
    assert v2.status == SpecStatus.APPROVED
    assert store.find_approved("erp", "Organization", schema.fingerprint) == v2
    assert store.get("erp.Organization", 1).status == SpecStatus.DRAFT  # v1 untouched
    assert [e["version"] for e in store.audit_log("erp.Organization")] == [1, 2]

    edited = review.edit(v2, "city", "sources: [CTRY]\nsteps: [{op: trim}]", user="ana", ops=OPS)
    assert edited.mapping_for("city").origin == Origin.HUMAN  # type: ignore[union-attr]
    with pytest.raises(SpecValidationError):
        review.edit(v2, "city", "sources: [NOPE]", user="ana", ops=OPS)

    removed = profile_table(erp_df.drop("VNDR_NM"), "erp")
    rec = drift.reconcile(v2, removed, org)
    assert rec.report.breaking_targets == ["legal_name"] and "legal_name" in rec.retarget
    added = profile_table(erp_df.with_columns(pl.lit("x").alias("PHONE_NO")), "erp")
    rec2 = drift.reconcile(v2, added, org)
    assert rec2.report.added == ["PHONE_NO"]
    assert set(rec2.retarget) == set(v2.unmapped_targets)
    assert drift.spec_diff(v1, v2)[0]["change"] == "modified"


def test_refresh_values(tmp_path: Path, erp_df: pl.DataFrame, org: CanonicalEntity) -> None:
    from meldr._async import run_sync

    schema = profile_table(erp_df, "erp")
    spec = build_spec(
        schema, org, run_sync(MappingEngine(OPS, MappingConfig()).propose(schema, org, erp_df))
    )
    grown = erp_df.with_columns(pl.lit("P").alias("VNDR_TYP_CD"))
    updated, unseen = drift.refresh_values(spec, grown, org)
    assert unseen == {"org_type": ["P"]} and updated is not None
    assert updated.mapping_for("org_type").steps[-1].args["mapping"]["P"] == "partner"  # type: ignore[union-attr]


def test_record_ids_are_stable_collision_safe_and_datagrid_compatible() -> None:
    ns = uuid.UUID("6f0f5d7e-3c2a-4c1b-9a8e-2d4b6c8a0e1f")
    # Reference value computed with datagrid's generateRecordId (uuid v5 in Node).
    assert record_id(["V1", "Acme, Inc.", None], ns) == "259c2d1b-7f4f-55db-affa-c084ab282f4f"
    assert record_id(["a|b", "c"], ns) != record_id(["a", "b|c"], ns)
    assert namespace_for("crm", "Organization") != namespace_for("erp", "Organization")


def test_executor_and_validator(erp_df: pl.DataFrame, org: CanonicalEntity) -> None:
    from meldr.core.mapping import FieldMapping, TransformStep

    def fm(t: str, s: str, steps: list[TransformStep] | None = None) -> FieldMapping:
        return FieldMapping(
            target=t, sources=[s], steps=steps or [], confidence=1, origin=Origin.HUMAN
        )

    mappings = [
        fm("org_id", "VNDR_ID"),
        fm("legal_name", "VNDR_NM"),
        fm("email", "CONTACT_EMAIL"),
        fm("founded_on", "EST_DT", [TransformStep(op="parse_date", args={"formats": ["%Y"]})]),
        fm("country", "LAST_UPD_USR"),
    ]
    out = execute(erp_df, mappings, org, OPS, Lineage("erp", "erp.Organization", 1, "run"))
    assert {"_meldr_source_id", "_meldr_record_id", "_meldr_run_id"} <= set(out.frame.columns)
    ns = namespace_for("erp", "Organization")
    assert out.frame["_meldr_record_id"].to_list() == [
        record_id([v], ns) for v in erp_df["VNDR_ID"]
    ]
    checked = validate(out.frame, org, loss_fields=out.loss_fields, source=erp_df)
    assert checked.valid.height == 0 and checked.quarantine.height == 4
    assert checked.error_counts["value_lost:founded_on"] == 3
    assert checked.error_counts["pattern_mismatch:country"] == 4
    assert "_meldr_source_row" in checked.quarantine.columns
