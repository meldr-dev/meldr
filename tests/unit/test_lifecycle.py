from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from meldr import Meldr
from meldr._compat import UTC
from meldr.api import Applied
from meldr.core.mapping import FieldMapping, FieldStatus, Origin
from meldr.core.segments import ExtractedRecord
from meldr.observability import RunReport
from meldr.pipeline.results import ExtractResult, MeldrResult
from tests.helpers import make_config, make_spec


def _report() -> RunReport:
    return RunReport(
        run_id="run",
        source_id="source",
        target="Entity",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        rows_in=2,
        rows_out=1,
        rows_quarantined=1,
        llm_calls=3,
        warnings=["warning"],
    )


def test_meldr_result_to_dict_masks_rows() -> None:
    result = MeldrResult(
        data=pl.DataFrame(
            {
                "id": ["1"],
                "email": ["a@example.com"],
                "_meldr_source_row": ['{"email":"a@example.com"}'],
                "_meldr_loss__email": [False],
            }
        ),
        quarantine=pl.DataFrame({"id": ["2"], "email": ["b@example.com"]}),
        spec=make_spec(
            [
                FieldMapping(
                    target="id",
                    sources=["id"],
                    confidence=1.0,
                    origin=Origin.HUMAN,
                    status=FieldStatus.APPROVED,
                )
            ]
        ),
        pending_review=[],
        report=_report(),
        pii_fields=("email",),
    )

    summary = result.to_dict()
    assert summary["rows_valid"] == 1
    assert "rows" not in summary

    rows = result.to_dict(include_rows=True, limit=1)
    assert rows["rows"][0]["email"] == "***"
    assert rows["quarantine"][0]["email"] == "***"
    assert "_meldr_source_row" not in rows["rows"][0]
    assert "_meldr_loss__email" not in rows["rows"][0]
    assert result.to_dict(include_rows=True, reveal_pii=True)["rows"][0]["email"] == "a@example.com"


def test_extract_result_to_dict_limits_rows() -> None:
    record = ExtractedRecord(
        entity="Entity",
        entity_version="1.0.0",
        record_id="1",
        values={"name": "A"},
        evidence={},
        confidence={},
    )
    result = ExtractResult(
        records=[record],
        data=pl.DataFrame({"name": ["A", "B"]}),
        quarantine=pl.DataFrame(),
        segments=[],
        report=_report(),
    )

    output = result.to_dict(include_rows=True, limit=1)
    assert output["rows"] == [{"name": "A"}]
    assert output["records"] == [record.model_dump(mode="json")]


def test_applied_to_dict_masks_rows() -> None:
    applied = Applied(
        data=pl.DataFrame({"email": ["a@example.com"]}),
        quarantine=pl.DataFrame(),
        pii_fields=("email",),
    )

    assert applied.to_dict()["rows_valid"] == 1
    assert applied.to_dict(include_rows=True)["rows"][0]["email"] == "***"


def test_meldr_context_manager_closes_resources(tmp_path: Path) -> None:
    with Meldr(make_config(tmp_path / "state")) as mr:
        assert mr.table_sink is not None
    assert mr._closed


async def test_meldr_async_context_manager_closes_resources(tmp_path: Path) -> None:
    async with Meldr(make_config(tmp_path / "state")) as mr:
        assert mr.table_sink is not None
    assert mr._closed
