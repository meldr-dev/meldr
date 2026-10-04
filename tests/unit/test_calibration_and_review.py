"""Threshold calibration, ranking metrics, review ordering, and numbers embedded in labels."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from meldr.api import apply_mapping, propose_mapping
from meldr.core.canonical import CanonicalEntity
from meldr.core.mapping import (
    Alternative,
    FieldMapping,
    FieldStatus,
    Origin,
)
from meldr.eval.calibration import calibrate, wilson_lower
from meldr.eval.metrics import aggregate, score_spec
from meldr.eval.tables import entity_from_table
from meldr.mapping import review
from meldr.profiling.profiler import profile_table
from meldr.transforms.registry import default_registry
from tests.helpers import make_spec


def test_wilson_lower_bound() -> None:
    assert wilson_lower(0, 0) == 0.0
    # With no errors the bound is n / (n + z^2): 125 clean matches are needed to show 97%.
    assert wilson_lower(124, 124) < 0.97 <= wilson_lower(125, 125)
    assert wilson_lower(50, 100) == pytest.approx(0.4038, abs=1e-4)


def test_calibrate_picks_lowest_safe_threshold() -> None:
    points = [(0.9, True)] * 200 + [(0.8, True)] * 100 + [(0.8, False)] * 10 + [(0.3, False)] * 30
    cal = calibrate(points, target_precision=0.97)
    assert cal.threshold == 0.9 and cal.precision == 1.0
    assert cal.auto_rate == round(200 / len(points), 4)
    assert sum(int(b["count"]) for b in cal.reliability) == len(points)
    assert calibrate(points, target_precision=0.9).threshold == 0.8


def test_calibrate_admits_ties_and_refuses_small_samples() -> None:
    tied = calibrate([(0.9, True)] * 199 + [(0.9, False)], target_precision=0.97)
    assert tied.threshold == 0.9 and tied.precision == 0.995 and tied.auto_rate == 1.0
    small = calibrate([(0.95, True)] * 77, target_precision=0.97)
    assert small.threshold is None and small.auto_rate == 0.0 and small.points == 77


def _mapping(target: str, status: FieldStatus, confidence: float = 0.5) -> FieldMapping:
    return FieldMapping(
        target=target,
        sources=[target],
        confidence=confidence,
        origin=Origin.HEURISTIC,
        status=status,
    )


@dataclass
class _Case:
    name: str
    source: pl.DataFrame
    entity: CanonicalEntity
    truth: dict[str, list[str]]


def test_mrr_and_calibration_points() -> None:
    fields = [_mapping("a", FieldStatus.PROPOSED, 0.4), _mapping("b", FieldStatus.APPROVED, 0.9)]
    candidates = {
        "a": [Alternative(path="x", score=0.9), Alternative(path="a", score=0.8)],
        "b": [Alternative(path="b", score=0.7)],
    }
    df = pl.DataFrame({"a": ["1"], "b": ["2"]})
    case = _Case("c", df, entity_from_table(df, "E"), {"a": ["a"], "b": ["b"]})
    score = score_spec(make_spec(fields, candidates), case, default_registry())
    assert score.reciprocal_ranks == 1.5
    assert score.points == [(0.4, True), (0.9, True)]
    agg = aggregate(
        "p", [score], llm_calls=0, input_tokens=0, output_tokens=0, est_cost_usd=None, seconds=0
    )
    assert agg.mrr == 0.75 and agg.hit_at_1 == 0.5


def test_pending_fields_put_required_then_ambiguous_first() -> None:
    fields = [
        _mapping("clear", FieldStatus.PROPOSED),
        _mapping("close", FieldStatus.PROPOSED),
        _mapping("done", FieldStatus.APPROVED),
        _mapping("a", FieldStatus.PROPOSED),
    ]
    candidates = {
        "clear": [Alternative(path="clear", score=0.9), Alternative(path="z", score=0.2)],
        "close": [Alternative(path="close", score=0.61), Alternative(path="y", score=0.6)],
    }
    order = [f.target for f in make_spec(fields, candidates).pending_fields()]
    assert order == ["a", "close", "clear"]


def test_number_inside_a_label_is_extracted() -> None:
    source = pl.DataFrame({"district": ["District 14", "District 3", "District 27"]})
    hints = profile_table(source, "s").field("district").hints
    assert hints.get("embedded_number") is True
    entity = entity_from_table(pl.DataFrame({"district": [14, 3, 27]}), "Office")
    spec = review.approve(propose_mapping(source, entity, source_id="s"), user="t", above=0.0)
    assert apply_mapping(source, spec, entity).data["district"].to_list() == [14, 3, 27]
    phones = pl.DataFrame({"p": ["555-1234", "555-9876", "555-0000"]})
    assert "embedded_number" not in profile_table(phones, "s").field("p").hints
