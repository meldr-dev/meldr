"""Field-level P/R/F1, cell-level value accuracy, and auto-approve precision."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol

import polars as pl

from meldr.core.canonical import CanonicalEntity
from meldr.core.mapping import FieldStatus, MappingSpec
from meldr.execution.executor import Lineage, execute
from meldr.transforms.registry import TransformRegistry


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, and F1 (an empty prediction or truth set counts as perfect)."""
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class VariantScore:
    variant: str
    tp: int
    fp: int
    fn: int
    approved: int
    approved_correct: int
    cells: int
    cells_correct: int
    targets: int = 0
    hits: dict[int, int] = field(default_factory=dict)
    reciprocal_ranks: float = 0.0
    points: list[tuple[float, bool]] = field(default_factory=list)  # (confidence, correct)

    @property
    def f1(self) -> float:
        return _prf(self.tp, self.fp, self.fn)[2]

    @property
    def value_accuracy(self) -> float | None:
        return self.cells_correct / self.cells if self.cells else None


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(a, datetime) and isinstance(b, datetime):
        return a == b
    if isinstance(a, date) and isinstance(b, date):
        return a == b
    return str(a).strip().casefold() == str(b).strip().casefold()


class Case(Protocol):
    """A benchmark case: a source table, a target entity, and the true target -> sources."""

    name: str
    source: pl.DataFrame
    entity: CanonicalEntity
    truth: dict[str, list[str]]


HIT_KS = (1, 3, 5)


def score_spec(spec: MappingSpec, case: Case, ops: TransformRegistry) -> VariantScore:
    """Compare a spec (all non-rejected fields) against the case's ground truth.

    Value accuracy works only with expected canonical rows (`case.expected`). HitRate@k and MRR
    use the ranked candidates that the spec stores.
    """
    mapped = [f for f in spec.fields if f.status != FieldStatus.REJECTED and f.sources]
    predicted = {(f.target, frozenset(f.sources)) for f in mapped}
    truth = {(t, frozenset(s)) for t, s in case.truth.items()}
    approved = [f for f in mapped if f.status == FieldStatus.APPROVED]
    hits = dict.fromkeys(HIT_KS, 0)
    reciprocal = 0.0
    for target, sources in case.truth.items():
        ranked = [a.path for a in spec.candidates.get(target, [])]
        for k in HIT_KS:
            hits[k] += bool(set(sources) & set(ranked[:k]))
        rank = next((i for i, p in enumerate(ranked, start=1) if p in sources), None)
        reciprocal += 1 / rank if rank else 0.0
    cells = cells_correct = 0
    expected: pl.DataFrame | None = getattr(case, "expected", None)
    if expected is not None and expected.width:
        out = execute(case.source, mapped, case.entity, ops, Lineage("eval", spec.spec_id, 0, "e"))
        for target in case.truth:
            for e, g in zip(expected[target].to_list(), out.frame[target].to_list(), strict=True):
                if e is not None:
                    cells += 1
                    cells_correct += _same(g, e)
    return VariantScore(
        variant=case.name,
        tp=len(predicted & truth),
        fp=len(predicted - truth),
        fn=len(truth - predicted),
        approved=len(approved),
        approved_correct=sum(1 for f in approved if (f.target, frozenset(f.sources)) in truth),
        cells=cells,
        cells_correct=cells_correct,
        targets=len(case.truth),
        hits=hits,
        reciprocal_ranks=reciprocal,
        points=[(f.confidence, (f.target, frozenset(f.sources)) in truth) for f in mapped],
    )


@dataclass
class Aggregate:
    provider: str
    variants: int
    precision: float
    recall: float
    f1: float
    value_accuracy: float | None
    hit_at_1: float | None
    hit_at_3: float | None
    hit_at_5: float | None
    mrr: float | None
    auto_approve_precision: float | None
    auto_approved: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    est_cost_usd: float | None
    seconds: float


def aggregate(
    provider: str,
    scores: list[VariantScore],
    *,
    llm_calls: int,
    input_tokens: int,
    output_tokens: int,
    est_cost_usd: float | None,
    seconds: float,
) -> Aggregate:
    p, r, f1 = _prf(sum(s.tp for s in scores), sum(s.fp for s in scores), sum(s.fn for s in scores))
    cells = sum(s.cells for s in scores)
    approved = sum(s.approved for s in scores)
    targets = sum(s.targets for s in scores)
    return Aggregate(
        provider=provider,
        variants=len(scores),
        precision=round(p, 4),
        recall=round(r, 4),
        f1=round(f1, 4),
        value_accuracy=round(sum(s.cells_correct for s in scores) / cells, 4) if cells else None,
        hit_at_1=_hit_rate(scores, 1),
        hit_at_3=_hit_rate(scores, 3),
        hit_at_5=_hit_rate(scores, 5),
        mrr=round(sum(s.reciprocal_ranks for s in scores) / targets, 4) if targets else None,
        auto_approve_precision=round(sum(s.approved_correct for s in scores) / approved, 4)
        if approved
        else None,
        auto_approved=approved,
        llm_calls=llm_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        est_cost_usd=est_cost_usd,
        seconds=round(seconds, 2),
    )


def _hit_rate(scores: list[VariantScore], k: int) -> float | None:
    targets = sum(s.targets for s in scores)
    return round(sum(s.hits.get(k, 0) for s in scores) / targets, 4) if targets else None
