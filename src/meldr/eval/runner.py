"""Eval runner: compare providers on the synthetic or table-pair suite, then check a baseline."""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from meldr._compat import UTC
from meldr.api import apropose_mapping
from meldr.config import MappingConfig
from meldr.core.canonical import CanonicalEntity
from meldr.eval.calibration import Calibration, calibrate
from meldr.eval.metrics import Aggregate, Case, VariantScore, aggregate, score_spec
from meldr.eval.synth import make_variants
from meldr.eval.tables import load_pairs
from meldr.llm.gateway import LLMGateway
from meldr.llm.pricing import UsageTotals
from meldr.llm.providers.fake import FakeProvider, FakeRequest
from meldr.pipeline.oracle import heuristic_oracle
from meldr.transforms.registry import TransformRegistry, default_registry

GatewayFactory = Callable[[Case], LLMGateway | None]


def truth_oracle(case: Case) -> Callable[[FakeRequest], dict[str, Any]]:
    """A fake model that knows the ground truth, to test the full pipeline from end to end."""

    def oracle(req: FakeRequest) -> dict[str, Any]:
        payload = req.payload() or {}
        if payload.get("task") not in ("mapping", "mapping_repair"):
            return heuristic_oracle(req)
        decisions = []
        for t in payload.get("target_fields", []):
            name = t["name"]
            if name in case.truth:
                steps = getattr(case, "truth_steps", {}).get(name, [])
                decisions.append(
                    {
                        "target": name,
                        "sources": case.truth[name],
                        "steps": [s.model_dump() for s in steps],
                        "confidence": 0.97,
                        "rationale": "ground truth",
                    }
                )
            else:
                decisions.append(
                    {
                        "target": name,
                        "sources": [],
                        "steps": [],
                        "confidence": 0.3,
                        "rationale": "not present",
                    }
                )
        return {"decisions": decisions}

    return oracle


def builtin_factories() -> dict[str, GatewayFactory]:
    return {
        "heuristic": lambda v: None,
        "fake": lambda v: LLMGateway(FakeProvider(mode="oracle", oracle=truth_oracle(v))),
        "fake-heuristic": lambda v: LLMGateway(
            FakeProvider(mode="oracle", oracle=heuristic_oracle)
        ),
    }


@dataclass
class EvalReport:
    suite: str
    entities: list[str]
    variants: int
    seed: int
    created_at: str
    results: list[Aggregate] = field(default_factory=list)
    per_variant: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    calibration: dict[str, Calibration] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)


async def evaluate_provider(
    name: str,
    factory: GatewayFactory,
    cases: Sequence[Case],
    *,
    cfg: MappingConfig,
    ops: TransformRegistry,
) -> tuple[Aggregate, list[VariantScore]]:
    scores: list[VariantScore] = []
    usage = UsageTotals()
    t0 = time.monotonic()
    for v in cases:
        gateway = factory(v)
        spec = await apropose_mapping(
            v.source, v.entity, source_id=v.name, llm=gateway, config=cfg, ops=ops
        )
        scores.append(score_spec(spec, v, ops))
        if gateway is not None:
            usage = usage.merged(gateway.totals)
    agg = aggregate(
        name,
        scores,
        llm_calls=usage.calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        est_cost_usd=usage.est_cost_usd,
        seconds=time.monotonic() - t0,
    )
    return agg, scores


async def run_eval(
    entities: Sequence[CanonicalEntity],
    providers: dict[str, GatewayFactory],
    *,
    variants: int = 10,
    rows: int = 200,
    seed: int = 7,
    cfg: MappingConfig | None = None,
    target_precision: float = 0.97,
) -> EvalReport:
    cfg = cfg or MappingConfig()
    cases = [v for e in entities for v in make_variants(e, variants, rows=rows, seed=seed)]
    return await _run(
        "synthetic",
        [e.name for e in entities],
        cases,
        providers,
        cfg=cfg,
        seed=seed,
        target_precision=target_precision,
    )


async def run_table_eval(
    root: str | Path,
    providers: dict[str, GatewayFactory],
    *,
    cfg: MappingConfig | None = None,
    target_precision: float = 0.97,
) -> EvalReport:
    """Evaluate on table pairs (Valentine / NYU layout) in `root`."""
    pairs = load_pairs(root)
    return await _run(
        "tables",
        [p.name for p in pairs],
        pairs,
        providers,
        cfg=cfg or MappingConfig(),
        target_precision=target_precision,
    )


async def _run(
    suite: str,
    names: list[str],
    cases: Sequence[Case],
    providers: dict[str, GatewayFactory],
    *,
    cfg: MappingConfig,
    seed: int = 0,
    target_precision: float = 0.97,
) -> EvalReport:
    ops = default_registry()
    report = EvalReport(
        suite=suite,
        entities=names,
        variants=len(cases),
        seed=seed,
        created_at=datetime.now(UTC).isoformat(),
    )
    for name, factory in providers.items():
        agg, scores = await evaluate_provider(name, factory, cases, cfg=cfg, ops=ops)
        report.results.append(agg)
        report.calibration[name] = calibrate(
            [p for s in scores for p in s.points], target_precision=target_precision
        )
        report.per_variant[name] = [
            {
                "variant": s.variant,
                "f1": round(s.f1, 4),
                "value_accuracy": s.value_accuracy,
                "perturbations": getattr(case, "perturbations", []),
            }
            for case, s in zip(cases, scores, strict=True)
        ]
    return report


def check_baseline(
    report: EvalReport, baseline_path: Path, *, tolerance: float = 0.02
) -> list[str]:
    """Regressions (empty = pass): metrics more than `tolerance` below `baseline_path`."""
    baseline = json.loads(baseline_path.read_text())
    failures = []
    for r in report.results:
        base = baseline.get(r.provider)
        if not base:
            continue
        for metric in ("f1", "value_accuracy", "hit_at_1", "mrr"):
            value = getattr(r, metric)
            if metric in base and value is not None and value < base[metric] - tolerance:
                failures.append(
                    f"{r.provider}.{metric} = {getattr(r, metric):.4f} < baseline "
                    f"{base[metric]:.4f} - {tolerance}"
                )
    return failures
