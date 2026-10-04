"""Dry-run proposed mappings on sample rows (§4.9)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from meldr.core.canonical import CanonicalEntity
from meldr.core.mapping import DryRunStats, FieldMapping
from meldr.execution.executor import field_exprs
from meldr.execution.validator import field_checks
from meldr.transforms.registry import TransformRegistry

MAX_EXAMPLES = 10


@dataclass
class FieldDryRun:
    stats: DryRunStats
    examples: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def failure(self) -> float:
        return max(self.stats.loss_rate, self.stats.invalid_rate)


def dry_run_field(
    sample: pl.DataFrame, mapping: FieldMapping, entity: CanonicalEntity, ops: TransformRegistry
) -> FieldDryRun:
    """Loss and invalid rates for one mapping on `sample`, plus the failed examples."""
    rows = sample.height

    def failed(error: str) -> FieldDryRun:
        stats = DryRunStats(rows=rows, loss_rate=1.0, invalid_rate=0.0, failed_examples=rows)
        return FieldDryRun(stats, error=error)

    missing = [s for s in mapping.sources if s not in sample.columns]
    if missing:
        return failed(f"missing sources: {missing}")
    target = entity.field(mapping.target)
    try:
        value, lost = field_exprs(mapping, entity, ops)
        out = sample.select(
            *[pl.col(s).alias(f"__src{i}") for i, s in enumerate(mapping.sources)],
            value,
            lost.alias("__lost"),
        )
    except Exception as e:  # a broken mapping is a failed proposal, not a crash
        return failed(type(e).__name__)
    src_cols = [f"__src{i}" for i in range(len(mapping.sources))]
    any_src = pl.any_horizontal([pl.col(c).is_not_null() for c in src_cols])
    invalid_checks = [e for reason, e in field_checks(target) if not reason.startswith("required")]
    invalid = pl.any_horizontal(invalid_checks) if invalid_checks else pl.lit(False)
    flagged = out.with_columns(any_src.alias("__any"), invalid.fill_null(False).alias("__invalid"))
    nonnull = int(flagged["__any"].sum())
    n_lost = int(flagged["__lost"].sum())
    n_invalid = int(flagged["__invalid"].sum())
    failing = flagged.filter(pl.col("__lost") | pl.col("__invalid"))
    examples = [
        {
            "input": [None if r[c] is None else str(r[c]) for c in src_cols],
            "output": None if r[target.name] is None else str(r[target.name]),
            "problem": "value_lost" if r["__lost"] else "invalid",
        }
        for r in failing.head(MAX_EXAMPLES).iter_rows(named=True)
    ]
    denom = max(nonnull, 1)
    return FieldDryRun(
        DryRunStats(
            rows=rows,
            loss_rate=round(n_lost / denom, 4),
            invalid_rate=round(n_invalid / denom, 4),
            failed_examples=failing.height,
        ),
        examples=examples,
    )


def dry_run(
    sample: pl.DataFrame,
    mappings: Sequence[FieldMapping],
    entity: CanonicalEntity,
    ops: TransformRegistry,
) -> dict[str, FieldDryRun]:
    return {m.target: dry_run_field(sample, m, entity, ops) for m in mappings}
