"""Validate executor output against the canonical entity and split off quarantine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import DataType
from meldr.core.errors import MeldrError
from meldr.execution.executor import LOSS_PREFIX, ROW, source_rows

ERRORS = "_meldr_errors"


@dataclass
class Validated:
    valid: pl.DataFrame
    quarantine: pl.DataFrame
    error_counts: dict[str, int]


def field_checks(f: CanonicalField) -> list[tuple[str, pl.Expr]]:
    """(cause, expression that is True when the row FAILS) for one canonical field."""
    col = pl.col(f.name)
    checks: list[tuple[str, pl.Expr]] = []
    if f.required:
        checks.append((f"required_missing:{f.name}", col.is_null()))
    if f.vocabulary is not None:
        vocab = list(f.vocabulary)
        if f.dtype == DataType.ARRAY:
            outside = col.list.eval(~pl.element().cast(pl.String).is_in(vocab)).list.any()
        else:
            outside = ~col.cast(pl.String, strict=False).is_in(vocab)
        checks.append((f"value_not_in_vocabulary:{f.name}", col.is_not_null() & outside))
    if f.minimum is not None or f.maximum is not None:
        low = col < f.minimum if f.minimum is not None else pl.lit(False)
        high = col > f.maximum if f.maximum is not None else pl.lit(False)
        checks.append((f"out_of_range:{f.name}", col.is_not_null() & (low | high)))
    if f.pattern and f.dtype in (DataType.STRING, DataType.UNKNOWN, DataType.JSON):
        checks.append(
            (
                f"pattern_mismatch:{f.name}",
                col.is_not_null() & ~col.cast(pl.String).str.contains(f.pattern),
            )
        )
    return checks


def validate(
    df: pl.DataFrame,
    entity: CanonicalEntity,
    *,
    loss_fields: Sequence[str] = (),
    on_invalid: Literal["quarantine", "drop", "fail"] = "quarantine",
    source: pl.DataFrame | None = None,
) -> Validated:
    """Split `df` into valid rows and quarantined rows, with the failure causes in `_meldr_errors`.

    If you give `source` (the table that `execute()` used to make `df`), quarantined rows also
    get their original values in `_meldr_source_row`.
    """
    reasons: list[pl.Expr] = []
    for f in entity.fields:
        for reason, failing in field_checks(f):
            reasons.append(pl.when(failing).then(pl.lit(reason)).otherwise(None))
    for t in loss_fields:
        col = LOSS_PREFIX + t
        if col in df.columns:
            reasons.append(pl.when(pl.col(col)).then(pl.lit(f"value_lost:{t}")).otherwise(None))
    if reasons:
        errors = pl.concat_list(reasons).list.drop_nulls().alias(ERRORS)
    else:
        errors = pl.lit([], dtype=pl.List(pl.String)).alias(ERRORS)
    checked = df.with_columns(errors)
    loss_cols = [c for c in checked.columns if c.startswith(LOSS_PREFIX)]
    bad = pl.col(ERRORS).list.len() > 0
    valid = checked.filter(~bad).drop([ERRORS, ROW, *loss_cols], strict=False)
    quarantine = checked.filter(bad).drop(loss_cols)
    if source is not None and ROW in quarantine.columns:
        quarantine = quarantine.with_columns(source_rows(source, quarantine[ROW]))
    quarantine = quarantine.drop(ROW, strict=False)
    counts: dict[str, int] = {}
    if quarantine.height:
        exploded = (
            quarantine.select(pl.col(ERRORS).explode(empty_as_null=True)).group_by(ERRORS).len()
        )
        counts = {str(r): int(n) for r, n in exploded.iter_rows()}
    if quarantine.height and on_invalid == "fail":
        raise MeldrError(
            f"{quarantine.height} rows failed validation for {entity.name} "
            f"({', '.join(sorted(counts))}). Set execution.on_invalid=quarantine to quarantine "
            "these rows and continue."
        )
    if on_invalid == "drop":
        quarantine = quarantine.head(0)
    return Validated(valid=valid, quarantine=quarantine, error_counts=counts)
