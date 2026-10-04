"""Compile approved field mappings into one Polars select and run it."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from meldr._compat import UTC
from meldr.core.canonical import CanonicalEntity
from meldr.core.dtypes import as_text, to_polars
from meldr.core.errors import SpecValidationError
from meldr.core.mapping import FieldMapping
from meldr.execution.record_id import content_id, key_id_expr, namespace_for
from meldr.indexing.sinks.base import SEQUENCE
from meldr.transforms.registry import TransformRegistry

LOSS_PREFIX = "_meldr_loss__"
SOURCE_ROW = "_meldr_source_row"
ROW = "_meldr_row"  # position in the input table (validate() drops it)


@dataclass
class Lineage:
    """Values stamped on every output row."""

    source_id: str
    spec_id: str
    spec_version: int
    run_id: str
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    record_namespace: uuid.UUID | None = None


@dataclass
class ExecutionOutput:
    frame: pl.DataFrame
    loss_fields: list[str]
    warnings: list[str] = field(default_factory=list)


def field_exprs(
    mapping: FieldMapping, entity: CanonicalEntity, ops: TransformRegistry
) -> tuple[pl.Expr, pl.Expr]:
    """(value, lost) for one target: the steps, then a non-strict cast to the target dtype.

    `lost` is true where a source value did not survive. The exception is a null that an
    intentional-null step (`nullify`) made. If a step after that step fails, the loss still counts.
    """
    target = entity.field(mapping.target)
    args = ops.validate(mapping)  # also enforces one source unless the first op combines sources
    exprs = [pl.col(s) for s in mapping.sources]
    first = ops.get(mapping.steps[0].op) if mapping.steps else None
    if first is not None and first.arity == "multi":
        expr = first.build(exprs, args[0])
        rest = list(zip(mapping.steps[1:], args[1:], strict=True))
    else:
        expr = exprs[0]
        rest = list(zip(mapping.steps, args, strict=True))
    intended: pl.Expr | None = None
    for step, a in rest:
        op = ops.get(step.op)
        expr = op.build([expr], a)
        if op.nulls_intentionally:
            intended = expr.is_null()
    value = expr.cast(to_polars(target.dtype), strict=False).alias(target.name)
    lost = _loss_expr(mapping, value)
    return value, lost if intended is None else (lost & ~intended).alias(LOSS_PREFIX + target.name)


def _loss_expr(mapping: FieldMapping, expr: pl.Expr) -> pl.Expr:
    """True where some source is non-null but the output is null."""
    any_source = pl.any_horizontal([pl.col(s).is_not_null() for s in mapping.sources])
    return (any_source & expr.is_null()).alias(LOSS_PREFIX + mapping.target)


def source_rows(table: pl.DataFrame, rows: pl.Series) -> pl.Series:
    """Original values of the given rows as JSON text. It builds the text only for these rows."""
    columns = [c for c in table.columns if not c.startswith("_meldr_")]
    if not columns:
        return pl.Series(SOURCE_ROW, ["{}"] * len(rows), dtype=pl.String)
    picked = table.select(pl.col(columns).gather(rows))
    text = pl.struct([as_text(c, table.schema[c]) for c in columns])
    return picked.select(text.struct.json_encode().alias(SOURCE_ROW)).to_series()


def _plan(
    table: pl.DataFrame,
    exprs: dict[str, tuple[pl.Expr, pl.Expr]],
    by_target: dict[str, FieldMapping],
    entity: CanonicalEntity,
    lineage: Lineage,
) -> tuple[pl.LazyFrame, list[str]]:
    selected = [
        exprs[f.name][0]
        if f.name in exprs
        else pl.lit(None, dtype=to_polars(f.dtype)).alias(f.name)
        for f in entity.fields
    ]
    # A mapping that failed to build or run gives nulls. Count these nulls as lost.
    # Then validate() quarantines the rows, and the nulls do not overwrite stored values.
    failed = [
        t
        for t, m in by_target.items()
        if t not in exprs and all(s in table.columns for s in m.sources)
    ]
    loss_targets = [*exprs, *failed]
    losses = [
        exprs[t][1] if t in exprs else _loss_expr(by_target[t], pl.lit(None)) for t in loss_targets
    ]
    key = entity.primary_key if all(k in exprs for k in entity.primary_key) else []
    sequence = (
        [pl.col(entity.sequence_field).alias(SEQUENCE)] if entity.sequence_field is not None else []
    )
    plan = (
        table.lazy()
        .select([*selected, *losses, pl.int_range(pl.len(), dtype=pl.UInt32).alias(ROW)])
        .with_columns(
            pl.lit(lineage.source_id).alias("_meldr_source_id"),
            key_id_expr(key, _namespace(lineage, entity)).alias("_meldr_record_id"),
            pl.lit(lineage.spec_id).alias("_meldr_spec_id"),
            pl.lit(lineage.spec_version, dtype=pl.Int64).alias("_meldr_spec_version"),
            pl.lit(lineage.run_id).alias("_meldr_run_id"),
            pl.lit(lineage.ingested_at, dtype=pl.Datetime("us", "UTC")).alias("_meldr_ingested_at"),
            *sequence,
        )
    )
    return plan, loss_targets


def _namespace(lineage: Lineage, entity: CanonicalEntity) -> uuid.UUID:
    return lineage.record_namespace or namespace_for(lineage.source_id, entity.name)


def _fill_content_ids(
    frame: pl.DataFrame, table: pl.DataFrame, namespace: uuid.UUID
) -> pl.DataFrame:
    keyless = frame.select(pl.arg_where(pl.col("_meldr_record_id").is_null())).to_series()
    if keyless.len() == 0:
        return frame
    ids = [content_id(row, namespace) for row in source_rows(table, keyless)]
    return frame.with_columns(frame["_meldr_record_id"].scatter(keyless, ids))


def execute(
    table: pl.DataFrame,
    mappings: Sequence[FieldMapping],
    entity: CanonicalEntity,
    ops: TransformRegistry,
    lineage: Lineage,
) -> ExecutionOutput:
    """Run `mappings`. If the expression of a field fails, the field is null and counts as lost."""
    warnings: list[str] = []
    by_target = {m.target: m for m in mappings if m.target in entity.field_names()}
    exprs: dict[str, tuple[pl.Expr, pl.Expr]] = {}
    for t, m in by_target.items():
        if any(s not in table.columns for s in m.sources):
            warnings.append(f"missing_source:{t}")
            continue
        try:
            exprs[t] = field_exprs(m, entity, ops)
        except SpecValidationError:
            warnings.append(f"invalid_mapping:{t}")
    if not entity.primary_key or not all(k in exprs for k in entity.primary_key):
        warnings.append(
            "no_primary_key: record ids are content hashes, so an update adds a new record"
        )
    plan, loss_targets = _plan(table, exprs, by_target, entity, lineage)
    try:
        frame = plan.collect()
    except Exception:
        # Isolate the field(s) that fail. The rest of the batch still runs.
        good: dict[str, tuple[pl.Expr, pl.Expr]] = {}
        for t, pair in exprs.items():
            try:
                table.lazy().select(*pair).collect()
                good[t] = pair
            except Exception:
                warnings.append(f"field_error:{t}")
        plan, loss_targets = _plan(table, good, by_target, entity, lineage)
        frame = plan.collect()
    frame = _fill_content_ids(frame, table, _namespace(lineage, entity))
    return ExecutionOutput(frame=frame, loss_fields=loss_targets, warnings=warnings)
