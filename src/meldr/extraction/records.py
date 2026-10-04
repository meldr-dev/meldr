"""Turn extracted records into validated canonical rows (same executor and validator as tables)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import polars as pl

from meldr.core.canonical import CanonicalEntity
from meldr.core.dtypes import DataType
from meldr.core.mapping import FieldMapping, Origin, TransformStep
from meldr.core.segments import ExtractedRecord
from meldr.execution.executor import Lineage, execute
from meldr.execution.validator import Validated, validate
from meldr.profiling.dates import DATE_FORMATS, DATETIME_FORMATS
from meldr.transforms.registry import TransformRegistry

EXTRACTED = "extract."  # spec id prefix of extracted rows (they have no spec file)


def spec_id(entity: CanonicalEntity) -> str:
    """Lineage spec id stamped on extracted rows."""
    return f"{EXTRACTED}{entity.name}"


def _parse_steps(dtype: DataType) -> list[TransformStep]:
    if dtype == DataType.DATE:
        return [TransformStep(op="parse_date", args={"formats": DATE_FORMATS})]
    if dtype == DataType.DATETIME:
        return [
            TransformStep(op="parse_datetime", args={"formats": [*DATETIME_FORMATS, "%Y-%m-%d"]})
        ]
    if dtype in (DataType.INTEGER, DataType.FLOAT, DataType.DECIMAL):
        return [TransformStep(op="parse_number")]
    if dtype == DataType.BOOLEAN:
        return [TransformStep(op="parse_bool")]
    return []


def to_rows(
    entity: CanonicalEntity,
    records: Sequence[ExtractedRecord],
    ops: TransformRegistry,
    lineage: Lineage,
    *,
    on_invalid: Literal["quarantine", "drop", "fail"] = "quarantine",
) -> Validated:
    """Cast extracted values to canonical types, stamp lineage, and validate."""
    names = entity.field_names()
    raw = pl.DataFrame(
        {
            n: [None if r.values.get(n) is None else str(r.values[n]) for r in records]
            for n in names
        },
        schema=dict.fromkeys(names, pl.String),
    )
    mappings = [
        FieldMapping(
            target=f.name,
            sources=[f.name],
            steps=_parse_steps(f.dtype),
            confidence=1.0,
            origin=Origin.LLM,
        )
        for f in entity.fields
    ]
    out = execute(raw, mappings, entity, ops, lineage)
    frame = out.frame.with_columns(
        pl.Series("_meldr_record_id", [r.record_id for r in records], dtype=pl.String)
    )
    return validate(frame, entity, loss_fields=out.loss_fields, on_invalid=on_invalid, source=raw)
