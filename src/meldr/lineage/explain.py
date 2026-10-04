"""Field-level lineage: which source paths and steps made a value."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl
from pydantic import BaseModel

from meldr.core.errors import MeldrError
from meldr.core.mapping import TransformStep
from meldr.extraction.records import EXTRACTED
from meldr.indexing.sinks.base import TableSink
from meldr.lineage.store import LineageStore
from meldr.mapping.store import MappingStore
from meldr.observability import RunReport


class Explanation(BaseModel):
    entity: str
    record_id: str
    field: str
    value: Any
    source_id: str
    source_paths: list[str]
    steps: list[TransformStep]
    origin: str | None
    confidence: float | None
    spec_id: str
    spec_version: int
    run_id: str
    run: RunReport | None = None


def explain(
    tables: TableSink,
    specs: MappingStore,
    lineage: LineageStore,
    entity: str,
    record_id: str,
    field: str,
    *,
    source_id: str | None = None,
    primary_key: Sequence[str] = (),
) -> Explanation:
    """Explain one harmonized value. `record_id` can also be a single-column primary key value."""
    df = tables.read(entity)
    if df.height == 0 or "_meldr_record_id" not in df.columns:
        raise MeldrError(f"No harmonized rows for {entity}. Run meld() first.")
    match = pl.col("_meldr_record_id") == record_id
    if len(primary_key) == 1 and primary_key[0] in df.columns:
        match = match | (pl.col(primary_key[0]).cast(pl.String) == record_id)
    rows = df.filter(match)
    if source_id is not None:
        rows = rows.filter(pl.col("_meldr_source_id") == source_id)
    if rows.height == 0:
        raise MeldrError(f"Record '{record_id}' not found in {entity}.")
    if field not in rows.columns:
        raise MeldrError(f"{entity} has no field '{field}'.")
    row = rows.row(0, named=True)
    spec_id, version = str(row["_meldr_spec_id"]), int(row["_meldr_spec_version"])
    extracted = spec_id.startswith(EXTRACTED)
    mapping = None if extracted else specs.get(spec_id, version).mapping_for(field)
    return Explanation(
        entity=entity,
        record_id=str(row["_meldr_record_id"]),
        field=field,
        value=row[field],
        source_id=str(row["_meldr_source_id"]),
        source_paths=mapping.sources if mapping else [],
        steps=mapping.steps if mapping else [],
        origin=str(mapping.origin) if mapping else ("extraction" if extracted else None),
        confidence=mapping.confidence if mapping else None,
        spec_id=spec_id,
        spec_version=version,
        run_id=str(row["_meldr_run_id"]),
        run=lineage.get(str(row["_meldr_run_id"])),
    )
