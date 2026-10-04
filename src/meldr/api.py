"""Functional API: propose a mapping, then apply it. No config, spec store, or sinks.

Other pipelines embed Meldr through this API (for example, an ingest worker that already
has rows). It has two calls:

- `propose_mapping`: rows + canonical entity -> MappingSpec.
- `apply_mapping`: rows + approved MappingSpec -> canonical rows.

`Meldr` adds storage, drift checks, review, and indexing on top of these two calls.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

import polars as pl

from meldr._async import run_sync
from meldr._compat import UTC
from meldr.config import MappingConfig
from meldr.core.canonical import CanonicalEntity
from meldr.core.errors import ConnectorError
from meldr.core.mapping import MappingSpec
from meldr.core.types import RawBatch
from meldr.execution.executor import LOSS_PREFIX, Lineage, execute
from meldr.execution.validator import validate
from meldr.llm.base import Embedder
from meldr.llm.gateway import LLMGateway, usage_scope
from meldr.llm.policy import SourcePolicy
from meldr.mapping import review
from meldr.mapping.engine import MappingEngine, build_spec
from meldr.observability import RunReport
from meldr.pipeline.results import HarmonizeResult, _frame_rows, _json_safe
from meldr.profiling.flatten import flatten
from meldr.profiling.profiler import profile_table
from meldr.transforms.registry import TRANSFORMS_VERSION, TransformRegistry, default_registry


@dataclass
class Applied:
    """Result of `apply_mapping` for one batch of rows."""

    data: pl.DataFrame
    quarantine: pl.DataFrame
    values_lost: dict[str, int] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    pii_fields: tuple[str, ...] = ()

    def to_dict(
        self, *, include_rows: bool = False, limit: int = 100, reveal_pii: bool = False
    ) -> dict[str, Any]:
        """Return a JSON-safe summary of one applied batch."""
        result: dict[str, Any] = {
            "rows_valid": self.data.height,
            "rows_quarantined": self.quarantine.height,
            "values_lost": dict(self.values_lost),
            "error_counts": dict(self.error_counts),
            "warnings": list(self.warnings),
        }
        if include_rows:
            result["rows"] = _frame_rows(
                self.data, limit=limit, pii_fields=self.pii_fields, reveal_pii=reveal_pii
            )
            result["quarantine"] = _frame_rows(
                self.quarantine, limit=limit, pii_fields=self.pii_fields, reveal_pii=reveal_pii
            )
        return cast(dict[str, Any], _json_safe(result))


HarmonizeInput = (
    RawBatch | pl.DataFrame | Sequence[Mapping[str, Any]] | Sequence[tuple[str, bytes]] | Any
)


def coerce_batch(value: HarmonizeInput, *, source_id: str) -> RawBatch:
    """Convert caller-owned input into a RawBatch without storage access."""
    if isinstance(value, RawBatch):
        return value
    if isinstance(value, pl.DataFrame):
        return RawBatch.from_polars(value, source_id)
    if isinstance(value, (str, bytes, bytearray)):
        raise ConnectorError("harmonize() must have a batch, table, rows, or named file bytes.")
    if hasattr(value, "__class__") and value.__class__.__module__.split(".")[0] == "pandas":
        return RawBatch.from_pandas(value, source_id)
    if not isinstance(value, Sequence):
        raise ConnectorError(
            "harmonize() must have a RawBatch, dataframe, list of rows, or named file bytes."
        )
    items = list(value)
    if not items:
        raise ConnectorError("harmonize() must have at least one row or named file.")
    if all(isinstance(item, Mapping) for item in items):
        return RawBatch.from_dicts(cast(Sequence[Mapping[str, Any]], items), source_id)
    if all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], bytes)
        for item in items
    ):
        return RawBatch.from_contents(cast(Sequence[tuple[str, bytes]], items), source_id)
    raise ConnectorError("harmonize() list items must be mappings or (name, bytes) file pairs.")


async def apropose_mapping(
    df: pl.DataFrame,
    entity: CanonicalEntity,
    *,
    source_id: str = "source",
    llm: LLMGateway | None = None,
    config: MappingConfig | None = None,
    ops: TransformRegistry | None = None,
    embedder: Embedder | None = None,
    source_policy: SourcePolicy | None = None,
) -> MappingSpec:
    cfg = config or MappingConfig()
    schema = profile_table(df, source_id, sample_rows=cfg.sample_rows, sample_size=cfg.sample_size)
    engine = MappingEngine(ops or default_registry(), cfg, gateway=llm, embedder=embedder)
    proposal = await engine.propose(
        schema, entity, df.head(cfg.dryrun_rows), source_policy=source_policy
    )
    return build_spec(schema, entity, proposal)


async def aharmonize(
    value: pl.DataFrame | Sequence[Mapping[str, Any]],
    entity: CanonicalEntity,
    *,
    source_id: str = "source",
    llm: LLMGateway | None = None,
    config: MappingConfig | None = None,
    ops: TransformRegistry | None = None,
    embedder: Embedder | None = None,
    source_policy: SourcePolicy | None = None,
    spec: MappingSpec | None = None,
    on_invalid: Literal["quarantine", "drop", "fail"] = "quarantine",
) -> HarmonizeResult:
    """Harmonize caller-owned table data without stores, sinks, or indexing."""
    started_at = datetime.now(UTC)
    started = time.monotonic()
    with usage_scope() as usage:
        if isinstance(value, pl.DataFrame):
            df = value
        else:
            rows = list(value)
            if not rows or not all(isinstance(row, Mapping) for row in rows):
                raise ConnectorError(
                    "aharmonize() must have a dataframe or a non-empty list of rows."
                )
            df = flatten(rows, source_id=source_id).table
        chosen = spec
        cache_hit = spec is not None
        if chosen is None:
            chosen = await apropose_mapping(
                df,
                entity,
                source_id=source_id,
                llm=llm,
                config=config,
                ops=ops,
                embedder=embedder,
                source_policy=source_policy,
            )
        applied = apply_mapping(
            df,
            chosen,
            entity,
            ops=ops,
            on_invalid=on_invalid,
        )
    report = RunReport(
        run_id=uuid.uuid4().hex[:16],
        source_id=source_id,
        target=entity.name,
        started_at=started_at,
        duration_s=round(time.monotonic() - started, 4),
        rows_in=df.height,
        rows_out=applied.data.height,
        rows_quarantined=applied.quarantine.height,
        fields_mapped=len(chosen.approved_fields()),
        fields_pending=len(chosen.pending_fields()),
        values_lost=applied.values_lost,
        spec_id=chosen.spec_id,
        spec_version=chosen.version,
        spec_cache_hit=cache_hit,
        llm_calls=usage.calls,
        llm_cache_hits=usage.cache_hits,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        est_cost_usd=usage.est_cost_usd,
        tokens_by_model={m: item.model_dump() for m, item in usage.by_model.items()},
        warnings=applied.warnings,
    )
    return HarmonizeResult(
        data=applied.data,
        quarantine=applied.quarantine,
        review=review.items(chosen),
        children={},
        report=report,
        spec=chosen,
    )


def harmonize(
    value: pl.DataFrame | Sequence[Mapping[str, Any]],
    entity: CanonicalEntity,
    *,
    source_id: str = "source",
    llm: LLMGateway | None = None,
    config: MappingConfig | None = None,
    ops: TransformRegistry | None = None,
    embedder: Embedder | None = None,
    source_policy: SourcePolicy | None = None,
    spec: MappingSpec | None = None,
    on_invalid: Literal["quarantine", "drop", "fail"] = "quarantine",
) -> HarmonizeResult:
    """Synchronously harmonize caller-owned table data."""
    return run_sync(
        aharmonize(
            value,
            entity,
            source_id=source_id,
            llm=llm,
            config=config,
            ops=ops,
            embedder=embedder,
            source_policy=source_policy,
            spec=spec,
            on_invalid=on_invalid,
        )
    )


def propose_mapping(
    df: pl.DataFrame,
    entity: CanonicalEntity,
    *,
    source_id: str = "source",
    llm: LLMGateway | None = None,
    config: MappingConfig | None = None,
    ops: TransformRegistry | None = None,
    embedder: Embedder | None = None,
    source_policy: SourcePolicy | None = None,
) -> MappingSpec:
    """Profile `df` and propose a spec (only heuristics when `llm` is None). It saves nothing."""
    return run_sync(
        apropose_mapping(
            df,
            entity,
            source_id=source_id,
            llm=llm,
            config=config,
            ops=ops,
            embedder=embedder,
            source_policy=source_policy,
        )
    )


def apply_mapping(
    df: pl.DataFrame,
    spec: MappingSpec,
    entity: CanonicalEntity,
    *,
    ops: TransformRegistry | None = None,
    run_id: str | None = None,
    record_namespace: uuid.UUID | str | None = None,
    on_invalid: Literal["quarantine", "drop", "fail"] = "quarantine",
    include_proposed: bool = False,
) -> Applied:
    """Deterministically map `df` with the spec's approved fields (no LLM, no I/O).

    `record_namespace` sets the UUID namespace of `_meldr_record_id`. The default is one namespace
    for each source and entity. To share ids with the destination table, pass its stable id.
    """
    mappings = spec.runnable_fields(include_proposed=include_proposed)
    namespace = (
        uuid.UUID(record_namespace) if isinstance(record_namespace, str) else (record_namespace)
    )
    lineage = Lineage(
        spec.source_id,
        spec.spec_id,
        spec.version,
        run_id or uuid.uuid4().hex[:16],
        record_namespace=namespace,
    )
    out = execute(df, mappings, entity, ops or default_registry(), lineage)
    warnings = list(out.warnings)
    pinned = spec.generator.transforms_version
    if pinned is not None and pinned != TRANSFORMS_VERSION:
        warnings.append(f"transforms_version_changed:{pinned}->{TRANSFORMS_VERSION}")
    lost = {
        t: int(out.frame[LOSS_PREFIX + t].sum())
        for t in out.loss_fields
        if out.frame[LOSS_PREFIX + t].any()
    }
    checked = validate(
        out.frame, entity, loss_fields=out.loss_fields, on_invalid=on_invalid, source=df
    )
    return Applied(
        data=checked.valid,
        quarantine=checked.quarantine,
        values_lost=lost,
        error_counts=checked.error_counts,
        warnings=warnings,
        pii_fields=tuple(field.name for field in entity.fields if field.pii),
    )
