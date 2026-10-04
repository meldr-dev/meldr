"""The Meldr facade: profile -> map -> transform -> validate -> store -> index."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from meldr._async import run_sync
from meldr._compat import UTC
from meldr._tenant import validate_tenant
from meldr.api import apply_mapping, coerce_batch
from meldr.config import MeldrConfig, SourceConfig
from meldr.connectors.base import Connector, build_connector
from meldr.core.canonical import CanonicalEntity, SchemaRegistry
from meldr.core.errors import ConfigError, DriftBlockedError, MeldrError
from meldr.core.mapping import MappingSpec, ReviewDecision, SpecStatus
from meldr.core.schema import SourceSchema, shape_fingerprint
from meldr.core.segments import ContentSegment, ExtractedRecord, Relation
from meldr.core.types import Modality, RawBatch, RawRecord, mime_for_path, sha256_bytes
from meldr.execution.executor import Lineage
from meldr.extraction import records as extracted_rows
from meldr.extraction.extractor import Extractor
from meldr.extraction.transcribe import Transcriber
from meldr.indexing.indexer import SEGMENTS, Indexer, records_collection
from meldr.indexing.sinks import build_table_sink, build_vector_sink
from meldr.indexing.sinks.base import (
    RECORD_KEY,
    SEQUENCE,
    Describable,
    SearchHit,
    TableSink,
    TenantTableSink,
    TenantVectorSink,
    VectorSink,
    entity_docs,
    unsuperseded,
)
from meldr.lineage.explain import Explanation
from meldr.lineage.explain import explain as explain_value
from meldr.lineage.store import (
    LineageStore,
    SQLiteLineageStore,
    SQLLineageStore,
    TenantLineageStore,
)
from meldr.llm.base import Embedder
from meldr.llm.cache import (
    ResponseCache,
    SQLiteResponseCache,
    SQLiteVectorCache,
    SQLResponseCache,
    SQLVectorCache,
    TenantResponseCache,
    TenantVectorCache,
    VectorCache,
)
from meldr.llm.gateway import LLMGateway, usage_scope
from meldr.llm.policy import DataPolicy, SourcePolicy
from meldr.llm.pricing import Pricing, UsageTotals
from meldr.llm.providers import build_embedder, build_provider
from meldr.mapping import drift, review
from meldr.mapping.discovery import discover
from meldr.mapping.engine import MappingEngine, Proposal, build_spec
from meldr.mapping.memory import ApprovedMappingMatcher
from meldr.mapping.store import (
    FileMappingStore,
    MappingStore,
    SQLMappingStore,
    TenantMappingStore,
)
from meldr.observability import RunReport, otel_span, safe_log, set_span_attributes
from meldr.pipeline.oracle import heuristic_oracle
from meldr.pipeline.results import ExtractResult, HarmonizeResult, MeldrResult
from meldr.plugins import load_all
from meldr.profiling.flatten import flatten
from meldr.profiling.profiler import profile_table, source_columns
from meldr.resolution.resolver import Resolution, resolve
from meldr.transforms.registry import TransformRegistry, default_registry


async def _close_resource(resource: Any) -> None:
    aclose = getattr(resource, "aclose", None)
    if callable(aclose):
        if inspect.iscoroutinefunction(aclose):
            await aclose()
        else:
            result = await asyncio.to_thread(aclose)
            if inspect.isawaitable(result):
                await result
        return
    close = getattr(resource, "close", None)
    if callable(close):
        result = await asyncio.to_thread(close)
        if inspect.isawaitable(result):
            await result


class _Connectors(Mapping[str, Connector]):
    """Connectors built lazily from `config.sources` on first access."""

    def __init__(self, sources: Mapping[str, SourceConfig], batch_size: int) -> None:
        self._sources = sources
        self._batch_size = batch_size
        self._built: dict[str, Connector] = {}
        self._lock = threading.Lock()

    def __getitem__(self, key: str) -> Connector:
        with self._lock:
            if key not in self._built:
                if key not in self._sources:
                    raise KeyError(f"No source '{key}' in config. Known: {sorted(self._sources)}")
                self._built[key] = build_connector(key, self._sources[key], self._batch_size)
            return self._built[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)


@dataclass
class _Run:
    usage: UsageTotals
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    t0: float = field(default_factory=time.monotonic)


ContentResolver = Callable[[str], bytes | Awaitable[bytes]]


class Meldr:
    """Meld RawBatches from all connector types into canonical entities."""

    def __init__(
        self,
        config: MeldrConfig | None = None,
        *,
        registry: SchemaRegistry | None = None,
        llm: LLMGateway | None = None,
        embedder: Embedder | None = None,
        store: MappingStore | None = None,
        lineage: LineageStore | None = None,
        llm_cache: ResponseCache | None = None,
        embedding_cache: VectorCache | None = None,
        table_sink: TableSink | None = None,
        vector_sink: VectorSink | None = None,
        ops: TransformRegistry | None = None,
        tenant: str | None = None,
    ) -> None:
        self.config = config or MeldrConfig()
        self.tenant = validate_tenant(tenant)
        self._closed = False
        cfg = self.config
        self.root = Path(cfg.storage.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or self._load_registry()
        self.ops = ops or default_registry()
        plugins = load_all(ops_registry=self.ops)
        sql_url = cfg.stores.url if cfg.stores.type == "sql" else None
        if cfg.stores.type == "sql" and sql_url is None:
            raise ConfigError("stores.url is necessary when stores.type is 'sql'.")
        self._llm_cache = llm_cache or (
            SQLResponseCache(sql_url, tenant=self.tenant)
            if sql_url is not None
            else SQLiteResponseCache(self.root / "cache" / "llm.sqlite", tenant=self.tenant)
        )
        if llm_cache is not None and self.tenant is not None:
            self._llm_cache = (
                llm_cache
                if getattr(llm_cache, "tenant", None) == self.tenant
                else TenantResponseCache(llm_cache, self.tenant)
            )
        self._extraction_cache = SQLiteResponseCache(
            self.root / "cache" / "extractions.sqlite", tenant=self.tenant
        )
        self._gateways: dict[str, LLMGateway] = {}
        self._gateway_lock = threading.Lock()
        routing = cfg.llm.routing
        default_name = routing.mapping or (
            next(iter(cfg.llm.providers)) if len(cfg.llm.providers) == 1 else None
        )
        self.llm = llm if llm is not None else self.gateway(default_name)
        self.escalation = self.gateway(routing.escalation)
        self.extraction_llm = self.gateway(routing.extraction) or self.llm
        self.vision_llm = self.gateway(routing.vision) or self.extraction_llm
        self.embedder = embedder or build_embedder(cfg.embeddings)
        self.store: MappingStore = store or (
            SQLMappingStore(sql_url, tenant=self.tenant)
            if sql_url is not None
            else FileMappingStore(self.root, tenant=self.tenant)
        )
        if store is not None and self.tenant is not None:
            self.store = (
                store
                if getattr(store, "tenant", None) == self.tenant
                else TenantMappingStore(store, self.tenant)
            )
        self.table_sink = table_sink or build_table_sink(cfg.sinks, self.root)
        if self.tenant is not None:
            self.table_sink = TenantTableSink(self.table_sink, self.tenant)
        self._described: set[tuple[str, str]] = set()
        self._describe_lock = threading.Lock()
        self.vector_sink = vector_sink or build_vector_sink(cfg.sinks, self.root)
        if self.tenant is not None:
            self.vector_sink = TenantVectorSink(self.vector_sink, self.tenant)
        self.lineage = lineage or (
            SQLLineageStore(sql_url, tenant=self.tenant)
            if sql_url is not None
            else SQLiteLineageStore(self.root / "lineage.sqlite", tenant=self.tenant)
        )
        if lineage is not None and self.tenant is not None:
            self.lineage = (
                lineage
                if getattr(lineage, "tenant", None) == self.tenant
                else TenantLineageStore(lineage, self.tenant)
            )
        self._embedding_cache = embedding_cache or (
            SQLVectorCache(sql_url, tenant=self.tenant)
            if sql_url is not None
            else SQLiteVectorCache(self.root / "cache" / "embeddings.sqlite", tenant=self.tenant)
        )
        if embedding_cache is not None and self.tenant is not None:
            self._embedding_cache = (
                embedding_cache
                if getattr(embedding_cache, "tenant", None) == self.tenant
                else TenantVectorCache(embedding_cache, self.tenant)
            )
        self.indexer = Indexer(
            self.embedder,
            self.vector_sink,
            cache=self._embedding_cache,
            batch_size=cfg.indexing.batch_size,
            max_concurrency=cfg.llm.max_concurrency,
            max_retries=cfg.llm.max_retries,
        )
        self.engine = MappingEngine(
            self.ops,
            cfg.mapping,
            gateway=self.llm,
            escalation=self.escalation,
            escalate_below=cfg.llm.escalate_below,
            embedder=self.embedder if cfg.embeddings.provider != "fake" else None,
            matchers=[
                *([ApprovedMappingMatcher(self.store)] if cfg.mapping.learn_from_approved else []),
                *plugins["meldr.matchers"].values(),
            ],
        )
        self.connectors: Mapping[str, Connector] = _Connectors(
            cfg.sources, cfg.execution.batch_size
        )
        self.transcriber: Transcriber | None = None

    @classmethod
    def from_config(cls, path: str | Path) -> Meldr:
        """Build a Meldr from a YAML config file."""
        return cls(MeldrConfig.load(path))

    def _ensure_open(self) -> None:
        if self._closed:
            raise MeldrError("Meldr is closed. Make a new instance before the next call.")

    async def aclose(self) -> None:
        """Close Meldr resources. A second call has no effect."""
        if self._closed:
            return
        self._closed = True
        resources: list[Any] = [
            self.llm,
            self.escalation,
            self.extraction_llm,
            self.vision_llm,
            *self._gateways.values(),
            self.embedder,
            self.table_sink,
            self.vector_sink,
            self.store,
            self.lineage,
            self._llm_cache,
            self._extraction_cache,
            self._embedding_cache,
        ]
        seen: set[int] = set()
        for resource in resources:
            if id(resource) in seen:
                continue
            seen.add(id(resource))
            await _close_resource(resource)

    def close(self) -> None:
        """Close Meldr resources. A second call has no effect."""
        run_sync(self.aclose())

    def __enter__(self) -> Meldr:
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    async def __aenter__(self) -> Meldr:
        self._ensure_open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    def _load_registry(self) -> SchemaRegistry:
        from meldr.domains import pack_path

        reg = SchemaRegistry()
        schemas = self.config.schemas
        for p in schemas.paths or ([] if schemas.linkml else [str(pack_path("generic"))]):
            reg.load_dir(p)
        for p in schemas.linkml:
            reg.load_linkml(p)
        return reg

    def gateway(self, name: str | None) -> LLMGateway | None:
        """The gateway for a configured provider name (built on the first call, then cached)."""
        if name is None:
            return None
        with self._gateway_lock:
            if name in self._gateways:
                return self._gateways[name]
            cfg = self.config
            pcfg = cfg.llm.providers.get(name)
            if pcfg is None:
                raise ConfigError(f"llm.routing references unknown provider '{name}'.")
            provider = build_provider(name, pcfg, fake_oracle=heuristic_oracle)
            policy = (
                DataPolicy(allowed=set(pcfg.data_policy))
                if pcfg.data_policy is not None
                else DataPolicy.default_for(is_local=provider.is_local)
            )
            pricing = Pricing(per_million={m: (p.input, p.output) for m, p in cfg.pricing.items()})
            gw = LLMGateway(
                provider,
                policy=policy,
                cache=self._llm_cache,
                max_concurrency=cfg.llm.max_concurrency,
                max_retries=cfg.llm.max_retries,
                max_repairs=cfg.llm.max_repairs,
                timeout_s=cfg.llm.timeout_s,
                pricing=pricing,
            )
            self._gateways[name] = gw
            return gw

    async def _finish(self, run: _Run, *, record_lineage: bool = True, **fields: Any) -> RunReport:
        """Fill the duration and LLM usage of `run`, and record the run in lineage."""
        report = RunReport(
            run_id=run.id,
            started_at=run.started_at,
            duration_s=round(time.monotonic() - run.t0, 4),
            llm_calls=run.usage.calls,
            llm_cache_hits=run.usage.cache_hits,
            input_tokens=run.usage.input_tokens,
            output_tokens=run.usage.output_tokens,
            est_cost_usd=run.usage.est_cost_usd,
            tokens_by_model={m: u.model_dump() for m, u in run.usage.by_model.items()},
            **fields,
        )
        if record_lineage:
            await asyncio.to_thread(self.lineage.record_run, report)
        return report

    def _source_cfg(self, source_id: str) -> SourceConfig | None:
        return self.config.sources.get(source_id)

    def _source_policy(self, source_id: str) -> SourcePolicy | None:
        src = self._source_cfg(source_id)
        return src.data_policy if src else None

    def _table(self, batch: RawBatch) -> tuple[pl.DataFrame, list[str]]:
        if batch.table is not None:
            return batch.table, []
        if batch.modality != Modality.JSON:
            raise MeldrError(
                f"meld() accepts only a tabular or JSON batch, not {batch.modality}. "
                "Use extract() for documents and media."
            )
        src = self._source_cfg(batch.source_id)
        arrays = src.arrays if src else "json"
        flat = flatten(
            [r.data or {} for r in batch.records],
            record_ids=[r.record_id for r in batch.records],
            arrays=arrays,
            source_id=batch.source_id,
        )
        warnings = list(flat.warnings)
        warnings.extend(f"child_table:{name}" for name in flat.children)
        return flat.table, warnings

    def _profile(self, batch: RawBatch, table: pl.DataFrame) -> SourceSchema:
        meta = batch.source_metadata
        src = self._source_cfg(batch.source_id)
        descriptions = dict(meta.get("descriptions") or {})
        return profile_table(
            table,
            batch.source_id,
            entity_hint=(src.entity_hint if src and src.entity_hint else meta.get("entity_hint")),
            descriptions=descriptions,
            sample_rows=self.config.mapping.sample_rows,
            sample_size=self.config.mapping.sample_size,
        )

    def to_table(self, batch: RawBatch) -> pl.DataFrame:
        """The batch as one table, as meld() sees it (JSON records are flattened)."""
        self._ensure_open()
        return self._table(batch)[0]

    async def aprofile(self, batch: RawBatch) -> SourceSchema:
        self._ensure_open()
        table, _ = self._table(batch)
        return self._profile(batch, table)

    def profile(self, batch: RawBatch) -> SourceSchema:
        """Profile a tabular or JSON batch.

        The samples are raw. The data policy applies when Meldr sends them.
        """
        return run_sync(self.aprofile(batch))

    async def _resolve_spec(
        self, batch: RawBatch, table: pl.DataFrame, entity: CanonicalEntity, *, force: bool = False
    ) -> tuple[MappingSpec, bool]:
        """Return `(spec, reused)`.

        When `reused` is True, the spec cost zero LLM calls and no profiling.
        """
        fingerprint = shape_fingerprint(source_columns(table))
        found = await asyncio.to_thread(
            self.store.find_approved, batch.source_id, entity.name, fingerprint
        )
        if found is not None and found.target_version == entity.version:
            return found, True
        prev = await asyncio.to_thread(self.store.latest, batch.source_id, entity.name)
        same_shape = (
            prev is not None
            and prev.source_fingerprint == fingerprint
            and prev.target_version == entity.version
        )
        if prev is not None and same_shape and not force:
            return prev, True
        schema = self._profile(batch, table)
        if table.height == 0:  # nothing to learn from: an empty, unsaved spec and no LLM call
            return build_spec(schema, entity, Proposal()), False
        sample = table.head(self.config.mapping.dryrun_rows)
        policy = self._source_policy(schema.source_id)
        if prev is None or prev.target_version != entity.version or same_shape:
            proposal = await self.engine.propose(schema, entity, sample, source_policy=policy)
            spec = build_spec(
                schema, entity, proposal, parent_version=prev.version if prev else None
            )
        else:
            rec = drift.reconcile(prev, schema, entity)
            if (
                rec.report.breaking_targets
                and self.config.mapping.on_breaking_drift == "block"
                and not force
            ):
                raise DriftBlockedError(
                    f"Source '{schema.source_id}' no longer has fields for the required targets "
                    f"{rec.report.breaking_targets} of {entity.name} "
                    f"(removed: {rec.report.removed}). Restore the fields, map again with "
                    "`meldr map` / Meldr.propose(), or set "
                    "`mapping.on_breaking_drift=quarantine`."
                )
            used = {s for m in rec.carried for s in m.sources}
            proposal = (
                await self.engine.propose(
                    schema,
                    entity,
                    sample,
                    source_policy=policy,
                    targets=rec.retarget,
                    exclude_sources=used,
                )
                if rec.retarget
                else Proposal()
            )
            spec = build_spec(
                schema, entity, proposal, carried=rec.carried, parent_version=prev.version
            )
            safe_log(
                "mapping.drift",
                spec_id=spec.spec_id,
                added=len(rec.report.added),
                removed=len(rec.report.removed),
                retargeted_fields=rec.retarget,
            )
        return await asyncio.to_thread(self.store.save, spec), False

    def _write(
        self, entity: CanonicalEntity, valid: pl.DataFrame, quarantine: pl.DataFrame
    ) -> None:
        """`<Entity>` keeps each record's latest valid version, so bad data does not replace good.

        `<Entity>__quarantine` keeps a failed version only until a newer valid version arrives.
        """
        with otel_span(
            "meldr.sink.write",
            sink=type(self.table_sink).__name__,
            kind="table",
            entity=entity.name,
            rows=valid.height,
            quarantine_rows=quarantine.height,
        ) as span:
            key = list(RECORD_KEY)
            quarantine_table = f"{entity.name}__quarantine"
            if valid.height:
                self.table_sink.write(entity.name, valid, key=key)
                resolved = [*key, SEQUENCE] if SEQUENCE in valid.columns else key
                self.table_sink.delete(quarantine_table, valid.select(resolved))
            quarantine = unsuperseded(quarantine, valid, key)
            if quarantine.height:
                self.table_sink.write(quarantine_table, quarantine, key=key)
            described = (entity.name, entity.version)
            if valid.height:
                with self._describe_lock:
                    if described not in self._described:
                        if isinstance(self.table_sink, Describable):
                            table_doc, column_docs = entity_docs(entity)
                            self.table_sink.describe(
                                entity.name, table=table_doc, columns=column_docs
                            )
                        self._described.add(described)
            set_span_attributes(span, rows_written=valid.height, rows_quarantined=quarantine.height)

    async def apropose(self, batch: RawBatch, target: str) -> MappingSpec:
        self._ensure_open()
        entity = self.registry.get(target)
        table, _ = self._table(batch)
        spec, _ = await self._resolve_spec(batch, table, entity, force=True)
        return spec

    def propose(self, batch: RawBatch, target: str) -> MappingSpec:
        """Make a new proposal for this source shape (the drift block does not apply)."""
        return run_sync(self.apropose(batch, target))

    async def ameld(
        self,
        batch: RawBatch,
        target: str,
        *,
        write: bool = True,
        index: bool | None = None,
        spec: MappingSpec | None = None,
        record_lineage: bool = True,
    ) -> MeldrResult:
        """Map one batch and keep LLM usage local to this call."""
        self._ensure_open()
        with usage_scope() as usage:
            return await self._ameld(
                batch,
                target,
                write=write,
                index=index,
                spec=spec,
                record_lineage=record_lineage,
                usage=usage,
            )

    async def _ameld(
        self,
        batch: RawBatch,
        target: str,
        *,
        write: bool = True,
        index: bool | None = None,
        spec: MappingSpec | None = None,
        record_lineage: bool = True,
        usage: UsageTotals,
    ) -> MeldrResult:
        run = _Run(usage)
        with otel_span(
            "meldr.run",
            run_id=run.id,
            kind="meld",
            source_id=batch.source_id,
            target=target,
        ) as span:
            result = await self._ameld_body(
                batch,
                target,
                write=write,
                index=index,
                spec=spec,
                record_lineage=record_lineage,
                usage=usage,
                run=run,
            )
            set_span_attributes(
                span,
                rows_in=result.report.rows_in,
                rows_out=result.report.rows_out,
                rows_quarantined=result.report.rows_quarantined,
                llm_calls=result.report.llm_calls,
            )
            return result

    async def _ameld_body(
        self,
        batch: RawBatch,
        target: str,
        *,
        write: bool,
        index: bool | None,
        spec: MappingSpec | None,
        record_lineage: bool,
        usage: UsageTotals,
        run: _Run,
    ) -> MeldrResult:
        entity = self.registry.get(target)
        table, warnings = self._table(batch)
        cache_hit = spec is not None
        if spec is None:
            spec, cache_hit = await self._resolve_spec(batch, table, entity)
        if self.config.mapping.on_unapproved == "block" and spec.status != SpecStatus.APPROVED:
            raise MeldrError(
                f"Spec {spec.spec_id} v{spec.version} has {len(spec.pending_fields())} fields "
                "pending review. Run `meldr review` or set mapping.on_unapproved="
                "run_approved_only."
            )
        applied = apply_mapping(
            table,
            spec,
            entity,
            ops=self.ops,
            run_id=run.id,
            on_invalid=self.config.execution.on_invalid,
        )
        warnings.extend(applied.warnings)
        warnings.extend(
            f"new_values:{t}:{len(vals)}" for t, vals in drift.unseen_values(spec, table).items()
        )
        if write:
            await asyncio.to_thread(self._write, entity, applied.data, applied.quarantine)
        do_index = self.config.indexing.records if index is None else index
        if do_index:
            await self.indexer.index_records(entity, applied.data)
        report = await self._finish(
            run,
            record_lineage=record_lineage,
            source_id=batch.source_id,
            target=entity.name,
            rows_in=table.height,
            rows_out=applied.data.height,
            rows_quarantined=applied.quarantine.height,
            fields_mapped=len(spec.approved_fields()),
            fields_pending=len(spec.pending_fields()),
            values_lost=applied.values_lost,
            spec_id=spec.spec_id,
            spec_version=spec.version,
            spec_cache_hit=cache_hit,
            warnings=warnings,
        )
        safe_log(
            "meld.done",
            run_id=run.id,
            source_id=batch.source_id,
            target=entity.name,
            rows_in=report.rows_in,
            rows_out=report.rows_out,
            rows_quarantined=report.rows_quarantined,
            llm_calls=report.llm_calls,
        )
        return MeldrResult(
            data=applied.data,
            quarantine=applied.quarantine,
            spec=spec,
            pending_review=spec.pending_fields(),
            report=report,
            pii_fields=tuple(field.name for field in entity.fields if field.pii),
        )

    def meld(
        self, batch: RawBatch, target: str, *, write: bool = True, index: bool | None = None
    ) -> MeldrResult:
        """Map (or use the approved spec again), transform, validate, write, and index one batch."""
        return run_sync(self.ameld(batch, target, write=write, index=index))

    @staticmethod
    async def _resolve_content(resolver: ContentResolver, link: str, timeout_s: float) -> bytes:
        """Call a caller resolver without a slow event-loop call."""
        if inspect.iscoroutinefunction(resolver):
            value = await asyncio.wait_for(resolver(link), timeout=timeout_s)
        else:
            value = await asyncio.wait_for(asyncio.to_thread(resolver, link), timeout=timeout_s)
            if inspect.isawaitable(value):
                value = await asyncio.wait_for(value, timeout=timeout_s)
        if not isinstance(value, bytes):
            raise TypeError("the resolve function must return bytes")
        return value

    async def _content_records(
        self,
        batch: RawBatch,
        table: pl.DataFrame,
        *,
        documents: Mapping[str, str],
        texts: Mapping[str, str],
        resolve: ContentResolver | None,
    ) -> tuple[dict[str, list[RawRecord]], dict[str, str], dict[str, str]]:
        """Build child records and return target groups, parent failures, and record parents."""
        if documents and resolve is None:
            raise ConfigError(
                "harmonize() must have `resolve` when documents contain links. "
                "The caller must fetch the content."
            )
        columns = set(table.columns)
        declared = [*documents, *texts]
        missing = sorted(set(declared) - columns)
        if missing:
            raise ConfigError(f"Content column(s) not found in the input: {missing}.")
        cfg = self.config.extraction
        parent_ids = [str(index) for index in range(table.height)]
        grouped: dict[str, list[RawRecord]] = {}
        failures: dict[str, str] = {}
        record_parents: dict[str, str] = {}
        resolved: dict[str, bytes | str] = {}
        unique_links: set[str] = set()
        rows = table.to_dicts()

        for index, row in enumerate(rows):
            parent_id = parent_ids[index]
            for column, child_target in documents.items():
                value = row.get(column)
                if value is None or not str(value).strip():
                    continue
                link = str(value).strip()
                if link not in resolved:
                    if link not in unique_links and len(unique_links) >= cfg.max_documents_per_call:
                        resolved[link] = "document_limit_exceeded"
                    else:
                        unique_links.add(link)
                        try:
                            if resolve is None:
                                raise ConfigError(
                                    "harmonize() must have `resolve` when documents contain links."
                                )
                            document_content = await self._resolve_content(
                                resolve, link, cfg.resolve_timeout_s
                            )
                            if len(document_content) > cfg.max_document_bytes:
                                resolved[link] = "document_too_large"
                            else:
                                resolved[link] = document_content
                        except Exception as exc:
                            resolved[link] = f"resolve_failed:{type(exc).__name__}"
                resolved_value = resolved[link]
                if isinstance(resolved_value, str):
                    failures[parent_id] = resolved_value
                    continue
                record_id = f"{parent_id}:{column}"
                record = RawRecord(
                    record_id=record_id,
                    source_id=batch.source_id,
                    modality=Modality.DOCUMENT,
                    content=resolved_value,
                    mime_type=mime_for_path(link),
                    metadata={"filename": Path(link).name or column, "parent_id": parent_id},
                    fetched_at=datetime.now(UTC),
                    content_hash=sha256_bytes(resolved_value),
                )
                grouped.setdefault(child_target, []).append(record)
                record_parents[record_id] = parent_id

            for column, child_target in texts.items():
                value = row.get(column)
                if value is None or not str(value).strip():
                    continue
                text_content = str(value).encode()
                if len(text_content) > cfg.max_document_bytes:
                    failures[parent_id] = "text_too_large"
                    continue
                record_id = f"{parent_id}:{column}"
                record = RawRecord(
                    record_id=record_id,
                    source_id=batch.source_id,
                    modality=Modality.DOCUMENT,
                    content=text_content,
                    mime_type="text/plain",
                    metadata={"filename": f"{column}.txt", "parent_id": parent_id},
                    fetched_at=datetime.now(UTC),
                    content_hash=sha256_bytes(text_content),
                )
                grouped.setdefault(child_target, []).append(record)
                record_parents[record_id] = parent_id
        return grouped, failures, record_parents

    @staticmethod
    def _add_parent_ids(frame: pl.DataFrame, record_parents: Mapping[str, str]) -> pl.DataFrame:
        """Add the source row id to extracted child rows."""
        parents = (
            [record_parents.get(str(value).split("#", 1)[0]) for value in frame["_meldr_record_id"]]
            if "_meldr_record_id" in frame.columns
            else [None] * frame.height
        )
        return frame.with_columns(pl.Series("_meldr_parent_record_id", parents, dtype=pl.String))

    @staticmethod
    def _content_failures(table: pl.DataFrame, failures: Mapping[str, str]) -> pl.DataFrame:
        """Return failed parent rows with safe failure causes."""
        if not failures:
            return pl.DataFrame()
        parent_ids = [str(index) for index in range(table.height)]
        indexes = [index for index, parent_id in enumerate(parent_ids) if parent_id in failures]
        result = table.with_row_index("_meldr_parent_row").filter(
            pl.col("_meldr_parent_row").is_in(indexes)
        )
        return result.with_columns(
            pl.Series(
                "_meldr_parent_record_id",
                [parent_ids[index] for index in indexes],
                dtype=pl.String,
            ),
            pl.Series(
                "_meldr_errors",
                [[failures[parent_ids[index]]] for index in indexes],
                dtype=pl.List(pl.String),
            ),
        ).drop("_meldr_parent_row")

    async def _aharmonize_content(
        self,
        batch: RawBatch,
        target: str,
        *,
        documents: Mapping[str, str],
        texts: Mapping[str, str],
        resolve: ContentResolver | None,
        write: bool,
        index: bool,
        spec: MappingSpec | None,
    ) -> HarmonizeResult:
        """Harmonize parent rows and extract declared child content."""
        if documents and resolve is None:
            raise ConfigError(
                "harmonize() must have `resolve` when documents contain links. "
                "The caller must fetch the content."
            )
        table, _ = self._table(batch)
        parent = await self.ameld(
            batch,
            target,
            write=write,
            index=index,
            spec=spec,
            record_lineage=write,
        )
        groups, failures, record_parents = await self._content_records(
            batch,
            table,
            documents=documents,
            texts=texts,
            resolve=resolve,
        )
        children: dict[str, list[pl.DataFrame]] = {}
        child_quarantines: list[pl.DataFrame] = []
        relation_frames: list[pl.DataFrame] = []
        child_reports: list[RunReport] = []
        for child_target, records in groups.items():
            child_batch = RawBatch(
                source_id=batch.source_id,
                modality=Modality.DOCUMENT,
                records=records,
            )
            child = await self.aextract(
                child_batch,
                child_target,
                write=write,
                index=index,
                record_lineage=write,
            )
            children.setdefault(child_target, []).append(
                self._add_parent_ids(child.data, record_parents)
            )
            child_quarantines.append(self._add_parent_ids(child.quarantine, record_parents))
            if child.relations:
                relation_frames.append(self._relation_table(child.relations))
            child_reports.append(child.report)

        child_data = {
            name: pl.concat(frames, how="diagonal_relaxed") for name, frames in children.items()
        }
        relations = (
            pl.concat(relation_frames, how="diagonal_relaxed")
            if relation_frames
            else pl.DataFrame()
        )
        quarantine_frames = [parent.quarantine, *child_quarantines]
        failed_parents = self._content_failures(table, failures)
        if failed_parents.height:
            quarantine_frames.append(failed_parents)
        quarantine = (
            pl.concat(
                [frame for frame in quarantine_frames if frame.height],
                how="diagonal_relaxed",
            )
            if any(frame.height for frame in quarantine_frames)
            else pl.DataFrame()
        )
        llm_calls = parent.report.llm_calls + sum(report.llm_calls for report in child_reports)
        warnings = [
            *parent.report.warnings,
            *[warning for report in child_reports for warning in report.warnings],
            *[f"content:{parent_id}:{cause}" for parent_id, cause in failures.items()],
        ]
        report = parent.report.model_copy(
            update={
                "rows_quarantined": quarantine.height,
                "llm_calls": llm_calls,
                "llm_cache_hits": parent.report.llm_cache_hits
                + sum(item.llm_cache_hits for item in child_reports),
                "input_tokens": parent.report.input_tokens
                + sum(item.input_tokens for item in child_reports),
                "output_tokens": parent.report.output_tokens
                + sum(item.output_tokens for item in child_reports),
                "warnings": warnings,
            }
        )
        return HarmonizeResult(
            data=parent.data,
            quarantine=quarantine,
            review=review.items(parent.spec),
            children=child_data,
            report=report,
            spec=parent.spec,
            relations=relations,
        )

    async def aharmonize(
        self,
        value: Any,
        target: str | None = None,
        *,
        source_id: str = "source",
        write: bool = False,
        index: bool = False,
        spec: MappingSpec | None = None,
        documents: Mapping[str, str] | None = None,
        texts: Mapping[str, str] | None = None,
        resolve: ContentResolver | None = None,
    ) -> HarmonizeResult:
        """Harmonize caller-owned data and return it without sink writes by default."""
        self._ensure_open()
        batch = coerce_batch(value, source_id=source_id)
        if batch.modality in (Modality.TABULAR, Modality.JSON):
            if target is None:
                raise ConfigError("harmonize() must have `target` for tabular or JSON input.")
            if documents or texts:
                return await self._aharmonize_content(
                    batch,
                    target,
                    documents=documents or {},
                    texts=texts or {},
                    resolve=resolve,
                    write=write,
                    index=index,
                    spec=spec,
                )
            meld_result = await self.ameld(
                batch,
                target,
                write=write,
                index=index,
                spec=spec,
                record_lineage=write,
            )
            return HarmonizeResult(
                data=meld_result.data,
                quarantine=meld_result.quarantine,
                review=review.items(meld_result.spec),
                children={},
                report=meld_result.report,
                spec=meld_result.spec,
            )
        extract_result = await self.aextract(
            batch,
            target,
            write=write,
            index=index,
            record_lineage=write,
        )
        return HarmonizeResult(
            data=extract_result.data,
            quarantine=extract_result.quarantine,
            review=[],
            children={},
            report=extract_result.report,
            spec=None,
            relations=self._relation_table(extract_result.relations),
        )

    async def aapply_review(
        self,
        spec: MappingSpec,
        decisions: Sequence[ReviewDecision | Mapping[str, Any]],
        *,
        user: str,
    ) -> MappingSpec:
        """Apply caller decisions and save the next mapping spec version."""
        self._ensure_open()
        changed = review.apply_review(spec, decisions, user=user, ops=self.ops)
        return await asyncio.to_thread(
            self.store.save,
            changed,
            expected_latest=spec.version,
            actor=user,
            action="review",
        )

    def apply_review(
        self,
        spec: MappingSpec,
        decisions: Sequence[ReviewDecision | Mapping[str, Any]],
        *,
        user: str,
    ) -> MappingSpec:
        """Synchronously apply caller decisions and save the next spec version."""
        return run_sync(self.aapply_review(spec, decisions, user=user))

    def harmonize(
        self,
        value: Any,
        target: str | None = None,
        *,
        source_id: str = "source",
        write: bool = False,
        index: bool = False,
        spec: MappingSpec | None = None,
        documents: Mapping[str, str] | None = None,
        texts: Mapping[str, str] | None = None,
        resolve: ContentResolver | None = None,
    ) -> HarmonizeResult:
        """Synchronously harmonize caller-owned data and return the result."""
        return run_sync(
            self.aharmonize(
                value,
                target,
                source_id=source_id,
                write=write,
                index=index,
                spec=spec,
                documents=documents,
                texts=texts,
                resolve=resolve,
            )
        )

    async def arun(self, source: str, target: str, **params: Any) -> list[MeldrResult]:
        """Meld every batch of a configured source, where the first batch decides the spec."""
        self._ensure_open()
        results: list[MeldrResult] = []
        spec: MappingSpec | None = None
        connector = await asyncio.to_thread(self.connectors.__getitem__, source)
        batches = iter(await asyncio.to_thread(connector.pull, **params))
        while (batch := await asyncio.to_thread(next, batches, None)) is not None:
            res = await self.ameld(batch, target, spec=spec)
            if spec is None and res.report.rows_in:  # an empty first batch must not fix the spec
                spec = res.spec
            results.append(res)
        return results

    def run(self, source: str, target: str, **params: Any) -> list[MeldrResult]:
        return run_sync(self.arun(source, target, **params))

    @staticmethod
    def _relation_frames(
        entity: CanonicalEntity, records: Sequence[ExtractedRecord]
    ) -> tuple[list[Relation], pl.DataFrame]:
        """Keep permitted relations and quarantine relations with unknown predicates."""
        valid: list[Relation] = []
        invalid: list[dict[str, Any]] = []
        allowed = set(entity.relations)
        for record in records:
            for relation in record.relations:
                row = {
                    "subject_id": relation.subject_id,
                    "predicate": relation.predicate,
                    "object_id": relation.object_id,
                    "evidence_segment_id": (
                        relation.evidence.segment_id if relation.evidence else None
                    ),
                    "evidence_quote": relation.evidence.quote if relation.evidence else None,
                    "evidence_verified": (
                        relation.evidence.verified if relation.evidence else False
                    ),
                    "confidence": relation.confidence,
                }
                if relation.predicate in allowed:
                    valid.append(relation)
                else:
                    invalid.append(
                        {
                            **row,
                            "_meldr_errors": [
                                f"relation_predicate_not_allowed:{relation.predicate}"
                            ],
                        }
                    )
        return valid, pl.DataFrame(invalid)

    @staticmethod
    def _relation_table(relations: Sequence[Relation]) -> pl.DataFrame:
        """Convert permitted relations to caller-owned rows."""
        return pl.DataFrame(
            [
                {
                    "subject_id": relation.subject_id,
                    "predicate": relation.predicate,
                    "object_id": relation.object_id,
                    "evidence_segment_id": (
                        relation.evidence.segment_id if relation.evidence else None
                    ),
                    "evidence_quote": relation.evidence.quote if relation.evidence else None,
                    "evidence_verified": (
                        relation.evidence.verified if relation.evidence else False
                    ),
                    "confidence": relation.confidence,
                }
                for relation in relations
            ]
        )

    async def _segments(
        self, batch: RawBatch, warnings: list[str]
    ) -> dict[str, list[ContentSegment]]:
        from meldr.extraction.documents import segment_record
        from meldr.extraction.media import segment_image, segment_media

        out: dict[str, list[ContentSegment]] = {}
        ex = self.config.extraction
        for rec in batch.records:
            if rec.modality in (Modality.VIDEO, Modality.AUDIO):
                if not rec.uri:
                    warnings.append(f"media_without_uri:{rec.record_id}")
                    continue
                segs, w = await segment_media(
                    Path(rec.uri),
                    record_id=rec.record_id,
                    source_id=rec.source_id,
                    cfg=self.config.media,
                    transcriber=self.transcriber,
                    vision=self.vision_llm,
                    source_policy=self._source_policy(batch.source_id),
                )
            elif rec.modality == Modality.IMAGE:
                segs, w = await segment_image(
                    rec, self.vision_llm, source_policy=self._source_policy(batch.source_id)
                )
            else:
                segs, w = segment_record(
                    rec, target_tokens=ex.chunk_tokens, overlap_tokens=ex.overlap_tokens
                )
            warnings.extend(w)
            out[rec.record_id] = segs
        return out

    async def aextract(
        self,
        batch: RawBatch,
        target: str | None = None,
        *,
        write: bool = True,
        index: bool = True,
        record_lineage: bool = True,
    ) -> ExtractResult:
        """Extract one batch and keep LLM usage local to this call."""
        self._ensure_open()
        with usage_scope() as usage:
            return await self._aextract(
                batch,
                target,
                write=write,
                index=index,
                record_lineage=record_lineage,
                usage=usage,
            )

    async def _aextract(
        self,
        batch: RawBatch,
        target: str | None = None,
        *,
        write: bool = True,
        index: bool = True,
        record_lineage: bool = True,
        usage: UsageTotals,
    ) -> ExtractResult:
        run = _Run(usage)
        with otel_span(
            "meldr.run",
            run_id=run.id,
            kind="extract",
            source_id=batch.source_id,
            target=target,
        ) as span:
            result = await self._aextract_body(
                batch,
                target,
                write=write,
                index=index,
                record_lineage=record_lineage,
                usage=usage,
                run=run,
            )
            set_span_attributes(
                span,
                rows_in=result.report.rows_in,
                rows_out=result.report.rows_out,
                rows_quarantined=result.report.rows_quarantined,
                llm_calls=result.report.llm_calls,
            )
            return result

    async def _aextract_body(
        self,
        batch: RawBatch,
        target: str | None,
        *,
        write: bool,
        index: bool,
        record_lineage: bool,
        usage: UsageTotals,
        run: _Run,
    ) -> ExtractResult:
        warnings: list[str] = []
        entity = self.registry.get(target) if target else None
        if entity is not None and self.extraction_llm is None:
            raise ConfigError(
                "extract() with a target must have an LLM. Set llm.routing.extraction (use a local "
                "provider for document content)."
            )
        by_record = await self._segments(batch, warnings)
        segments = [s for segs in by_record.values() for s in segs]
        records: list[ExtractedRecord] = []
        if entity is not None and self.extraction_llm is not None:
            extractor = Extractor(
                self.extraction_llm,
                cache=self._extraction_cache,
                single_pass_max_tokens=self.config.extraction.single_pass_max_tokens,
            )
            hashes = {r.record_id: r.content_hash for r in batch.records}
            warnings.extend(f"no_segments:{rid}" for rid, segs in by_record.items() if not segs)
            policy = self._source_policy(batch.source_id)
            extracted = await asyncio.gather(
                *(
                    extractor.extract(
                        entity, segs, record_id=rid, content_hash=hashes[rid], source_policy=policy
                    )
                    for rid, segs in by_record.items()
                    if segs
                )
            )
            records = [record for batch_records in extracted for record in batch_records]
            warnings.extend(f"{r.record_id}:{w}" for r in records for w in r.warnings)
        data = record_quarantine = relation_quarantine = pl.DataFrame()
        relations: list[Relation] = []
        if entity is not None and records:
            checked = extracted_rows.to_rows(
                entity,
                records,
                self.ops,
                Lineage(batch.source_id, extracted_rows.spec_id(entity), 0, run.id),
                on_invalid=self.config.execution.on_invalid,
            )
            data, record_quarantine = checked.valid, checked.quarantine
            relations, relation_quarantine = self._relation_frames(entity, records)
            if relation_quarantine.height:
                warnings.append(f"relation_quarantine:{relation_quarantine.height}")
        if write and entity is not None:
            await asyncio.to_thread(self._write, entity, data, record_quarantine)
        if index:
            links: dict[str, list[str]] = {}
            for rec in records:
                for ev in rec.evidence.values():
                    ids = links.setdefault(ev.segment_id, [])
                    if rec.record_id not in ids:
                        ids.append(rec.record_id)
                for relation in rec.relations:
                    if relation.evidence is not None:
                        ids = links.setdefault(relation.evidence.segment_id, [])
                        if rec.record_id not in ids:
                            ids.append(rec.record_id)
            await self.indexer.index_segments(segments, links)
            if entity is not None:
                await self.indexer.index_records(entity, data)
        quarantine_frames = [record_quarantine, relation_quarantine]
        nonempty_quarantine = [frame for frame in quarantine_frames if frame.height]
        quarantine = (
            pl.concat(nonempty_quarantine, how="diagonal_relaxed")
            if any(frame.height for frame in quarantine_frames)
            else pl.DataFrame()
        )
        report = await self._finish(
            run,
            record_lineage=record_lineage,
            source_id=batch.source_id,
            target=entity.name if entity else None,
            rows_in=len(batch.records),
            rows_out=data.height,
            rows_quarantined=quarantine.height,
            fields_mapped=len(entity.fields) if entity else 0,
            spec_id=extracted_rows.spec_id(entity) if entity else None,
            warnings=warnings,
        )
        return ExtractResult(
            records=records,
            data=data,
            quarantine=quarantine,
            segments=segments,
            report=report,
            relations=relations,
        )

    def extract(
        self,
        batch: RawBatch,
        target: str | None = None,
        *,
        write: bool = True,
        index: bool = True,
    ) -> ExtractResult:
        """Segment documents/media, extract `target` records with evidence, and index."""
        return run_sync(self.aextract(batch, target, write=write, index=index))

    async def asearch(
        self,
        query: str,
        *,
        entity: str | None = None,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[SearchHit]:
        self._ensure_open()
        if entity is not None:
            collections = [records_collection(self.registry.get(entity).name)]
        else:
            collections = [records_collection(n) for n, _ in self.registry.list()] + [SEGMENTS]
        return await self.indexer.search(
            query, collections=list(dict.fromkeys(collections)), filters=filters, k=k
        )

    def search(
        self,
        query: str,
        *,
        entity: str | None = None,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[SearchHit]:
        """Vector search of records (and segments when `entity` is None), with lineage."""
        return run_sync(self.asearch(query, entity=entity, filters=filters, k=k))

    def explain(
        self, entity: str, record_id: str, field: str, source_id: str | None = None
    ) -> Explanation:
        """Which source paths, steps, spec version, and run made one value.

        `record_id` is the `_meldr_record_id` or, for single-column keys, the primary key value.
        """
        ent = self.registry.get(entity)
        return explain_value(
            self.table_sink,
            self.store,
            self.lineage,
            ent.name,
            record_id,
            field,
            source_id=source_id,
            primary_key=ent.primary_key,
        )

    async def adiscover_schema(self, batches: Sequence[RawBatch], name: str) -> CanonicalEntity:
        self._ensure_open()
        schemas = []
        for b in batches:
            table, _ = self._table(b)
            schemas.append(self._profile(b, table))
        return discover(schemas, name)

    def discover_schema(self, batches: Sequence[RawBatch], name: str) -> CanonicalEntity:
        """Draft an entity from source batches.

        Review the YAML: this method does not register it.
        """
        return run_sync(self.adiscover_schema(batches, name))

    async def aresolve(
        self,
        entity: str,
        *,
        fields: Iterable[str],
        block_on: Iterable[str],
        hi: float = 0.92,
        lo: float = 0.75,
        write: bool = True,
    ) -> Resolution:
        self._ensure_open()
        ent = self.registry.get(entity)
        df = await asyncio.to_thread(self.table_sink.read, ent.name)
        res = await resolve(
            df,
            entity=ent.name,
            fields=list(fields),
            block_on=list(block_on),
            hi=hi,
            lo=lo,
            gateway=self.llm,
        )
        if write and res.links.height:
            await asyncio.to_thread(
                self.table_sink.write, f"{ent.name}__entity_links", res.links, key=["a", "b"]
            )
        return res

    def resolve(
        self,
        entity: str,
        *,
        fields: Iterable[str],
        block_on: Iterable[str],
        hi: float = 0.92,
        lo: float = 0.75,
        write: bool = True,
    ) -> Resolution:
        """Cross-source entity resolution on the harmonized table (P2)."""
        return run_sync(
            self.aresolve(entity, fields=fields, block_on=block_on, hi=hi, lo=lo, write=write)
        )
