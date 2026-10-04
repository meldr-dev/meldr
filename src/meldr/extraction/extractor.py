"""Schema-constrained extraction with verified evidence (§4.14)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import DataType
from meldr.core.segments import ContentSegment, Evidence, ExtractedRecord, Relation
from meldr.core.types import sha256_json
from meldr.extraction.chunker import estimate_tokens
from meldr.llm.base import Message
from meldr.llm.cache import ResponseCache
from meldr.llm.gateway import LLMGateway
from meldr.llm.policy import DataClass, SourcePolicy
from meldr.llm.prompts import load as load_prompt

PROMPT_VERSION = "extraction-v2"
MANY_PROMPT_VERSION = "extraction-many-v1"
UNVERIFIED_PENALTY = 0.5


class EvidenceIn(BaseModel):
    segment_id: str
    quote: str = Field(max_length=300)


class RelationIn(BaseModel):
    subject_id: str
    predicate: str
    object_id: str
    evidence: EvidenceIn | None = None
    confidence: float = Field(ge=0.0, le=1.0)


_PY: dict[DataType, Any] = {
    DataType.STRING: str,
    DataType.INTEGER: int,
    DataType.FLOAT: float,
    DataType.DECIMAL: float,
    DataType.BOOLEAN: bool,
    DataType.DATE: str,
    DataType.DATETIME: str,
}


def _field_type(f: CanonicalField) -> Any:
    if f.vocabulary:
        return Literal[tuple(f.vocabulary)]
    return _PY.get(f.dtype, str)


def response_model_for(entity: CanonicalEntity) -> type[BaseModel]:
    """Build a response model for one or many records."""
    fields: dict[str, Any] = {
        f.name: (_field_type(f) | None, Field(default=None, description=f.description))
        for f in entity.fields
    }
    values = create_model(f"{entity.name}Values", __config__=ConfigDict(extra="forbid"), **fields)
    item = create_model(
        f"{entity.name}ExtractionItem",
        values=(values, ...),
        evidence=(dict[str, EvidenceIn], Field(default_factory=dict)),
        confidence=(dict[str, float], Field(default_factory=dict)),
    )
    if entity.extraction.cardinality == "many":
        item_type = cast(Any, item)
        return create_model(
            f"{entity.name}ManyExtraction",
            items=(list[item_type], Field(default_factory=list)),  # type: ignore[valid-type]
            relations=(list[RelationIn], Field(default_factory=list)),
        )
    return create_model(
        f"{entity.name}Extraction",
        values=(values, ...),
        evidence=(dict[str, EvidenceIn], Field(default_factory=dict)),
        confidence=(dict[str, float], Field(default_factory=dict)),
        relations=(list[RelationIn], Field(default_factory=list)),
    )


def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def verify(quote: str, segment: ContentSegment | None) -> bool:
    """The quote must appear in the cited segment after whitespace/case normalization."""
    return bool(segment is not None and quote.strip() and _norm(quote) in _norm(segment.text))


def _payload(entity: CanonicalEntity, segments: Sequence[ContentSegment]) -> dict[str, Any]:
    return {
        "task": "extraction",
        "entity": {
            "name": entity.name,
            "description": entity.description,
            "fields": [
                {
                    k: v
                    for k, v in {
                        "name": f.name,
                        "dtype": str(f.dtype),
                        "description": f.description,
                        "vocabulary": f.vocabulary,
                        "unit": f.unit,
                        "synonyms": f.synonyms or None,
                    }.items()
                    if v is not None
                }
                for f in entity.fields
            ],
        },
        "segments": [
            {"segment_id": s.segment_id, "text": f"[{s.segment_id}] {s.text}"} for s in segments
        ],
    }


class Extractor:
    """Extracts one record for each document and entity."""

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        cache: ResponseCache | None = None,
        single_pass_max_tokens: int = 12_000,
    ) -> None:
        self.gateway = gateway
        self.cache = cache
        self.single_pass_max_tokens = single_pass_max_tokens
        self._models: dict[tuple[str, str], type[BaseModel]] = {}
        self._inflight: dict[str, asyncio.Task[list[ExtractedRecord]]] = {}

    def _cache_key(
        self, segments: Sequence[ContentSegment], content_hash: str, entity: CanonicalEntity
    ) -> str:
        """One key for each content value and entity."""
        prompt_version = (
            MANY_PROMPT_VERSION if entity.extraction.cardinality == "many" else PROMPT_VERSION
        )
        return sha256_json(
            [
                content_hash,
                entity.name,
                entity.version,
                prompt_version,
                self.gateway.provider.model,
            ]
        )

    @staticmethod
    def _rebind_cached(
        record: ExtractedRecord,
        segments: Sequence[ContentSegment],
        record_id: str,
        index: int = 0,
        many: bool = False,
    ) -> ExtractedRecord:
        """Attach cached evidence to the segment ids of the current record."""
        current = {
            str(segment_index): segment.segment_id for segment_index, segment in enumerate(segments)
        }
        evidence = {}
        for name, item in record.evidence.items():
            evidence_index = item.segment_id.rsplit(":", 1)[-1]
            evidence[name] = item.model_copy(
                update={"segment_id": current.get(evidence_index, item.segment_id)}
            )
        relations = []
        for relation in record.relations:
            if relation.evidence is None:
                relations.append(relation)
                continue
            evidence_index = relation.evidence.segment_id.rsplit(":", 1)[-1]
            relations.append(
                relation.model_copy(
                    update={
                        "evidence": relation.evidence.model_copy(
                            update={
                                "segment_id": current.get(
                                    evidence_index, relation.evidence.segment_id
                                )
                            }
                        )
                    }
                )
            )
        suffix = f"#{index}" if many else ""
        return record.model_copy(
            update={
                "record_id": f"{record_id}{suffix}",
                "evidence": evidence,
                "relations": relations,
            }
        )

    async def _call(
        self,
        entity: CanonicalEntity,
        segments: Sequence[ContentSegment],
        source_policy: SourcePolicy | None,
    ) -> tuple[
        list[tuple[dict[str, Any], dict[str, EvidenceIn], dict[str, float]]],
        list[RelationIn],
    ]:
        key = (entity.name, entity.version)
        if key not in self._models:
            self._models[key] = response_model_for(entity)
        model = self._models[key]
        prompt_version = (
            MANY_PROMPT_VERSION if entity.extraction.cardinality == "many" else PROMPT_VERSION
        )
        prompt_name = (
            "extraction_many_v1" if entity.extraction.cardinality == "many" else "extraction_v1"
        )
        res = await self.gateway.generate(
            [Message(role="user", content=json.dumps(_payload(entity, segments)))],
            response_model=model,
            contains={DataClass.CONTENT},
            purpose="extraction",
            prompt_version=prompt_version,
            system=load_prompt(prompt_name),
            source_policy=source_policy,
        )
        parsed: Any = res.parsed
        if entity.extraction.cardinality == "many":
            payloads = [
                (item.values.model_dump(), dict(item.evidence), dict(item.confidence))
                for item in parsed.items
            ]
            return payloads, list(parsed.relations)
        return (
            [(parsed.values.model_dump(), dict(parsed.evidence), dict(parsed.confidence))],
            list(parsed.relations),
        )

    async def extract(
        self,
        entity: CanonicalEntity,
        segments: Sequence[ContentSegment],
        *,
        record_id: str,
        content_hash: str,
        source_policy: SourcePolicy | None = None,
    ) -> list[ExtractedRecord]:
        """Extract one or many records, with one pass for each chunk when necessary."""
        key = self._cache_key(segments, content_hash, entity)
        if self.cache is not None:
            hit = await asyncio.to_thread(self.cache.get, key)
            if hit is not None:
                cached = json.loads(hit)
                records = (
                    [ExtractedRecord.model_validate(item) for item in cached]
                    if isinstance(cached, list)
                    else [ExtractedRecord.model_validate(cached)]
                )
                return [
                    self._rebind_cached(
                        record,
                        segments,
                        record_id,
                        index,
                        many=entity.extraction.cardinality == "many",
                    )
                    for index, record in enumerate(records)
                ]
        task = self._inflight.get(key)
        owner = task is None
        if task is None:
            task = asyncio.create_task(
                self._extract_uncached(
                    entity,
                    segments,
                    record_id=record_id,
                    content_hash=content_hash,
                    source_policy=source_policy,
                    key=key,
                )
            )
            self._inflight[key] = task
        try:
            records = await asyncio.shield(task)
        finally:
            if owner and task.done():
                self._inflight.pop(key, None)
        return [
            self._rebind_cached(
                record,
                segments,
                record_id,
                index,
                many=entity.extraction.cardinality == "many",
            )
            for index, record in enumerate(records)
        ]

    async def _extract_uncached(
        self,
        entity: CanonicalEntity,
        segments: Sequence[ContentSegment],
        *,
        record_id: str,
        content_hash: str,
        source_policy: SourcePolicy | None,
        key: str,
    ) -> list[ExtractedRecord]:
        """Extract one or many records after a cache miss."""
        total = sum(estimate_tokens(s.text) for s in segments)
        groups = (
            [list(segments)] if total <= self.single_pass_max_tokens else [[s] for s in segments]
        )
        by_id = {s.segment_id: s for s in segments}
        answers = await asyncio.gather(*(self._call(entity, g, source_policy) for g in groups))
        payload_answers = [payloads for payloads, _ in answers]
        relation_inputs = [relation for _, relations in answers for relation in relations]
        if entity.extraction.cardinality == "many":
            payloads: list[tuple[dict[str, Any], dict[str, EvidenceIn], dict[str, float]]] = []
            seen: set[str] = set()
            for answer in payload_answers:
                for values, evidence, confidence in answer:
                    marker = sha256_json(
                        [
                            values,
                            sorted(
                                (name, item.segment_id, item.quote)
                                for name, item in evidence.items()
                            ),
                        ]
                    )
                    if marker not in seen:
                        seen.add(marker)
                        payloads.append((values, evidence, confidence))
            records = [
                self._make_record(
                    entity,
                    values,
                    evidence,
                    confidence,
                    by_id,
                    record_id,
                )
                for values, evidence, confidence in payloads
            ]
        else:
            best: dict[str, tuple[Any, EvidenceIn | None, float]] = {}
            warnings: list[str] = []
            for answer in payload_answers:
                for values, evidence, confidence in answer:
                    for name, value in values.items():
                        if value is None:
                            continue
                        conf = float(confidence.get(name, 0.5))
                        prev = best.get(name)
                        if prev is not None and prev[0] != value:
                            warnings.append(f"conflict:{name}")
                        if prev is None or conf > prev[2]:
                            best[name] = (value, evidence.get(name), conf)
            records = [
                self._make_record(
                    entity,
                    {name: value for name, (value, _, _) in best.items()},
                    {
                        name: evidence
                        for name, (_, evidence, _) in best.items()
                        if evidence is not None
                    },
                    {name: confidence for name, (_, _, confidence) in best.items()},
                    by_id,
                    record_id,
                    warnings=sorted(set(warnings)),
                )
            ]
        relations = self._make_relations(relation_inputs, by_id)
        if records:
            records[0] = records[0].model_copy(update={"relations": relations})
        if self.cache is not None:
            payload = json.dumps([record.model_dump(mode="json") for record in records])
            await asyncio.to_thread(self.cache.set, key, payload)
        return records

    @staticmethod
    def _make_relations(
        values: Sequence[RelationIn], by_id: dict[str, ContentSegment]
    ) -> list[Relation]:
        """Check relation evidence and keep the model confidence."""
        result: list[Relation] = []
        for value in values:
            confidence = value.confidence
            evidence = None
            if value.evidence is not None:
                verified = verify(value.evidence.quote, by_id.get(value.evidence.segment_id))
                evidence = Evidence(
                    segment_id=value.evidence.segment_id,
                    quote=value.evidence.quote,
                    verified=verified,
                )
                if not verified:
                    confidence *= UNVERIFIED_PENALTY
            else:
                confidence *= UNVERIFIED_PENALTY
            result.append(
                Relation(
                    subject_id=value.subject_id,
                    predicate=value.predicate,
                    object_id=value.object_id,
                    evidence=evidence,
                    confidence=round(confidence, 4),
                )
            )
        return result

    @staticmethod
    def _make_record(
        entity: CanonicalEntity,
        values: dict[str, Any],
        evidence: dict[str, EvidenceIn],
        confidence: dict[str, float],
        by_id: dict[str, ContentSegment],
        record_id: str,
        *,
        warnings: list[str] | None = None,
    ) -> ExtractedRecord:
        """Build one extracted record and check its evidence."""
        record = ExtractedRecord(
            entity=entity.name,
            entity_version=entity.version,
            record_id=record_id,
            values={f.name: None for f in entity.fields},
            evidence={},
            confidence={},
            warnings=list(warnings or []),
        )
        for name, value in values.items():
            if value is None or name not in record.values:
                continue
            conf = float(confidence.get(name, 0.5))
            record.values[name] = value
            ev = evidence.get(name)
            if ev is None:
                conf *= UNVERIFIED_PENALTY
                record.warnings.append(f"missing_evidence:{name}")
            else:
                ok = verify(ev.quote, by_id.get(ev.segment_id))
                record.evidence[name] = Evidence(
                    segment_id=ev.segment_id, quote=ev.quote, verified=ok
                )
                if not ok:
                    conf *= UNVERIFIED_PENALTY
                    record.warnings.append(f"unverified_evidence:{name}")
            record.confidence[name] = round(conf, 4)
        return record
