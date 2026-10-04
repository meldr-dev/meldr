"""LLM adjudication of candidate mappings (§4.7)."""

from __future__ import annotations

import asyncio
import json
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import type_compat
from meldr.core.errors import SpecValidationError
from meldr.core.mapping import FieldMapping, Origin, TransformStep
from meldr.core.schema import SourceSchema
from meldr.llm.base import Message, Usage
from meldr.llm.gateway import LLMGateway
from meldr.llm.policy import DataClass, SourcePolicy
from meldr.llm.prompts import load as load_prompt
from meldr.mapping.candidates import Candidate
from meldr.profiling.masking import apply_policy, samples_class
from meldr.transforms.registry import TransformRegistry

PROMPT_VERSION = "mapping-v2"
FULL_PROFILE_LIMIT = 300


class LLMFieldDecision(BaseModel):
    target: str
    sources: list[str]
    steps: list[TransformStep] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(default="", max_length=300)


class LLMMappingResponse(BaseModel):
    decisions: list[LLMFieldDecision]


@dataclass
class Adjudication:
    """target -> mapping (None = the model chose no mapping)."""

    decisions: dict[str, FieldMapping | None] = field(default_factory=dict)
    invalid: set[str] = field(default_factory=set)
    usage: Usage = field(default_factory=Usage)
    calls: int = 0
    model: str | None = None


def target_payload(t: CanonicalField) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": t.name,
        "dtype": str(t.dtype),
        "description": t.description,
        "required": t.required,
    }
    if t.unit:
        d["unit"] = t.unit
    if t.vocabulary:
        d["vocabulary"] = t.vocabulary
    if t.synonyms:
        d["synonyms"] = t.synonyms
    if t.examples:
        d["examples"] = t.examples[:5]
    return d


def build_payload(
    schema: SourceSchema,
    entity: CanonicalEntity,
    targets: Sequence[CanonicalField],
    candidates: dict[str, list[Candidate]],
    ops: TransformRegistry,
) -> dict[str, Any]:
    """The user-message payload. `schema` must already be policy-filtered."""
    cand_paths = {c.path for t in targets for c in candidates.get(t.name, [])}
    full = schema.fields
    others: list[str] = []
    if len(schema.fields) > FULL_PROFILE_LIMIT:
        full = [f for f in schema.fields if f.path in cand_paths]
        others = [f.path for f in schema.fields if f.path not in cand_paths]
    payload: dict[str, Any] = {
        "task": "mapping",
        "target_entity": {"name": entity.name, "description": entity.description},
        "target_fields": [target_payload(t) for t in targets],
        "source_fields": [
            {
                "path": f.path,
                "dtype": str(f.dtype),
                "inferred_dtype": str(f.inferred_dtype),
                "patterns": f.patterns,
                "hints": f.hints,
                "samples": f.samples,
                "description": f.description,
            }
            for f in full
        ],
        "candidates": {
            t.name: [
                {"path": c.path, "score": round(c.score, 3)} for c in candidates.get(t.name, [])
            ]
            for t in targets
        },
        "ops": ops.describe_for_llm(),
    }
    if schema.entity_hint:
        payload["source_name"] = schema.entity_hint  # table, endpoint, or file the fields came from
    if others:
        payload["other_source_paths"] = others
    return payload


def validate_decision(
    d: LLMFieldDecision,
    chunk: set[str],
    schema: SourceSchema,
    entity: CanonicalEntity,
    ops: TransformRegistry,
) -> list[str]:
    """Semantic checks: target, sources, ops/args, arity, castability."""
    if d.target not in chunk:
        return [f"{d.target}: not a target field in this request"]
    if not d.sources:
        return []
    paths = set(schema.paths())
    errors = [f"{d.target}: unknown source path '{s}'" for s in d.sources if s not in paths]
    if errors:
        return errors
    mapping = to_mapping(d)
    try:
        ops.validate(mapping)
    except SpecValidationError as e:
        return [f"{d.target}: {e}"]
    src = schema.field(d.sources[0]).dtype
    out = ops.output_dtype(mapping, src)
    if type_compat(out, entity.field(d.target).dtype) == 0.0:
        return [f"{d.target}: output dtype {out} cannot be cast to {entity.field(d.target).dtype}"]
    return []


def to_mapping(d: LLMFieldDecision) -> FieldMapping:
    return FieldMapping(
        target=d.target,
        sources=d.sources,
        steps=d.steps,
        confidence=d.confidence,
        rationale=d.rationale,
        origin=Origin.LLM,
    )


def _chunks(items: Sequence[CanonicalField], size: int) -> Iterable[Sequence[CanonicalField]]:
    for i in range(0, len(items), max(size, 1)):
        yield items[i : i + size]


async def _decide_chunk(
    gateway: LLMGateway,
    payload: dict[str, Any],
    names: set[str],
    schema: SourceSchema,
    entity: CanonicalEntity,
    ops: TransformRegistry,
    result: Adjudication,
    *,
    contains: set[DataClass],
    source_policy: SourcePolicy | None,
    repair: bool,
) -> dict[str, LLMFieldDecision]:
    """One LLM call for a chunk, plus one semantic repair round.

    It returns the decisions that pass `_check`.
    """
    system = load_prompt("mapping_v2")
    messages = [Message(role="user", content=json.dumps(payload, default=str))]
    res = await gateway.generate(
        messages,
        response_model=LLMMappingResponse,
        contains=contains,
        purpose="mapping",
        prompt_version=PROMPT_VERSION,
        system=system,
        source_policy=source_policy,
    )
    result.calls += res.attempts
    result.usage = result.usage + res.usage
    result.model = res.model
    decisions, errors = _check(res.parsed, names, schema, entity, ops)
    if errors and repair:
        followup = [
            *messages,
            Message(role="assistant", content=res.text),
            Message(
                role="user",
                content="Some decisions are invalid:\n- "
                + "\n- ".join(errors)
                + "\nReturn the complete corrected JSON with one decision per target.",
            ),
        ]
        res2 = await gateway.generate(
            followup,
            response_model=LLMMappingResponse,
            contains=contains,
            purpose="mapping_repair",
            prompt_version=PROMPT_VERSION,
            system=system,
            source_policy=source_policy,
        )
        result.calls += res2.attempts
        result.usage = result.usage + res2.usage
        fixed, _ = _check(res2.parsed, names, schema, entity, ops)
        decisions.update(fixed)
    return decisions


def _ballot(d: LLMFieldDecision) -> tuple[tuple[str, ...], str]:
    steps = json.dumps([s.model_dump() for s in d.steps], sort_keys=True)
    return tuple(d.sources), steps


def vote(
    samples: Sequence[dict[str, LLMFieldDecision]], names: Iterable[str]
) -> dict[str, LLMFieldDecision]:
    """Majority vote for each target, with confidence scaled by the share of samples that agree."""
    out: dict[str, LLMFieldDecision] = {}
    for t in names:
        ballots = [s[t] for s in samples if t in s]
        if not ballots:
            continue
        counts = Counter(_ballot(d) for d in ballots)
        top = max(counts.values())
        winner_key = next(_ballot(d) for d in ballots if counts[_ballot(d)] == top)
        winners = [d for d in ballots if _ballot(d) == winner_key]
        agreement = top / len(samples)
        confidence = statistics.fmean(d.confidence for d in winners) * agreement
        note = f" [agreement {top}/{len(samples)}]"
        out[t] = winners[0].model_copy(
            update={
                "confidence": round(confidence, 4),
                "rationale": winners[0].rationale[: 300 - len(note)] + note,
            }
        )
    return out


async def adjudicate(
    gateway: LLMGateway,
    schema: SourceSchema,
    entity: CanonicalEntity,
    candidates: dict[str, list[Candidate]],
    ops: TransformRegistry,
    *,
    targets: Sequence[str] | None = None,
    source_policy: SourcePolicy | None = None,
    max_fields_per_call: int = 40,
    pii_paths: Iterable[str] = (),
    samples: int = 1,
) -> Adjudication:
    """Get the LLM decision for each target.

    With `samples > 1`, the function sends each chunk that many times, with the source fields in
    a different (fixed) order each time. A majority vote picks the answers (self-consistency).
    """
    policy = gateway.effective_policy(source_policy)
    visible = apply_policy(schema, policy, pii_paths=pii_paths)
    contains = {DataClass.SCHEMA}
    sc = samples_class(policy)
    if sc is not None and any(f.samples or f.hints.get("top_values") for f in visible.fields):
        contains.add(sc)
    wanted = [t for t in entity.fields if targets is None or t.name in targets]
    result = Adjudication()

    def variant(payload: dict[str, Any], i: int) -> dict[str, Any]:
        if not i:
            return payload
        fields = list(payload["source_fields"])
        random.Random(i).shuffle(fields)  # noqa: S311 - deterministic order, not crypto
        return {**payload, "source_fields": fields}

    async def decide(chunk: Sequence[CanonicalField]) -> tuple[set[str], list[Any]]:
        names = {t.name for t in chunk}
        payload = build_payload(visible, entity, chunk, candidates, ops)
        runs = await asyncio.gather(
            *(
                _decide_chunk(
                    gateway,
                    variant(payload, i),
                    names,
                    schema,
                    entity,
                    ops,
                    result,
                    contains=contains,
                    source_policy=source_policy,
                    repair=samples == 1,
                )
                for i in range(samples)
            )
        )
        return names, list(runs)

    chunks = list(_chunks(wanted, max_fields_per_call))
    for names, runs in await asyncio.gather(*(decide(c) for c in chunks)):
        decisions = runs[0] if samples == 1 else vote(runs, names)
        for t in names:
            if t in decisions:
                d = decisions[t]
                result.decisions[t] = to_mapping(d) if d.sources else None
            else:
                result.invalid.add(t)
    return result


def _check(
    resp: LLMMappingResponse,
    names: set[str],
    schema: SourceSchema,
    entity: CanonicalEntity,
    ops: TransformRegistry,
) -> tuple[dict[str, LLMFieldDecision], list[str]]:
    ok: dict[str, LLMFieldDecision] = {}
    errors: list[str] = []
    for d in resp.decisions:
        errs = validate_decision(d, names, schema, entity, ops)
        if errs:
            errors.extend(errs)
        elif d.target in ok:
            errors.append(f"{d.target}: duplicate decision")
        else:
            ok[d.target] = d
    decided = {d.target for d in resp.decisions}
    errors.extend(f"{t}: missing decision" for t in sorted(names - decided))
    return ok, errors
