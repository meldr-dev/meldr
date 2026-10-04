"""Mapping engine: candidates -> heuristic/LLM proposals -> dry run -> approval."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from meldr._compat import UTC
from meldr._version import __version__
from meldr.config import MappingConfig
from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.errors import LLMError, PolicyViolation
from meldr.core.mapping import Alternative, FieldMapping, GeneratorInfo, MappingSpec, Origin
from meldr.core.schema import SourceSchema
from meldr.llm.base import Embedder, Message, Usage
from meldr.llm.gateway import LLMGateway
from meldr.llm.policy import DataClass, SourcePolicy
from meldr.llm.prompts import load as load_prompt
from meldr.mapping import adjudicator, candidates, heuristic
from meldr.mapping.candidates import Candidate
from meldr.mapping.confidence import DISAGREEMENT_MIN, decide
from meldr.mapping.dryrun import FieldDryRun, dry_run
from meldr.mapping.values import (
    ValueMapResponse,
    map_values_step,
    match_values,
    observed_values,
    value_map_request,
)
from meldr.observability import safe_log
from meldr.profiling.masking import apply_policy, is_pii_name, mask_value, samples_class
from meldr.transforms.registry import TRANSFORMS_VERSION, TransformRegistry

VALUE_PROMPT_VERSION = "value-map-v2"

Matcher = Callable[[SourceSchema, CanonicalEntity], dict[str, list[Candidate]]]


@dataclass
class Proposal:
    fields: list[FieldMapping] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    llm_calls: int = 0
    provider: str | None = None
    model: str | None = None
    warnings: list[str] = field(default_factory=list)
    candidates: dict[str, list[Candidate]] = field(default_factory=dict)

    def charge(self, calls: int, usage: Usage) -> None:
        """Account for LLM calls made while building this proposal."""
        self.llm_calls += calls
        self.usage = self.usage + usage


class MappingEngine:
    """Proposes mappings for one (source schema, canonical entity) pair."""

    def __init__(
        self,
        ops: TransformRegistry,
        config: MappingConfig | None = None,
        *,
        gateway: LLMGateway | None = None,
        escalation: LLMGateway | None = None,
        escalate_below: float = 0.9,
        embedder: Embedder | None = None,
        matchers: Sequence[Matcher] = (),
    ) -> None:
        self.ops = ops
        self.matchers = list(matchers)
        self.cfg = config or MappingConfig()
        self.gateway = gateway
        self.escalation = escalation
        self.escalate_below = escalate_below
        self.embedder = embedder
        self._target_vectors: dict[tuple[str, str], list[list[float]]] = {}

    async def propose(
        self,
        schema: SourceSchema,
        entity: CanonicalEntity,
        sample: pl.DataFrame,
        *,
        source_policy: SourcePolicy | None = None,
        targets: Iterable[str] | None = None,
        exclude_sources: Iterable[str] = (),
    ) -> Proposal:
        """Propose mappings for `targets` (default: every entity field)."""
        wanted = [t.name for t in entity.fields if targets is None or t.name in set(targets)]
        excluded = set(exclude_sources)
        pool = schema.model_copy(
            update={"fields": [f for f in schema.fields if f.path not in excluded]}
        )
        cands = await candidates.generate(
            pool,
            entity,
            top_k=self.cfg.top_k_candidates,
            min_score=self.cfg.min_candidate_score,
            embedder=self.embedder,
            embed_view=self._embed_view(pool, source_policy),
            target_vectors=self._target_vectors,
        )
        for matcher in self.matchers:
            cands = merge_candidates(cands, matcher(pool, entity), self.cfg.top_k_candidates)
        heur = heuristic.propose(pool, entity, cands, targets=set(wanted))
        proposal = Proposal(candidates={t: cands.get(t, []) for t in wanted})
        chosen: dict[str, FieldMapping] = {}
        reasons: dict[str, str] = {}

        adj: adjudicator.Adjudication | None = None
        if self.gateway is not None and wanted:
            try:
                adj = await self._adjudicate(
                    self.gateway, pool, entity, cands, wanted, source_policy
                )
                proposal.provider = self.gateway.provider.name
                proposal.model = adj.model
                proposal.charge(adj.calls, adj.usage)
            except (LLMError, PolicyViolation) as e:
                proposal.warnings.append(f"llm_unavailable:{type(e).__name__}")
                safe_log("mapping.llm_fallback", reason=type(e).__name__, entity=entity.name)

        for t in wanted:
            h = heur.get(t)
            if adj is None:
                if h is not None:
                    chosen[t] = h[0]
                continue
            if t in adj.invalid:
                if h is not None:
                    chosen[t] = h[0]
                    reasons[t] = "llm_invalid_output"
                continue
            decision = adj.decisions.get(t)
            if decision is not None:
                chosen[t] = decision
            elif h is not None and h[1] >= DISAGREEMENT_MIN:
                chosen[t] = h[0]
                reasons[t] = "heuristic_llm_disagreement"

        await self._complete_value_maps(chosen, pool, entity, source_policy, proposal, reasons)
        for t, path in _shared_sources(chosen).items():
            reasons.setdefault(t, f"shared_source:{path}")

        runs = dry_run(sample, list(chosen.values()), entity, self.ops)
        await self._self_correct(chosen, runs, pool, entity, cands, sample, source_policy, proposal)

        if self.escalation is not None and adj is not None:
            await self._escalate(chosen, runs, pool, entity, cands, sample, source_policy, proposal)

        for t in wanted:
            m = chosen.get(t)
            if m is None:
                continue
            run = runs[t]
            top = cands[t][0] if cands.get(t) else None
            if run.error:
                reasons.setdefault(t, "execution_error")
            decided = decide(
                m.model_copy(update={"dryrun": run.stats}),
                entity=entity,
                schema=schema,
                candidate_score=_candidate_score(cands, m),
                failure=run.failure,
                heuristic_top=top,
                cfg=self.cfg,
                extra_reason=reasons.get(t),
            )
            proposal.fields.append(decided)
        return proposal

    def _embed_view(
        self, schema: SourceSchema, source_policy: SourcePolicy | None
    ) -> SourceSchema | None:
        """What the embedder can see of `schema`: the same policy as LLM payloads, else nothing."""
        if self.gateway is None:
            return None
        return apply_policy(schema, self.gateway.effective_policy(source_policy))

    async def _adjudicate(
        self,
        gateway: LLMGateway,
        schema: SourceSchema,
        entity: CanonicalEntity,
        cands: dict[str, list[Candidate]],
        targets: Sequence[str],
        source_policy: SourcePolicy | None,
    ) -> adjudicator.Adjudication:
        return await adjudicator.adjudicate(
            gateway,
            schema,
            entity,
            cands,
            self.ops,
            targets=targets,
            source_policy=source_policy,
            max_fields_per_call=self.cfg.max_fields_per_llm_call,
            pii_paths=_pii_paths(entity, cands),
            samples=self.cfg.self_consistency,
        )

    async def _complete_value_maps(
        self,
        chosen: dict[str, FieldMapping],
        schema: SourceSchema,
        entity: CanonicalEntity,
        source_policy: SourcePolicy | None,
        proposal: Proposal,
        reasons: dict[str, str],
    ) -> None:
        """Vocabulary targets fed by categorical sources get a complete map_values step."""
        for t, m in list(chosen.items()):
            target = entity.field(t)
            if not target.vocabulary or len(m.sources) != 1:
                continue
            if any(s.op == "map_values" for s in m.steps):
                continue
            observed = observed_values(schema.field(m.sources[0]))
            if not observed:
                continue
            vm = match_values(observed, target.vocabulary)
            mapping = dict(vm.mapping)
            if vm.residue:
                resolved = await self._llm_residue(
                    target, m.sources[0], vm.residue, source_policy, proposal
                )
                if resolved is None:
                    reasons.setdefault(t, "unresolved_values")
                    resolved = dict.fromkeys(vm.residue)
                mapping.update(resolved)
            step = map_values_step(mapping, target.vocabulary)
            chosen[t] = m.model_copy(update={"steps": [*m.steps, step]})

    async def _llm_residue(
        self,
        target: CanonicalField,
        path: str,
        residue: list[str],
        source_policy: SourcePolicy | None,
        proposal: Proposal,
    ) -> dict[str, str | None] | None:
        if self.gateway is None or target.pii or is_pii_name(path):
            return None
        data_class = samples_class(self.gateway.effective_policy(source_policy))
        if data_class is None:
            return None
        # With a masked-only policy, the model sees masked values.
        # The answers map back to the raw values.
        sent: dict[str, list[str]] = {}
        for v in residue:
            sent.setdefault(
                mask_value(v) if data_class == DataClass.MASKED_SAMPLES else v, []
            ).append(v)
        payload = {
            "task": "value_map",
            "target_field": target.name,
            **value_map_request(list(sent), target.vocabulary or []),
        }
        try:
            res = await self.gateway.generate(
                [Message(role="user", content=json.dumps(payload))],
                response_model=ValueMapResponse,
                contains={DataClass.SCHEMA, data_class},
                purpose="value_map",
                prompt_version=VALUE_PROMPT_VERSION,
                system=load_prompt("value_map_v2"),
                source_policy=source_policy,
            )
        except (LLMError, PolicyViolation):
            return None
        proposal.charge(res.attempts, res.usage)
        vocab = set(target.vocabulary or [])
        out: dict[str, str | None] = {}
        for shown, raws in sent.items():
            answer = res.parsed.mapping.get(shown)
            for raw in raws:
                out[raw] = answer if answer in vocab and len(raws) == 1 else None
        return out

    async def _self_correct(
        self,
        chosen: dict[str, FieldMapping],
        runs: dict[str, FieldDryRun],
        schema: SourceSchema,
        entity: CanonicalEntity,
        cands: dict[str, list[Candidate]],
        sample: pl.DataFrame,
        source_policy: SourcePolicy | None,
        proposal: Proposal,
    ) -> None:
        """One repair round for the fields with a dry-run failure above the threshold."""
        gw = self.gateway
        failing = [
            t
            for t, m in chosen.items()
            if m.origin == Origin.LLM and runs[t].failure > self.cfg.max_dryrun_failure
        ]
        if gw is None or not failing:
            return
        pol = gw.effective_policy(source_policy)
        feedback = {}
        for t in failing:
            run = runs[t]
            item: dict[str, object] = {
                "previous": chosen[t].model_dump(include={"sources", "steps"}),
                "loss_rate": run.stats.loss_rate,
                "invalid_rate": run.stats.invalid_rate,
            }
            if pol.allows(DataClass.SAMPLES):
                item["examples"] = run.examples
            elif pol.allows(DataClass.MASKED_SAMPLES):
                item["examples"] = [_mask_example(e) for e in run.examples]
            else:
                item["problems"] = sorted({str(e["problem"]) for e in run.examples})
            feedback[t] = item
        visible = apply_policy(schema, pol, pii_paths=_pii_paths(entity, cands))
        payload = adjudicator.build_payload(
            visible, entity, [entity.field(t) for t in failing], cands, self.ops
        )
        payload["task"] = "mapping_repair"
        payload["dry_run_feedback"] = feedback
        sc = samples_class(pol)
        contains = {DataClass.SCHEMA} | ({sc} if sc else set())
        try:
            res = await gw.generate(
                [
                    Message(
                        role="user",
                        content="These mappings lost values or failed validation in a dry run. "
                        "Fix them.\n" + json.dumps(payload, default=str),
                    )
                ],
                response_model=adjudicator.LLMMappingResponse,
                contains=contains,
                purpose="mapping_selfcorrect",
                prompt_version=adjudicator.PROMPT_VERSION,
                system=load_prompt("mapping_v2"),
                source_policy=source_policy,
            )
        except (LLMError, PolicyViolation):
            return
        proposal.charge(res.attempts, res.usage)
        names = set(failing)
        for d in res.parsed.decisions:
            if adjudicator.validate_decision(d, names, schema, entity, self.ops) or not d.sources:
                continue
            candidate = adjudicator.to_mapping(d)
            rerun = dry_run(sample, [candidate], entity, self.ops)[d.target]
            if rerun.failure <= runs[d.target].failure:
                chosen[d.target] = candidate
                runs[d.target] = rerun

    async def _escalate(
        self,
        chosen: dict[str, FieldMapping],
        runs: dict[str, FieldDryRun],
        schema: SourceSchema,
        entity: CanonicalEntity,
        cands: dict[str, list[Candidate]],
        sample: pl.DataFrame,
        source_policy: SourcePolicy | None,
        proposal: Proposal,
    ) -> None:
        """Re-ask a stronger model for low-confidence LLM fields (R17)."""
        if self.escalation is None:
            return
        low = [
            t
            for t, m in chosen.items()
            if m.origin == Origin.LLM and m.confidence < self.escalate_below
        ]
        if not low:
            return
        try:
            adj = await self._adjudicate(self.escalation, schema, entity, cands, low, source_policy)
        except (LLMError, PolicyViolation):
            proposal.warnings.append("escalation_failed")
            return
        proposal.charge(adj.calls, adj.usage)
        for t, m in adj.decisions.items():
            if m is None:
                continue
            rerun = dry_run(sample, [m], entity, self.ops)[t]
            if m.confidence >= chosen[t].confidence and rerun.failure <= runs[t].failure:
                chosen[t] = m.model_copy(update={"rationale": f"[escalated] {m.rationale}"})
                runs[t] = rerun


def merge_candidates(
    base: dict[str, list[Candidate]], extra: dict[str, list[Candidate]], top_k: int
) -> dict[str, list[Candidate]]:
    """Combine candidate lists and keep the best score for each (target, path)."""
    out: dict[str, list[Candidate]] = {}
    for t in base.keys() | extra.keys():
        best: dict[str, Candidate] = {}
        for c in [*base.get(t, []), *extra.get(t, [])]:
            if c.path not in best or c.score > best[c.path].score:
                best[c.path] = c
        out[t] = sorted(best.values(), key=lambda c: (-c.score, c.path))[: max(top_k, 1)]
    return out


_SPLITTING_OPS = {"split_part", "regex_extract"}


def _shared_sources(chosen: dict[str, FieldMapping]) -> dict[str, str]:
    """Targets that copy the same source column as a different target (usually one is wrong).

    Mappings that deliberately take part of a value (split_part, regex_extract) are exempt.
    """
    users: dict[str, list[str]] = {}
    for t, m in chosen.items():
        if len(m.sources) == 1 and not any(s.op in _SPLITTING_OPS for s in m.steps):
            users.setdefault(m.sources[0], []).append(t)
    return {t: path for path, ts in users.items() if len(ts) > 1 for t in ts}


def _pii_paths(entity: CanonicalEntity, cands: dict[str, list[Candidate]]) -> set[str]:
    """Source paths that are candidates for PII targets (shape-masked before they go to the LLM)."""
    return {c.path for t in entity.fields if t.pii for c in cands.get(t.name, [])}


def _mask_example(e: dict[str, object]) -> dict[str, object]:
    inputs = e.get("input")
    out = e.get("output")
    return {
        "input": [None if v is None else mask_value(str(v)) for v in inputs]
        if isinstance(inputs, list)
        else None,
        "output": None if out is None else mask_value(str(out)),
        "problem": e.get("problem"),
    }


def _candidate_score(cands: dict[str, list[Candidate]], m: FieldMapping) -> float:
    by_path = {c.path: c.score for c in cands.get(m.target, [])}
    scores = [by_path.get(s, 0.0) for s in m.sources]
    return sum(scores) / len(scores) if scores else 0.0


def build_spec(
    schema: SourceSchema,
    entity: CanonicalEntity,
    proposal: Proposal,
    *,
    carried: Sequence[FieldMapping] = (),
    parent_version: int | None = None,
) -> MappingSpec:
    """Assemble a MappingSpec (version is assigned by the store on save)."""
    now = datetime.now(UTC)
    fields = [*carried, *proposal.fields]
    order = {n: i for i, n in enumerate(entity.field_names())}
    fields.sort(key=lambda f: order.get(f.target, len(order)))
    used = {s for f in fields for s in f.sources}
    mapped = {f.target for f in fields}
    return MappingSpec(
        spec_id=f"{schema.source_id}.{entity.name}",
        version=0,
        parent_version=parent_version,
        source_id=schema.source_id,
        source_fingerprint=schema.fingerprint,
        source_type_signature=schema.type_signature(),
        target_entity=entity.name,
        target_version=entity.version,
        fields=fields,
        unmapped_sources=[p for p in schema.paths() if p not in used],
        unmapped_targets=[n for n in entity.field_names() if n not in mapped],
        required_targets=[f.name for f in entity.fields if f.required],
        candidates={
            t: [Alternative(path=c.path, score=c.score) for c in cands]
            for t, cands in proposal.candidates.items()
            if cands
        },
        generator=GeneratorInfo(
            provider=proposal.provider,
            model=proposal.model,
            prompt_version=adjudicator.PROMPT_VERSION if proposal.provider else None,
            transforms_version=TRANSFORMS_VERSION,
            input_tokens=proposal.usage.input_tokens,
            output_tokens=proposal.usage.output_tokens,
            meldr_version=__version__,
        ),
        created_at=now,
        updated_at=now,
    )
