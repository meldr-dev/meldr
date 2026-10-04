"""Final confidence and the auto-approval policy (§4.10)."""

from __future__ import annotations

from meldr.config import MappingConfig
from meldr.core.canonical import CanonicalEntity
from meldr.core.mapping import FieldMapping, FieldStatus, Origin
from meldr.core.schema import SourceSchema
from meldr.mapping.candidates import Candidate

DISAGREEMENT_MIN = 0.8


def final_confidence(
    mapping: FieldMapping, *, candidate_score: float, failure: float, cfg: MappingConfig
) -> float:
    if mapping.origin == Origin.HEURISTIC:
        value = 0.8 * candidate_score * (1 - failure)
    else:
        w = cfg.confidence_weights
        value = (
            w.llm * mapping.confidence + w.candidate * candidate_score + w.dryrun * (1 - failure)
        )
    return round(min(max(value, 0.0), 1.0), 4)


def decide(
    mapping: FieldMapping,
    *,
    entity: CanonicalEntity,
    schema: SourceSchema,
    candidate_score: float,
    failure: float,
    heuristic_top: Candidate | None,
    cfg: MappingConfig,
    extra_reason: str | None = None,
) -> FieldMapping:
    """Return `mapping` with final confidence, status, and `review_reason` set."""
    conf = final_confidence(mapping, candidate_score=candidate_score, failure=failure, cfg=cfg)
    target = entity.field(mapping.target)
    reason: str | None = extra_reason
    if reason is None and conf < cfg.auto_approve_threshold:
        reason = f"below auto-approve threshold ({cfg.auto_approve_threshold:.2f})"
    if reason is None and failure > cfg.max_dryrun_failure:
        reason = f"dry-run failure {failure:.1%} above {cfg.max_dryrun_failure:.1%}"
    if reason is None and target.pii and cfg.review_required_for_pii:
        reason = "pii_review_required"
    if (
        reason is None
        and mapping.origin == Origin.LLM
        and heuristic_top is not None
        and heuristic_top.score >= DISAGREEMENT_MIN
        and heuristic_top.path not in mapping.sources
    ):
        reason = "heuristic_llm_disagreement"
    if reason is None and any(
        s in schema.paths() and schema.field(s).hints.get("ambiguous_date") for s in mapping.sources
    ):
        reason = "ambiguous_date"
    if reason is None and f"{entity.name}.{target.name}" in cfg.always_review:
        reason = "always_review"
    return mapping.model_copy(
        update={
            "confidence": conf,
            "status": FieldStatus.PROPOSED if reason else FieldStatus.APPROVED,
            "review_reason": reason,
        }
    )
