"""Review operations.

Each operation returns a changed copy. When you save the copy, the store makes a new version.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from meldr._compat import UTC
from meldr.core.errors import SpecValidationError
from meldr.core.mapping import (
    FieldMapping,
    FieldStatus,
    MappingSpec,
    Origin,
    ReviewCandidate,
    ReviewDecision,
    ReviewItem,
)
from meldr.transforms.registry import TransformRegistry


def _with_fields(spec: MappingSpec, fields: list[FieldMapping], user: str | None) -> MappingSpec:
    """Copy with new fields, and with `user` as the approver when given."""
    return spec.model_copy(
        update={
            "fields": fields,
            "parent_version": spec.version,
            "approved_by": user or spec.approved_by,
            "updated_at": datetime.now(UTC),
        }
    )


def approve(
    spec: MappingSpec,
    *,
    user: str,
    targets: Iterable[str] | None = None,
    above: float | None = None,
) -> MappingSpec:
    """Approve the named targets, or every proposed field with confidence ≥ `above`."""
    wanted = set(targets) if targets is not None else None
    unknown = (wanted or set()) - {f.target for f in spec.fields}
    if unknown:
        raise SpecValidationError(f"No mapping for target(s) {sorted(unknown)} in {spec.spec_id}.")
    fields = []
    for f in spec.fields:
        hit = f.status == FieldStatus.PROPOSED and (
            (wanted is not None and f.target in wanted)
            or (wanted is None and above is not None and f.confidence >= above)
        )
        fields.append(
            f.model_copy(update={"status": FieldStatus.APPROVED, "review_reason": None})
            if hit
            else f
        )
    return _with_fields(spec, fields, user)


def reject(spec: MappingSpec, target: str, *, user: str, reason: str = "rejected") -> MappingSpec:
    if spec.mapping_for(target) is None:
        raise SpecValidationError(f"No mapping for target '{target}' in {spec.spec_id}.")
    fields = [
        f.model_copy(update={"status": FieldStatus.REJECTED, "review_reason": f"{reason} ({user})"})
        if f.target == target
        else f
        for f in spec.fields
    ]
    return _with_fields(spec, fields, None)


def edit(
    spec: MappingSpec,
    target: str,
    changes: dict[str, Any] | str,
    *,
    user: str,
    ops: TransformRegistry,
    source_paths: Iterable[str] | None = None,
) -> MappingSpec:
    """Replace one field mapping (dict or YAML text) with a human-approved mapping."""
    raw = yaml.safe_load(changes) if isinstance(changes, str) else dict(changes)
    if not isinstance(raw, dict):
        raise SpecValidationError("Edited mapping must be a YAML mapping.")
    current = spec.mapping_for(target)
    base: dict[str, Any] = current.model_dump() if current else {"target": target}
    base.update(raw)
    base.update(
        target=target,
        origin=Origin.HUMAN,
        status=FieldStatus.APPROVED,
        review_reason=None,
        confidence=1.0,
        dryrun=None,
        carried_from=None,
    )
    try:
        mapping = FieldMapping.model_validate(base)
    except ValidationError as e:
        raise SpecValidationError(f"Invalid mapping for '{target}': {e}") from e
    ops.validate(mapping)
    known = set(source_paths) if source_paths is not None else set(spec.source_type_signature)
    missing = [s for s in mapping.sources if s not in known]
    if missing:
        raise SpecValidationError(f"Unknown source path(s) {missing} for {spec.spec_id}.")
    fields = [f for f in spec.fields if f.target != target] + [mapping]
    unmapped_targets = [t for t in spec.unmapped_targets if t != target]
    used = {s for f in fields for s in f.sources}
    out = _with_fields(spec, fields, user)
    return out.model_copy(
        update={
            "unmapped_targets": unmapped_targets,
            "unmapped_sources": [s for s in spec.source_type_signature if s not in used],
        }
    )


def items(spec: MappingSpec, *, new_values: Mapping[str, int] | None = None) -> list[ReviewItem]:
    """Build review items from a mapping spec and optional value drift counts."""
    required = set(spec.required_targets)
    result: list[ReviewItem] = []
    for field in spec.pending_fields():
        kind: Literal["uncertain_field", "unmapped_required"] = "uncertain_field"
        if field.target in required and not field.sources:
            kind = "unmapped_required"
        result.append(
            ReviewItem(
                kind=kind,
                target=field.target,
                candidates=[
                    ReviewCandidate(source=alternative.path, confidence=alternative.score)
                    for alternative in spec.candidates.get(field.target, [])
                ],
                spec_id=spec.spec_id,
                version=spec.version,
            )
        )
    for target in spec.unmapped_targets:
        if target not in required:
            continue
        if any(item.target == target for item in result):
            continue
        result.append(
            ReviewItem(
                kind="unmapped_required",
                target=target,
                candidates=[
                    ReviewCandidate(source=alternative.path, confidence=alternative.score)
                    for alternative in spec.candidates.get(target, [])
                ],
                spec_id=spec.spec_id,
                version=spec.version,
            )
        )
    for target, count in (new_values or {}).items():
        result.append(
            ReviewItem(
                kind="new_values",
                target=target,
                spec_id=spec.spec_id,
                version=spec.version,
                count=count,
            )
        )
    return result


def apply_review(
    spec: MappingSpec,
    decisions: Sequence[ReviewDecision | Mapping[str, Any]],
    *,
    user: str,
    ops: TransformRegistry | None = None,
) -> MappingSpec:
    """Apply caller decisions to a spec and return its next revision."""
    changed = spec
    for raw in decisions:
        decision = (
            raw if isinstance(raw, ReviewDecision) else ReviewDecision.model_validate(dict(raw))
        )
        if decision.action == "approve":
            changed = approve(changed, user=user, targets=[decision.target])
        elif decision.action == "reject":
            changed = reject(changed, decision.target, user=user, reason=decision.reason)
        else:
            if ops is None:
                raise SpecValidationError(
                    "An edit decision must have a TransformRegistry. Pass `ops` to apply_review()."
                )
            if decision.changes is None:
                raise SpecValidationError(
                    f"Edit decision for '{decision.target}' must have `changes`."
                )
            changed = edit(
                changed,
                decision.target,
                decision.changes,
                user=user,
                ops=ops,
            )
    return changed
