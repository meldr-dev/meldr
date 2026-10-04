"""Heuristic proposer: greedy one-to-one assignment with automatic steps (no LLM)."""

from __future__ import annotations

import re

from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import DataType
from meldr.core.mapping import FieldMapping, Origin, TransformStep
from meldr.core.schema import FieldProfile, SourceSchema
from meldr.mapping.candidates import Candidate
from meldr.mapping.values import map_values_step, match_values, observed_values
from meldr.transforms import units
from meldr.transforms.ops import UnitConvertArgs

PROPOSE_MIN = 0.55
# A candidate this close to the runner-up is a coin flip. Leave it to the LLM or a reviewer.
AMBIGUOUS_MARGIN = 0.05
_NUMERIC = {DataType.INTEGER, DataType.FLOAT, DataType.DECIMAL}
_UNIT_SUFFIX = re.compile(r"(?:^|[_\-. ])(cents|ft|feet|m|km|mi|kg|lbs|lb|g|in|sqft|c|f)$", re.I)


def unit_step(source: FieldProfile, target: CanonicalField) -> TransformStep | None:
    """`scale` for cents, `unit_convert` for unit suffixes, when the target has a unit."""
    if not target.unit:
        return None
    m = _UNIT_SUFFIX.search(source.path)
    if not m:
        return None
    suffix = m.group(1).lower()
    if suffix == "cents":
        return TransformStep(op="scale", args={"factor": 0.01})
    from_u, to_u = units.canonical_unit(suffix), units.canonical_unit(target.unit)
    if from_u == to_u or not (units.is_known(from_u) and units.is_known(to_u)):
        return None
    try:
        UnitConvertArgs(from_unit=from_u, to_unit=to_u)
    except ValueError:
        return None
    return TransformStep(op="unit_convert", args={"from_unit": from_u, "to_unit": to_u})


def suggest_steps(source: FieldProfile, target: CanonicalField) -> list[TransformStep]:
    """Automatic steps for a (source, target) pair (§4.6)."""
    steps: list[TransformStep] = []
    sentinels = [s for s in source.hints.get("sentinels", []) if s]
    if sentinels or "" in source.hints.get("sentinels", []):
        steps.append(TransformStep(op="nullify", args={"values": [*sentinels, ""]}))
    src_t = source.inferred_dtype
    is_string = source.dtype == DataType.STRING
    epoch = source.hints.get("epoch")
    if (
        target.dtype in (DataType.DATE, DataType.DATETIME)
        and epoch
        and "date_formats" not in (source.hints)
    ):
        steps.append(TransformStep(op="from_epoch", args=dict(epoch)))
    elif target.dtype == DataType.DATE and (is_string or source.dtype == DataType.INTEGER):
        formats = source.hints.get("date_formats") or []
        if formats:
            steps.append(TransformStep(op="parse_date", args={"formats": formats}))
        elif source.hints.get("datetime_formats"):
            steps.append(
                TransformStep(
                    op="parse_datetime", args={"formats": source.hints["datetime_formats"]}
                )
            )
    elif target.dtype == DataType.DATETIME and is_string:
        formats = source.hints.get("datetime_formats") or source.hints.get("date_formats") or []
        if formats:
            steps.append(TransformStep(op="parse_datetime", args={"formats": formats}))
    elif target.dtype in _NUMERIC and is_string and src_t in _NUMERIC:
        args: dict[str, str] = {}
        if source.hints.get("decimal_sep") == ",":
            args = {"decimal_sep": ",", "thousands_sep": "."}
        steps.append(TransformStep(op="parse_number", args=args))
    elif target.dtype in _NUMERIC and is_string and source.hints.get("embedded_number"):
        steps.append(TransformStep(op="regex_extract", args={"pattern": r"(\d+(?:\.\d+)?)"}))
        steps.append(TransformStep(op="parse_number"))
    elif target.dtype == DataType.BOOLEAN and (is_string or src_t != DataType.BOOLEAN):
        steps.append(TransformStep(op="parse_bool"))
    elif target.vocabulary and (observed := observed_values(source)):
        vm = match_values(observed, target.vocabulary)
        mapping = {**vm.mapping, **dict.fromkeys(vm.residue)}
        steps.append(map_values_step(mapping, target.vocabulary))
    elif target.dtype == DataType.STRING and is_string:
        steps.append(TransformStep(op="trim"))
    if target.dtype in _NUMERIC:
        u = unit_step(source, target)
        if u is not None:
            steps.append(u)
    return steps


def propose(
    schema: SourceSchema,
    entity: CanonicalEntity,
    candidates: dict[str, list[Candidate]],
    *,
    targets: set[str] | None = None,
) -> dict[str, tuple[FieldMapping, float]]:
    """Greedy one-to-one assignment. Returns target -> (mapping, candidate score).

    If a target's best free candidate barely beats its runner-up, the target stays unmapped,
    unless the names are the same. Heuristics are wrong most often on near-ties.
    """
    pairs = [
        (c, t)
        for t, cands in candidates.items()
        if targets is None or t in targets
        for c in cands
        if c.score >= PROPOSE_MIN
    ]
    pairs.sort(key=lambda p: (-p[0].score, p[1], p[0].path))
    used_sources: set[str] = set()
    decided: set[str] = set()
    out: dict[str, tuple[FieldMapping, float]] = {}
    for cand, t in pairs:
        score, path = cand.score, cand.path
        if t in decided or path in used_sources:
            continue
        decided.add(t)
        runner_up = max((c.score for c in candidates[t] if c.path != path), default=0.0)
        exact = cand.signals.get("name", 0.0) >= 1.0 or "memory" in cand.signals
        if not exact and score - runner_up < AMBIGUOUS_MARGIN:
            continue
        used_sources.add(path)
        target = entity.field(t)
        out[t] = (
            FieldMapping(
                target=t,
                sources=[path],
                steps=suggest_steps(schema.field(path), target),
                confidence=round(0.8 * score, 4),
                rationale=f"heuristic: candidate score {score:.2f}",
                origin=Origin.HEURISTIC,
            ),
            score,
        )
    return out
