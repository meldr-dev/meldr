"""Drift detection and incremental mapping updates (§4.11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from meldr.core.canonical import CanonicalEntity
from meldr.core.dtypes import type_compat
from meldr.core.mapping import FieldMapping, FieldStatus, MappingSpec
from meldr.core.schema import SourceSchema
from meldr.mapping.values import match_values
from meldr.profiling.profiler import CATEGORICAL_MAX_DISTINCT, most_frequent


@dataclass
class DriftReport:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    type_changed: list[str] = field(default_factory=list)
    breaking_targets: list[str] = field(default_factory=list)


@dataclass
class Reconciliation:
    carried: list[FieldMapping]
    retarget: list[str]
    report: DriftReport


def diff(prev: MappingSpec, schema: SourceSchema, entity: CanonicalEntity) -> DriftReport:
    """Added, removed, and retyped paths, plus required targets that lost an approved source."""
    old = prev.source_type_signature
    new = schema.type_signature()
    report = DriftReport(
        added=sorted(set(new) - set(old)),
        removed=sorted(set(old) - set(new)),
        type_changed=sorted(p for p in set(old) & set(new) if type_compat(old[p], new[p]) < 1.0),
    )
    required = {f.name for f in entity.fields if f.required}
    report.breaking_targets = sorted(
        m.target
        for m in prev.fields
        if m.status == FieldStatus.APPROVED
        and m.target in required
        and any(s in report.removed for s in m.sources)
    )
    return report


def reconcile(prev: MappingSpec, schema: SourceSchema, entity: CanonicalEntity) -> Reconciliation:
    """Keep a mapping if its sources are present with compatible types, then map the rest again."""
    report = diff(prev, schema, entity)
    broken = set(report.removed) | set(report.type_changed)
    carried: list[FieldMapping] = []
    for m in prev.fields:
        if m.target not in entity.field_names() or any(s in broken for s in m.sources):
            continue
        carried.append(m.model_copy(update={"carried_from": prev.version}))
    carried_targets = {m.target for m in carried}
    retarget = [n for n in entity.field_names() if n not in carried_targets]
    return Reconciliation(carried=carried, retarget=retarget, report=report)


def unseen_values(spec: MappingSpec, table: pl.DataFrame) -> dict[str, list[str]]:
    """For each target: frequent source values that its map_values step does not cover."""
    out: dict[str, list[str]] = {}
    for f in spec.fields:
        step = next((s for s in f.steps if s.op == "map_values"), None)
        if step is None or len(f.sources) != 1 or f.sources[0] not in table.columns:
            continue
        fold = (
            (lambda v: v.strip().casefold())
            if step.args.get("case_insensitive", True)
            else str.strip
        )
        known = {fold(str(k)) for k in step.args.get("mapping", {})}
        frequent = most_frequent(table[f.sources[0]], CATEGORICAL_MAX_DISTINCT)
        unseen = [v for v, _ in frequent if fold(v) not in known]
        if unseen:
            out[f.target] = unseen
    return out


def refresh_values(
    spec: MappingSpec, table: pl.DataFrame, entity: CanonicalEntity
) -> tuple[MappingSpec | None, dict[str, list[str]]]:
    """Propose map_values additions for new categorical values (as PROPOSED fields)."""
    unseen = unseen_values(spec, table)
    if not unseen:
        return None, {}
    fields: list[FieldMapping] = []
    for f in spec.fields:
        values = unseen.get(f.target)
        if values is None or f.target not in entity.field_names():
            fields.append(f)
            continue
        vm = match_values(values, entity.field(f.target).vocabulary or [])
        steps = []
        for s in f.steps:
            if s.op == "map_values":
                mapping = {**s.args.get("mapping", {}), **vm.mapping, **dict.fromkeys(vm.residue)}
                s = s.model_copy(update={"args": {**s.args, "mapping": mapping}})
            steps.append(s)
        fields.append(
            f.model_copy(
                update={
                    "steps": steps,
                    "status": FieldStatus.PROPOSED,
                    "review_reason": f"new_values:{len(values)}",
                }
            )
        )
    return spec.model_copy(update={"fields": fields, "parent_version": spec.version}), unseen


def spec_diff(a: MappingSpec, b: MappingSpec) -> list[dict[str, Any]]:
    """Field-level changes from spec `a` to spec `b`."""
    fa = {f.target: f for f in a.fields}
    fb = {f.target: f for f in b.fields}
    changes: list[dict[str, Any]] = []
    for t in sorted(set(fa) | set(fb)):
        x, y = fa.get(t), fb.get(t)
        if x is None:
            changes.append({"target": t, "change": "added", "to": _summary(y)})
        elif y is None:
            changes.append({"target": t, "change": "removed", "from": _summary(x)})
        elif _summary(x) != _summary(y):
            changes.append(
                {"target": t, "change": "modified", "from": _summary(x), "to": _summary(y)}
            )
    return changes


def _summary(m: FieldMapping | None) -> dict[str, Any]:
    if m is None:
        return {}
    return {
        "sources": m.sources,
        "steps": [s.model_dump() for s in m.steps],
        "status": str(m.status),
    }
