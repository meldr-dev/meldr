"""Default oracle for the `fake` provider: answers like a careful heuristic model.

It lets demos and tests exercise the full LLM code path without network access. It reads only
the request payload (it does not run anything from it).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from meldr.core.canonical import CanonicalField
from meldr.core.dtypes import DataType
from meldr.core.schema import FieldProfile
from meldr.llm.providers.fake import FakeRequest
from meldr.mapping.heuristic import PROPOSE_MIN, suggest_steps
from meldr.profiling.dates import DATE_FORMATS


def heuristic_oracle(req: FakeRequest) -> dict[str, Any]:
    payload = req.payload() or {}
    task = payload.get("task")
    if task in ("mapping", "mapping_repair"):
        return _mapping(payload)
    if task == "value_map":
        values = [v["value"] if isinstance(v, dict) else v for v in payload.get("values", [])]
        return {"mapping": dict.fromkeys(values), "confidence": 0.3}
    if task == "extraction":
        return _extraction(payload)
    if task == "captions":
        return {
            "captions": [
                {"frame_ms": ms, "caption": f"Frame at {ms} ms", "visible_text": ""}
                for ms in payload.get("frame_ms", [])
            ]
        }
    if task == "entity_match":
        a, b = payload.get("a", {}), payload.get("b", {})
        same = all(str(a.get(k, "")).casefold() == str(b.get(k, "")).casefold() for k in a)
        return {"same": same, "confidence": 0.6}
    return {}


def _profile(d: dict[str, Any]) -> FieldProfile:
    return FieldProfile(
        path=d["path"],
        name=d["path"].rsplit(".", 1)[-1],
        dtype=DataType(d.get("dtype", "string")),
        inferred_dtype=DataType(d.get("inferred_dtype", d.get("dtype", "string"))),
        null_ratio=0.0,
        distinct_ratio=0.0,
        patterns=d.get("patterns") or [],
        samples=d.get("samples") or [],
        hints=d.get("hints") or {},
        description=d.get("description"),
    )


def _mapping(payload: dict[str, Any]) -> dict[str, Any]:
    sources = {d["path"]: d for d in payload.get("source_fields", [])}
    targets = payload.get("target_fields", [])
    pairs = sorted(
        (-float(c["score"]), t["name"], c["path"])
        for t in targets
        for c in payload.get("candidates", {}).get(t["name"], [])
        if c["score"] >= PROPOSE_MIN and c["path"] in sources
    )
    chosen: dict[str, tuple[str, float]] = {}
    used: set[str] = set()
    for neg, name, path in pairs:
        if name not in chosen and path not in used:
            chosen[name] = (path, -neg)
            used.add(path)
    decisions = []
    for t in targets:
        if t["name"] not in chosen:
            decisions.append(
                {
                    "target": t["name"],
                    "sources": [],
                    "steps": [],
                    "confidence": 0.2,
                    "rationale": "no convincing source field",
                }
            )
            continue
        path, score = chosen[t["name"]]
        target = CanonicalField(
            name=t["name"],
            dtype=DataType(t["dtype"]),
            description=t.get("description", ""),
            unit=t.get("unit"),
            vocabulary=t.get("vocabulary"),
            synonyms=t.get("synonyms", []),
        )
        steps = suggest_steps(_profile(sources[path]), target)
        decisions.append(
            {
                "target": t["name"],
                "sources": [path],
                "steps": [s.model_dump() for s in steps],
                "confidence": round(min(0.99, 0.55 + 0.45 * score), 3),
                "rationale": "best candidate by name, type and pattern",
            }
        )
    return {"decisions": decisions}


def _typed(value: str, f: dict[str, Any]) -> Any:
    dtype = f.get("dtype")
    v = value.strip().rstrip(".")
    if f.get("vocabulary"):
        low = {x.casefold(): x for x in f["vocabulary"]}
        return low.get(v.casefold())
    if dtype in ("integer", "float", "decimal"):
        num = re.sub(r"[^\d.\-]", "", v)
        try:
            return int(num) if dtype == "integer" else float(num)
        except ValueError:
            return None
    if dtype == "boolean":
        return v.casefold() in {"yes", "true", "active", "y"}
    if dtype == "date":
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(v, fmt).date().isoformat()
            except ValueError:
                continue
        return None
    return v


def _extraction(payload: dict[str, Any]) -> dict[str, Any]:
    fields = payload.get("entity", {}).get("fields", [])
    values: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    confidence: dict[str, float] = {}
    for seg in payload.get("segments", []):
        text = seg["text"].split("] ", 1)[-1]
        for f in fields:
            if f["name"] in values:
                continue
            labels = [
                f["name"].replace("_", " "),
                *[s.replace("_", " ") for s in f.get("synonyms", []) or []],
            ]
            for label in labels:
                m = re.search(rf"(?im)^[\s*\-]*{re.escape(label)}\s*[:\-–]\s*(.+?)\s*$", text)
                if not m:
                    continue
                typed = _typed(m.group(1), f)
                if typed is None:
                    continue
                values[f["name"]] = typed
                evidence[f["name"]] = {
                    "segment_id": seg["segment_id"],
                    "quote": m.group(0).strip()[:300],
                }
                confidence[f["name"]] = 0.9
                break
    return {"values": values, "evidence": evidence, "confidence": confidence}
