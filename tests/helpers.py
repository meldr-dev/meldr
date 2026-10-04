from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from meldr import MeldrConfig
from meldr._compat import UTC
from meldr.core.dtypes import DataType
from meldr.core.mapping import Alternative, FieldMapping, GeneratorInfo, MappingSpec


def make_config(root: Path) -> MeldrConfig:
    """Config with local state under `root` and the offline `fake` provider."""
    return MeldrConfig.from_dict(
        {"storage": {"root": str(root)}, "llm": {"providers": {"fake": {"type": "fake"}}}}
    )


def make_spec(
    fields: list[FieldMapping], candidates: dict[str, list[Alternative]] | None = None
) -> MappingSpec:
    """A minimal spec `s.E` whose required target is `a`."""
    now = datetime.now(UTC)
    return MappingSpec(
        spec_id="s.E",
        version=1,
        source_id="s",
        source_fingerprint="fp",
        source_type_signature={f.target: DataType.STRING for f in fields},
        target_entity="E",
        target_version="1.0.0",
        fields=fields,
        required_targets=["a"],
        candidates=candidates or {},
        generator=GeneratorInfo(meldr_version="0"),
        created_at=now,
        updated_at=now,
    )


def decision(
    target: str,
    sources: list[str],
    steps: list[dict[str, Any]] | None = None,
    conf: float = 0.97,
    rationale: str = "r",
) -> dict[str, Any]:
    """One mapping decision as the LLM returns it."""
    return {
        "target": target,
        "sources": sources,
        "steps": steps or [],
        "confidence": conf,
        "rationale": rationale,
    }
