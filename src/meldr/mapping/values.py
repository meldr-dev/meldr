"""Categorical value matching to canonical vocabularies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from meldr.core.mapping import TransformStep
from meldr.core.schema import FieldProfile

VALUE_SYNONYMS: dict[str, set[str]] = {
    "active": {"a", "act", "y", "yes", "true", "t", "1", "enabled", "open", "current"},
    "inactive": {"i", "inact", "n", "no", "false", "f", "0", "disabled", "closed", "former"},
    "female": {"f", "fem", "w", "woman"},
    "male": {"m", "man"},
    "other": {"o", "oth", "misc"},
    "unknown": {"u", "unk", "?"},
}
FUZZY_RATIO = 90
SHORTLIST_ABOVE = 40  # for larger vocabularies, the LLM gets a shortlist for each value
SHORTLIST_SIZE = 10


class ValueMapResponse(BaseModel):
    """LLM response for residue values."""

    mapping: dict[str, str | None]
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class ValueMatch:
    mapping: dict[str, str | None]
    residue: list[str] = field(default_factory=list)


def observed_values(profile: FieldProfile) -> list[str]:
    """The profiled most frequent values of a source field, as strings."""
    return [str(v["value"]) for v in profile.hints.get("top_values", [])]


def _norm(v: str) -> str:
    return v.strip().casefold()


def match_values(values: Sequence[str], vocabulary: Sequence[str]) -> ValueMatch:
    """Same value -> synonym -> unique prefix -> fuzzy (ratio ≥ 90), with the rest as residue."""
    vocab = {_norm(v): v for v in vocabulary}
    mapping: dict[str, str | None] = {}
    residue: list[str] = []
    for raw in dict.fromkeys(values):
        v = _norm(raw)
        if not v:
            continue
        if v in vocab:
            mapping[raw] = vocab[v]
            continue
        syn = [canon for canon, alts in VALUE_SYNONYMS.items() if v in alts and canon in vocab]
        if len(syn) == 1:
            mapping[raw] = vocab[syn[0]]
            continue
        prefixed = [orig for norm, orig in vocab.items() if norm.startswith(v)]
        if len(prefixed) == 1:
            mapping[raw] = prefixed[0]
            continue
        scored = sorted(((fuzz.ratio(v, n), orig) for n, orig in vocab.items()), reverse=True)
        if scored and scored[0][0] >= FUZZY_RATIO:
            mapping[raw] = scored[0][1]
            continue
        residue.append(raw)
    return ValueMatch(mapping=mapping, residue=residue)


def value_map_request(values: Sequence[str], vocabulary: Sequence[str]) -> dict[str, object]:
    """LLM payload for unresolved values (a large vocabulary gets a shortlist for each value)."""
    if len(vocabulary) <= SHORTLIST_ABOVE:
        return {"vocabulary": list(vocabulary), "values": list(values)}
    return {
        "values": [
            {
                "value": v,
                "candidates": [
                    match
                    for match, _, _ in process.extract(
                        v, vocabulary, scorer=fuzz.WRatio, processor=_norm, limit=SHORTLIST_SIZE
                    )
                ],
            }
            for v in values
        ]
    }


def default_for(vocabulary: Sequence[str]) -> str | None:
    """`other`/`unknown` when present in the vocabulary, else None."""
    for cand in ("other", "unknown"):
        if cand in vocabulary:
            return cand
    return None


def map_values_step(mapping: dict[str, str | None], vocabulary: Sequence[str]) -> TransformStep:
    """The `map_values` step for a complete source-value mapping."""
    return TransformStep(
        op="map_values", args={"mapping": mapping, "default": default_for(vocabulary)}
    )
