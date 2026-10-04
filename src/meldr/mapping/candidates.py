"""Cheap candidate retrieval: name, type, pattern, value, and embedding signals."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

import polars as pl
from rapidfuzz import fuzz

from meldr._math import cosine
from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import DataType, type_compat
from meldr.core.schema import FieldProfile, SourceSchema
from meldr.llm.base import Embedder
from meldr.mapping.normalize import normalize, tokens
from meldr.mapping.values import observed_values
from meldr.profiling.patterns import CONFLICTS, TARGET_PATTERN_HINTS
from meldr.transforms.ops import clean_number

WEIGHTS = {
    "name": 0.35,
    "embed": 0.30,
    "instance": 0.25,
    "type": 0.15,
    "pattern": 0.10,
    "value": 0.10,
}
VALUE_MATCH_RATIO = 85
NO_SHARED_TOKEN_PENALTY = 0.75
INFORMATIVE_VALUES = 6


@dataclass(frozen=True)
class Candidate:
    path: str
    score: float
    signals: dict[str, float] = field(default_factory=dict, compare=False)


def _share_token(a: str, b: str) -> bool:
    """Same word, or one word a prefix of the other ("sys"/"system"), after normalization."""
    return any(
        x == y or (min(len(x), len(y)) >= 3 and (x.startswith(y) or y.startswith(x)))
        for x in a.split()
        for y in b.split()
    )


def name_score(source_path: str, target: CanonicalField) -> float:
    s = normalize(source_path)
    leaf = normalize(source_path.rsplit(".", 1)[-1])
    best = 0.0
    for name in [target.name, *target.synonyms]:
        t = normalize(name)
        if not s or not t:
            continue
        if s == t or leaf == t:
            return 1.0
        score = (0.6 * fuzz.token_set_ratio(s, t) + 0.4 * fuzz.token_sort_ratio(s, t)) / 100
        if not _share_token(s, t):
            score *= (
                NO_SHARED_TOKEN_PENALTY  # "pickleball" vs "kickball" look alike, mean different
            )
        best = max(best, score)
    return best


def _type_score(s: FieldProfile, t: CanonicalField) -> float:
    score = type_compat(s.inferred_dtype, t.dtype)
    if t.dtype in (DataType.DATE, DataType.DATETIME) and s.hints.get("date_formats"):
        score = max(score, 0.6)
    return score


def _pattern_score(s: FieldProfile, t: CanonicalField) -> float:
    target_tokens = set(tokens(t.name))
    expected: set[str] = set()
    conflicts: set[str] = set()
    for tok in target_tokens:
        expected |= TARGET_PATTERN_HINTS.get(tok, set())
        conflicts |= CONFLICTS.get(tok, set())
    if t.dtype in (DataType.DATE, DataType.DATETIME):
        expected |= {"iso_date", "iso_datetime"}
    found = set(s.patterns)
    if found & expected:
        return 1.0
    if found & conflicts or (expected and found & {"email", "url", "uuid"} - expected):
        return 0.0
    return 0.5


def _value_score(s: FieldProfile, t: CanonicalField) -> float | None:
    values = observed_values(s)
    if not t.vocabulary or not values:
        return None
    vocab = [v.casefold() for v in t.vocabulary]
    hits = sum(
        1 for v in values if max(fuzz.ratio(v.casefold(), x) for x in vocab) >= VALUE_MATCH_RATIO
    )
    return hits / len(values)


def _canonical_value(v: str) -> str:
    """Formatting-insensitive form: casefold, drop an ISO time part and a '.0' at the end."""
    out = v.strip().casefold()
    if len(out) > 10 and out[4:5] == "-" and out[10:11] in ("t", " "):
        out = out[:10]
    return out.removesuffix(".0")


def _magnitudes(values: list[str]) -> list[float]:
    frame = pl.DataFrame({"v": values}, schema={"v": pl.String})
    numbers = frame.select(clean_number(pl.col("v"))).to_series().drop_nulls()
    return [math.log10(abs(x) + 1) for x in numbers]


def _instance_score(s: FieldProfile, t: CanonicalField) -> float | None:
    """Compare source samples with the target's example values (instance-based matching).

    Numbers: closeness of typical magnitude. Other values: overlap of formatting-insensitive
    values, with similar length as the fallback.
    """
    if not t.examples or not s.samples:
        return None
    src, tgt = _magnitudes(s.samples), _magnitudes(t.examples)
    if len(src) >= 0.8 * len(s.samples) and len(tgt) >= 0.8 * len(t.examples) and src and tgt:
        return max(0.0, 1.0 - abs(statistics.median(src) - statistics.median(tgt)) / 2)
    a = {_canonical_value(v) for v in s.samples}
    b = {_canonical_value(v) for v in t.examples}
    overlap = len(a & b) / len(a | b)
    la = statistics.mean(len(v) for v in a)
    lb = statistics.mean(len(v) for v in b)
    # Tiny value sets (Yes/No flags) overlap with every other flag, so they say little.
    informative = min(1.0, len(a | b) / INFORMATIVE_VALUES)
    return informative * max(overlap, 0.6 * min(la, lb) / max(la, lb, 1))


def descriptor_source(s: FieldProfile, samples: Sequence[str] = ()) -> str:
    parts = [s.path, str(s.inferred_dtype), ",".join(s.patterns), s.description or ""]
    if samples:
        parts.append("e.g. " + ", ".join(samples[:5]))
    return " | ".join(parts)


def descriptor_target(t: CanonicalField) -> str:
    return (
        f"{t.name} | {t.dtype} | {t.description} | synonyms: {', '.join(t.synonyms)} | "
        f"unit: {t.unit or ''}"
    )


def score_pair(s: FieldProfile, t: CanonicalField, *, embed_sim: float | None = None) -> Candidate:
    signals: dict[str, float] = {
        "name": name_score(s.path, t),
        "type": _type_score(s, t),
        "pattern": _pattern_score(s, t),
    }
    for key, value in (
        ("value", _value_score(s, t)),
        ("instance", _instance_score(s, t)),
        ("embed", embed_sim),
    ):
        if value is not None:
            signals[key] = value
    total_w = sum(WEIGHTS[k] for k in signals)
    score = sum(WEIGHTS[k] * v for k, v in signals.items()) / total_w
    if signals["name"] < 1.0:  # a name score of 1.0 outweighs a type or pattern mismatch
        if signals["type"] == 0.0:
            score *= 0.5
        if signals["pattern"] == 0.0:
            score *= 0.7
    return Candidate(path=s.path, score=round(score, 4), signals=signals)


async def generate(
    schema: SourceSchema,
    entity: CanonicalEntity,
    *,
    top_k: int = 5,
    min_score: float = 0.25,
    embedder: Embedder | None = None,
    embed_view: SourceSchema | None = None,
    target_vectors: dict[tuple[str, str], list[list[float]]] | None = None,
) -> dict[str, list[Candidate]]:
    """Top-k candidates for each target (plus each source with the same normalized name).

    The embedder sees samples only from `embed_view`, the policy-filtered schema (none if absent).
    `target_vectors` caches target embeddings for each (entity, version) across calls.
    """
    sims: dict[tuple[str, str], float] = {}
    if embedder is not None and schema.fields:
        cache = target_vectors if target_vectors is not None else {}
        key = (entity.name, entity.version)
        if key not in cache:
            cache[key] = await embedder.embed([descriptor_target(t) for t in entity.fields])
        tvecs = cache[key]
        visible = {f.path: f.samples for f in embed_view.fields} if embed_view else {}
        svecs = await embedder.embed(
            [descriptor_source(s, visible.get(s.path, ())) for s in schema.fields]
        )
        for t, tv in zip(entity.fields, tvecs, strict=True):
            for s, sv in zip(schema.fields, svecs, strict=True):
                sims[(s.path, t.name)] = max(0.0, cosine(sv, tv))
    out: dict[str, list[Candidate]] = {}
    for t in entity.fields:
        scored = [score_pair(s, t, embed_sim=sims.get((s.path, t.name))) for s in schema.fields]
        scored.sort(key=lambda c: (-c.score, c.path))
        keep = [c for c in scored if c.score >= min_score][:top_k]
        for c in scored:
            if c.signals.get("name") == 1.0 and c not in keep:
                keep.append(c)
        out[t.name] = keep
    return out
