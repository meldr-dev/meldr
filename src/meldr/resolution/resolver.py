"""Entity resolution v1: blocks -> fuzzy scoring -> LLM tie-break -> union-find (§4.18)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import polars as pl
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from meldr.core.errors import LLMError, PolicyViolation
from meldr.llm.base import Message
from meldr.llm.gateway import LLMGateway
from meldr.llm.policy import DataClass
from meldr.llm.prompts import load as load_prompt
from meldr.mapping.normalize import tokens

PROMPT_VERSION = "entity-match-v1"
ENTITY_ID = "_meldr_entity_id"
MAX_BLOCK = 200


class SameEntity(BaseModel):
    same: bool
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class Resolution:
    data: pl.DataFrame
    links: pl.DataFrame
    llm_calls: int = 0
    pairs_scored: int = 0
    warnings: list[str] = field(default_factory=list)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _block_key(row: dict[str, object], block_on: Sequence[str]) -> str | None:
    parts = []
    for f in block_on:
        v = row.get(f)
        if v is None:
            return None
        toks = tokens(str(v))
        parts.append(toks[0] if toks else str(v).strip().casefold())
    return "|".join(parts)


def score(a: dict[str, object], b: dict[str, object], fields: Sequence[str]) -> float:
    """Mean token_sort_ratio for the fields present in the two records (0..1)."""
    scores = [
        fuzz.token_sort_ratio(str(a[f]).casefold(), str(b[f]).casefold()) / 100
        for f in fields
        if a.get(f) is not None and b.get(f) is not None
    ]
    return sum(scores) / len(scores) if scores else 0.0


async def resolve(
    df: pl.DataFrame,
    *,
    entity: str,
    fields: Sequence[str],
    block_on: Sequence[str],
    hi: float = 0.92,
    lo: float = 0.75,
    gateway: LLMGateway | None = None,
) -> Resolution:
    """Assign `_meldr_entity_id` to every row, and return the matched pairs as `links`."""
    rows = df.to_dicts()
    blocks: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        key = _block_key(r, block_on)
        if key is not None:
            blocks[key].append(i)
    uf = _UnionFind(len(rows))
    links: list[dict[str, object]] = []
    result = Resolution(data=df, links=pl.DataFrame())
    scored: list[tuple[int, int, float]] = []
    for members in blocks.values():
        if len(members) > MAX_BLOCK:
            result.warnings.append(f"block_too_large:{len(members)}")
            members = members[:MAX_BLOCK]
        scored.extend((i, j, score(rows[i], rows[j], fields)) for i, j in combinations(members, 2))
    result.pairs_scored = len(scored)
    ties = [(i, j) for i, j, s in scored if lo < s < hi]
    verdicts = await asyncio.gather(
        *(_tie_break(gateway, entity, fields, rows[i], rows[j], result) for i, j in ties)
    )
    tie_verdict = dict(zip(ties, verdicts, strict=True))
    for i, j, s in scored:
        match = s >= hi or (s > lo and tie_verdict[(i, j)] == "match")
        if match:
            uf.union(i, j)
            links.append({"a": _rid(rows[i]), "b": _rid(rows[j]), "score": round(s, 4)})
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        clusters[uf.find(i)].append(i)
    ids = [""] * len(rows)
    for members in clusters.values():
        keys = sorted(_rid(rows[m]) for m in members)
        eid = "ent_" + hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
        for m in members:
            ids[m] = eid
    result.data = df.with_columns(pl.Series(ENTITY_ID, ids, dtype=pl.String))
    result.links = pl.DataFrame(links, schema={"a": pl.String, "b": pl.String, "score": pl.Float64})
    return result


def _rid(row: dict[str, object]) -> str:
    return f"{row.get('_meldr_source_id')}:{row.get('_meldr_record_id')}"


async def _tie_break(
    gateway: LLMGateway | None,
    entity: str,
    fields: Sequence[str],
    a: dict[str, object],
    b: dict[str, object],
    result: Resolution,
) -> str:
    if gateway is None:
        return "non_match"
    payload = {
        "task": "entity_match",
        "entity": entity,
        "fields": list(fields),
        "a": {f: a.get(f) for f in fields},
        "b": {f: b.get(f) for f in fields},
    }
    try:
        res = await gateway.generate(
            [Message(role="user", content=json.dumps(payload, default=str))],
            response_model=SameEntity,
            contains={DataClass.SAMPLES},
            purpose="entity_match",
            prompt_version=PROMPT_VERSION,
            system=load_prompt("entity_match_v1"),
        )
    except PolicyViolation:
        if "tie_break_policy_restricted" not in result.warnings:
            result.warnings.append("tie_break_policy_restricted")
        return "non_match"
    except LLMError:
        return "non_match"
    result.llm_calls += res.attempts
    return "match" if res.parsed.same and res.parsed.confidence >= 0.5 else "non_match"
