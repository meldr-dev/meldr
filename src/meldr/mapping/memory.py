"""Learn from approved mappings.

A column approved for a target one time is a candidate for that target everywhere.
"""

from __future__ import annotations

from meldr.core.canonical import CanonicalEntity
from meldr.core.schema import SourceSchema
from meldr.mapping.candidates import Candidate
from meldr.mapping.normalize import normalize
from meldr.mapping.store import MappingStore

LEARNED_SCORE = 0.95


class ApprovedMappingMatcher:
    """A matcher (see MappingEngine.matchers) backed by the latest approved specs in a store.

    The matcher compares normalized column names. An approval of `VNDR_NM` for `legal_name` in
    one source makes `vndr_nm` a top candidate for `legal_name` in the next one.
    """

    def __init__(self, store: MappingStore) -> None:
        self.store = store

    def learned(self, entity: str) -> dict[str, set[str]]:
        """normalized source name -> targets it was approved for."""
        out: dict[str, set[str]] = {}
        for spec_id, version, _ in self.store.list():
            spec = self.store.get(spec_id, version)
            if spec.target_entity != entity:
                continue
            for f in spec.approved_fields():
                if len(f.sources) == 1:
                    out.setdefault(normalize(f.sources[0]), set()).add(f.target)
        return out

    def __call__(self, schema: SourceSchema, entity: CanonicalEntity) -> dict[str, list[Candidate]]:
        learned = self.learned(entity.name)
        out: dict[str, list[Candidate]] = {}
        for s in schema.fields:
            for target in learned.get(normalize(s.path), ()):
                out.setdefault(target, []).append(
                    Candidate(path=s.path, score=LEARNED_SCORE, signals={"memory": 1.0})
                )
        return out
