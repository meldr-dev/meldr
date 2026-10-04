"""Draft a canonical entity from source profiles (R18). It does not register the draft."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import yaml
from rapidfuzz import fuzz

from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.dtypes import DataType
from meldr.core.schema import FieldProfile, SourceSchema
from meldr.mapping.normalize import normalize
from meldr.profiling.masking import is_pii_name

SAME_FIELD = 0.85


def _snake(name: str) -> str:
    return normalize(name).replace(" ", "_") or name.casefold()


def discover(
    schemas: Sequence[SourceSchema], name: str, *, version: str = "0.1.0"
) -> CanonicalEntity:
    """Cluster fields across sources by normalized name (majority dtype, raw paths as synonyms)."""
    clusters: list[list[tuple[str, FieldProfile]]] = []
    for schema in schemas:
        for f in schema.fields:
            key = normalize(f.path)
            for cluster in clusters:
                if fuzz.token_sort_ratio(key, normalize(cluster[0][1].path)) / 100 >= SAME_FIELD:
                    cluster.append((schema.source_id, f))
                    break
            else:
                clusters.append([(schema.source_id, f)])
    fields: list[CanonicalField] = []
    seen: set[str] = set()
    for cluster in clusters:
        names = Counter(_snake(f.path) for _, f in cluster)
        fname = names.most_common(1)[0][0]
        if fname in seen:
            continue
        seen.add(fname)
        dtype = Counter(f.inferred_dtype for _, f in cluster).most_common(1)[0][0]
        sources = {s for s, _ in cluster}
        required = len(sources) == len(schemas) and all(f.null_ratio < 0.05 for _, f in cluster)
        desc = next((f.description for _, f in cluster if f.description), "")
        synonyms = sorted({f.path for _, f in cluster if f.path != fname})
        fields.append(
            CanonicalField(
                name=fname,
                dtype=dtype if dtype != DataType.UNKNOWN else DataType.STRING,
                description=desc or f"Seen in {len(sources)} source(s)",
                required=required,
                synonyms=synonyms,
                pii=is_pii_name(fname),
            )
        )
    return CanonicalEntity(
        name=name,
        version=version,
        description=f"DRAFT discovered from {len(schemas)} source(s); review before registering.",
        fields=fields,
    )


def entity_to_yaml(entity: CanonicalEntity) -> str:
    return yaml.safe_dump(entity.model_dump(mode="json", exclude_defaults=True), sort_keys=False)
