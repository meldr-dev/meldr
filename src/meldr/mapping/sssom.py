"""Export a MappingSpec as an SSSOM/TSV mapping set (https://mapping-commons.github.io/sssom/).

Each source column -> target field pair becomes one row. Single-source mappings use
`skos:exactMatch`. Parts of a multi-source mapping (concat, coalesce, template) use
`skos:relatedMatch`. The `comment` column summarizes the transform steps, so the spec YAML stays
the executable source of truth. The SSSOM file is for exchange and review.
"""

from __future__ import annotations

import csv
import io

import yaml

from meldr.core.mapping import MappingSpec, Origin

COLUMNS = [
    "subject_id",
    "subject_label",
    "predicate_id",
    "object_id",
    "object_label",
    "mapping_justification",
    "confidence",
    "author_label",
    "mapping_tool",
    "comment",
]
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"


def to_sssom(
    spec: MappingSpec,
    *,
    base_iri: str = "urn:meldr:",
    license: str = CC0,
    include_proposed: bool = False,
) -> str:
    """SSSOM/TSV text: a commented YAML metadata block, then one row for each source -> target."""
    source_prefix, target_prefix = f"src_{spec.source_id}", spec.target_entity
    metadata = {
        "curie_map": {
            source_prefix: f"{base_iri}source:{spec.source_id}:",
            target_prefix: f"{base_iri}entity:{spec.target_entity}:",
            "semapv": "https://w3id.org/semapv/vocab/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
        },
        "mapping_set_id": f"{base_iri}spec:{spec.spec_id}:v{spec.version}",
        "mapping_set_version": str(spec.version),
        "license": license,
        "mapping_date": spec.updated_at.date().isoformat(),
        "mapping_tool": "meldr",
    }
    header = "".join(
        f"#{line}\n" for line in yaml.safe_dump(metadata, sort_keys=False).splitlines()
    )
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    for f in spec.runnable_fields(include_proposed=include_proposed):
        predicate = "skos:exactMatch" if len(f.sources) == 1 else "skos:relatedMatch"
        steps = " -> ".join(s.op for s in f.steps) or "copy"
        comment = f"steps: {steps}; status: {f.status}"
        for source in f.sources:
            writer.writerow(
                [
                    f"{source_prefix}:{source}",
                    source,
                    predicate,
                    f"{target_prefix}:{f.target}",
                    f.target,
                    "semapv:ManualMappingCuration"
                    if f.origin == Origin.HUMAN
                    else "semapv:CompositeMatching",
                    f"{f.confidence:.4f}",
                    spec.approved_by or "",
                    "meldr",
                    comment,
                ]
            )
    return header + buf.getvalue()
