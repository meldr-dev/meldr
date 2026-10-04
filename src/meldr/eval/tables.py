"""Table-to-table benchmarks (the Valentine / NYU layout).

Each case is a directory with `source_table.csv`, `target_table.csv`, and `ground_truth.json`
in the format `{"matches": [{"source_column": ..., "target_column": ...}]}`. The target table
becomes a CanonicalEntity with example values in its fields, so the matcher can use
instance-based signals.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from meldr.core.canonical import CanonicalEntity, CanonicalField
from meldr.core.errors import MeldrError
from meldr.profiling.profiler import profile_table

EXAMPLES_PER_FIELD = 10


@dataclass
class TablePair:
    name: str
    source: pl.DataFrame
    entity: CanonicalEntity
    truth: dict[str, list[str]]


def entity_from_table(df: pl.DataFrame, name: str) -> CanonicalEntity:
    """Canonical entity for a table: inferred dtypes plus example values for each column."""
    schema = profile_table(df, name, sample_size=EXAMPLES_PER_FIELD)
    return CanonicalEntity(
        name=name,
        version="1.0.0",
        fields=[
            CanonicalField(name=f.path, dtype=f.inferred_dtype, examples=f.samples)
            for f in schema.fields
        ],
    )


def load_pairs(root: str | Path) -> list[TablePair]:
    """Every case directory in `root`, sorted by name."""
    base = Path(root)
    if not base.is_dir():
        raise MeldrError(f"Benchmark directory {base} is not found. Check the benchmark path.")
    pairs = []
    for case in sorted(p for p in base.iterdir() if (p / "ground_truth.json").exists()):
        truth: dict[str, list[str]] = defaultdict(list)
        for m in json.loads((case / "ground_truth.json").read_text()).get("matches", []):
            truth[m["target_column"]].append(m["source_column"])
        pairs.append(
            TablePair(
                name=case.name,
                source=pl.read_csv(case / "source_table.csv", infer_schema=False),
                entity=entity_from_table(
                    pl.read_csv(case / "target_table.csv", infer_schema=False), "Target"
                ),
                truth=dict(truth),
            )
        )
    if not pairs:
        raise MeldrError(f"No benchmark cases (dirs with ground_truth.json) in {base}.")
    return pairs
