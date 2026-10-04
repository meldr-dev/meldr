"""Interop standards (LinkML, SSSOM) and the transform ops the NYU benchmark needed."""

from __future__ import annotations

import csv
import io
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from meldr._compat import UTC
from meldr.api import apply_mapping, propose_mapping
from meldr.core.canonical import CanonicalEntity, CanonicalField, SchemaRegistry
from meldr.core.dtypes import DataType
from meldr.core.errors import ConfigError
from meldr.core.mapping import FieldMapping, Origin, TransformStep
from meldr.mapping import review
from meldr.mapping.sssom import to_sssom
from meldr.profiling.dates import epoch_hint
from meldr.transforms.registry import default_registry

REG = default_registry()


def run(values: list[Any], steps: list[dict[str, Any]]) -> list[Any]:
    m = FieldMapping(
        target="t",
        sources=["a"],
        steps=[TransformStep(**s) for s in steps],
        confidence=1,
        origin=Origin.HUMAN,
    )
    expr = pl.col("a")
    for step, args in zip(m.steps, REG.validate(m), strict=True):
        expr = REG.get(step.op).build([expr], args)
    return pl.DataFrame({"a": values}, strict=False).select(expr.alias("t"))["t"].to_list()


def test_new_ops() -> None:
    assert run(["42", "7", None], [{"op": "pad", "args": {"width": 4}}]) == ["0042", "0007", None]
    assert run(
        [45000.0, None], [{"op": "from_epoch", "args": {"unit": "d", "origin": "1899-12-30"}}]
    ) == [datetime(2023, 3, 15, tzinfo=UTC), None]
    assert run([0, 1_700_000_000], [{"op": "from_epoch"}])[1] == datetime(
        2023, 11, 14, 22, 13, 20, tzinfo=UTC
    )
    assert run([1.26, None], [{"op": "round", "args": {"decimals": 1}}]) == [1.3, None]
    assert run([0.0, 100.0], [{"op": "scale", "args": {"factor": 1.8, "offset": 32}}]) == [
        32.0,
        212.0,
    ]
    rules = {"rules": {"bronze": "Bronze", "steel": "Steel"}, "default": "Other"}
    assert run(
        ["Cast BRONZE", "stainless steel", "wood", None], [{"op": "map_contains", "args": rules}]
    ) == ["Bronze", "Steel", "Other", None]
    keep = {"pattern": r"Sims - (.+)", "keep_unmatched": True}
    assert run(["Sims - Queens", "Bronx"], [{"op": "regex_extract", "args": keep}]) == [
        "Queens",
        "Bronx",
    ]


def test_epoch_numbers_become_dates(org: CanonicalEntity) -> None:
    assert epoch_hint(["45000", "45123.5"]) == {"unit": "d", "origin": "1899-12-30"}
    assert epoch_hint(["1700000000"]) == {"unit": "s", "origin": "1970-01-01"}
    assert epoch_hint(["12", "x"]) is None
    df = pl.DataFrame({"org_id": ["1"], "legal_name": ["Acme"], "founded_on": [36526.0]})
    spec = review.approve(propose_mapping(df, org, source_id="s"), user="t", above=0.0)
    assert apply_mapping(df, spec, org).data["founded_on"].to_list() == [date(2000, 1, 1)]


LINKML: dict[str, Any] = {
    "id": "https://example.org/personinfo",
    "name": "personinfo",
    "version": "2.1.0",
    "default_range": "string",
    "slots": {
        "id": {"identifier": True},
        "name": {"required": True, "aliases": ["full_name"]},
        "age_in_years": {"range": "integer", "minimum_value": 0, "maximum_value": 999},
        "gender": {"range": "GenderType"},
        "aliases": {"multivalued": True},
    },
    "classes": {
        "NamedThing": {"abstract": True, "slots": ["id", "name"]},
        "Person": {
            "is_a": "NamedThing",
            "description": "A person",
            "slots": ["age_in_years", "gender", "aliases"],
            "attributes": {"email": {"pattern": "^\\S+@\\S+$", "examples": [{"value": "a@b.io"}]}},
            "slot_usage": {"age_in_years": {"required": True}},
        },
    },
    "enums": {"GenderType": {"permissible_values": {"female": {}, "male": {}, "nonbinary": {}}}},
}


def test_linkml_import(tmp_path: Path) -> None:
    e = CanonicalEntity.from_linkml(LINKML, "Person")
    assert (e.name, e.version, e.primary_key, e.description) == (
        "Person",
        "2.1.0",
        ["id"],
        "A person",
    )
    fields = {f.name: f for f in e.fields}
    assert list(fields) == ["id", "name", "age_in_years", "gender", "aliases", "email"]
    assert fields["name"].synonyms == ["full_name"] and fields["name"].required
    age = fields["age_in_years"]
    assert (age.dtype, age.minimum, age.maximum, age.required) == (DataType.INTEGER, 0, 999, True)
    assert fields["gender"].vocabulary == ["female", "male", "nonbinary"]
    assert fields["aliases"].dtype == DataType.ARRAY
    assert fields["email"].examples == ["a@b.io"]
    with pytest.raises(ConfigError, match="no class"):
        CanonicalEntity.from_linkml(LINKML, "Nope")
    path = tmp_path / "personinfo.yaml"
    path.write_text(yaml.safe_dump(LINKML))
    registry = SchemaRegistry()
    assert registry.load_linkml(path) == ["Person"]  # abstract NamedThing is skipped
    assert registry.get("Person").version == "2.1.0"


def test_sssom_export(erp_df: pl.DataFrame, org: CanonicalEntity) -> None:
    spec = review.approve(propose_mapping(erp_df, org, source_id="erp"), user="ana", above=0.0)
    text = to_sssom(spec)
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert any(line.startswith("#mapping_set_id:") for line in header)
    assert any(line.startswith("#license:") for line in header)
    rows = list(
        csv.DictReader(
            io.StringIO("\n".join(line for line in text.splitlines() if not line.startswith("#"))),
            delimiter="\t",
        )
    )
    legal = next(r for r in rows if r["object_label"] == "legal_name")
    assert legal["subject_id"] == "src_erp:VNDR_NM" and legal["predicate_id"] == "skos:exactMatch"
    assert legal["mapping_justification"] == "semapv:CompositeMatching"
    assert 0 <= float(legal["confidence"]) <= 1
    assert CanonicalField(name="x", dtype=DataType.STRING).minimum is None
