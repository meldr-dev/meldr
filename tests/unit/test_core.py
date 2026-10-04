from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, Field

from meldr.core.canonical import CanonicalEntity, SchemaRegistry
from meldr.core.dtypes import DataType, from_polars, to_polars, type_compat
from meldr.core.errors import ConfigError, ConnectorError
from meldr.core.mapping import (
    FieldMapping,
    FieldStatus,
    MappingSpec,
    Origin,
    SpecStatus,
    TransformStep,
)
from meldr.core.schema import FieldProfile, SourceSchema, shape_fingerprint
from meldr.core.types import Modality, RawBatch
from tests.helpers import make_spec


def test_dtype_roundtrip() -> None:
    for dt in (
        DataType.STRING,
        DataType.INTEGER,
        DataType.FLOAT,
        DataType.BOOLEAN,
        DataType.DATE,
        DataType.DATETIME,
        DataType.BINARY,
        DataType.ARRAY,
    ):
        assert from_polars(to_polars(dt)) == dt
    assert from_polars(pl.Struct({"a": pl.Int64})) == DataType.JSON
    assert from_polars(pl.Null()) == DataType.UNKNOWN


def test_type_compat() -> None:
    assert type_compat(DataType.INTEGER, DataType.FLOAT) == 1.0
    assert type_compat(DataType.STRING, DataType.DATE) == 0.6
    assert type_compat(DataType.BOOLEAN, DataType.DATE) == 0.0
    assert type_compat(DataType.DATE, DataType.DATE) == 1.0


def test_rawbatch_requires_exactly_one() -> None:
    with pytest.raises(ConnectorError):
        RawBatch(source_id="s", modality=Modality.TABULAR)
    df = pl.DataFrame({"a": [1]})
    assert RawBatch.from_polars(df, "s", entity_hint="t").source_metadata["entity_hint"] == "t"
    assert RawBatch.from_pandas(pd.DataFrame({"a": [1]}), "s").table is not None
    assert RawBatch.from_arrow(pa.table({"a": [1]}), "s").table is not None
    b = RawBatch.from_dicts([{"x": 1}, {"x": 2}], "api")
    assert b.modality == Modality.JSON and len(b.records) == 2
    assert b.records[0].content_hash != b.records[1].content_hash


def test_from_files(tmp_path: Path) -> None:
    p = tmp_path / "a.md"
    p.write_text("# hi")
    b = RawBatch.from_files([p], "docs")
    assert b.modality == Modality.DOCUMENT and b.records[0].mime_type == "text/markdown"
    with pytest.raises(ConnectorError):
        RawBatch.from_files([tmp_path / "missing.pdf"], "docs")
    with pytest.raises(ConnectorError):
        RawBatch.from_files([], "docs")


@given(
    st.lists(
        st.text(alphabet="abcdefXYZ_.", min_size=1, max_size=8),
        min_size=1,
        max_size=10,
        unique=True,
    )
)
def test_fingerprint_order_and_case_independent(paths: list[str]) -> None:
    assert shape_fingerprint(paths) == shape_fingerprint(list(reversed(paths)))
    assert shape_fingerprint(paths) == shape_fingerprint([p.upper() + " " for p in paths])


def test_fingerprint_changes_with_shape() -> None:
    assert shape_fingerprint(["a", "b"]) != shape_fingerprint(["a", "b", "c"])


def _profile(path: str) -> FieldProfile:
    return FieldProfile(
        path=path,
        name=path,
        dtype=DataType.STRING,
        inferred_dtype=DataType.STRING,
        null_ratio=0,
        distinct_ratio=1,
    )


def test_source_schema_helpers() -> None:
    s = SourceSchema(
        source_id="s", fields=[_profile("a")], row_count=1, fingerprint=shape_fingerprint(["a"])
    )
    assert s.type_signature() == {"a": DataType.STRING}
    assert s.field("a").path == "a"
    with pytest.raises(KeyError):
        s.field("b")


def test_entity_yaml_and_errors(tmp_path: Path) -> None:
    p = tmp_path / "e.yaml"
    p.write_text("name: E\nversion: 1.0.0\nfields: [{name: a, dtype: string}]\nprimary_key: [a]\n")
    e = CanonicalEntity.from_yaml(p)
    assert e.field("a").dtype == DataType.STRING
    p.write_text("name: E\nversion: one\nfields: []\n")
    with pytest.raises(ConfigError, match="semver"):
        CanonicalEntity.from_yaml(p)
    p.write_text("name: E\nversion: 1.0.0\nfields: [{name: a, dtype: string}]\nprimary_key: [b]\n")
    with pytest.raises(ConfigError, match="unknown fields"):
        CanonicalEntity.from_yaml(p)
    p.write_text("name: E\nversion: 1.0.0\nfields: [{name: a, dtype: nope}]\n")
    with pytest.raises(ConfigError):
        CanonicalEntity.from_yaml(p)


def test_entity_from_pydantic() -> None:
    class Vendor(BaseModel):
        """A vendor."""

        model_config = {"json_schema_extra": {"primary_key": ["vendor_id"]}}
        vendor_id: str
        name: str = Field(description="Name", json_schema_extra={"synonyms": ["nm"]})
        revenue: float | None = None
        tags: list[str] = []

    e = CanonicalEntity.from_pydantic(Vendor)
    assert e.name == "Vendor" and e.primary_key == ["vendor_id"]
    assert e.field("vendor_id").required
    assert e.field("revenue").dtype == DataType.FLOAT and not e.field("revenue").required
    assert e.field("name").synonyms == ["nm"]
    assert e.field("tags").dtype == DataType.ARRAY


def test_registry(registry: SchemaRegistry) -> None:
    assert ("Organization", "1.0.0") in registry.list()
    e = registry.get("Organization")
    newer = e.model_copy(update={"version": "1.1.0"})
    registry.register(newer)
    assert registry.get("Organization").version == "1.1.0"
    assert registry.get("Organization", "1.0.0").version == "1.0.0"
    with pytest.raises(ConfigError, match="Unknown entity"):
        registry.get("Nope")
    with pytest.raises(ConfigError, match="different definition"):
        registry.register(e.model_copy(update={"description": "changed"}))
    with pytest.raises(ConfigError):
        registry.load_dir("/does/not/exist")


def test_spec_status_and_yaml() -> None:
    f = FieldMapping(
        target="a",
        sources=["a"],
        steps=[TransformStep(op="trim")],
        confidence=0.9,
        origin=Origin.LLM,
    )
    assert make_spec([f]).status == SpecStatus.DRAFT
    approved = f.model_copy(update={"status": FieldStatus.APPROVED})
    other = f.model_copy(update={"target": "b"})
    assert make_spec([approved, other]).status == SpecStatus.PARTIAL
    spec = make_spec([approved])
    assert spec.status == SpecStatus.APPROVED
    assert MappingSpec.from_yaml(spec.to_yaml()) == spec
    with pytest.raises(ValueError):
        TransformStep.model_validate({"op": "trim", "extra": 1})
