from __future__ import annotations

import json

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from meldr.core.dtypes import DataType
from meldr.llm.policy import DataClass, DataPolicy
from meldr.profiling.flatten import PARENT_ID, flatten
from meldr.profiling.masking import apply_policy, mask_value, shape_mask
from meldr.profiling.profiler import profile_table


def test_profiler_inference(erp_df: pl.DataFrame) -> None:
    s = profile_table(erp_df, "erp")
    est = s.field("EST_DT")
    assert est.inferred_dtype == DataType.DATE and est.hints["date_formats"] == ["%m/%d/%Y"]
    assert s.field("ACTV_FLG").inferred_dtype == DataType.BOOLEAN
    assert "email" in s.field("CONTACT_EMAIL").patterns
    assert s.field("VNDR_TYP_CD").hints["categorical"]
    assert (
        s.fingerprint == profile_table(erp_df.select(reversed(erp_df.columns)), "erp").fingerprint
    )


def test_leading_zero_codes_stay_strings() -> None:
    s = profile_table(pl.DataFrame({"zip": ["02134", "10001"], "n": ["1", "20"]}), "x")
    assert s.field("zip").inferred_dtype == DataType.STRING
    assert s.field("n").inferred_dtype == DataType.INTEGER


def test_ambiguous_dates_and_numbers() -> None:
    df = pl.DataFrame(
        {
            "d": ["01/02/2020", "03/04/2021", "05/06/2022"],
            "amt": ["€1.234,50", "€2.000,00", "€3,10"],
        }
    )
    s = profile_table(df, "x")
    assert s.field("d").hints.get("ambiguous_date") is True
    assert s.field("amt").inferred_dtype == DataType.FLOAT
    assert s.field("amt").hints["decimal_sep"] == ","


PII = ["ann.lee@example.com", "+1 415 555 0101", "123-45-6789", "4111 1111 1111 1111"]


def test_masked_samples_leave_no_raw_pii() -> None:
    df = pl.DataFrame(
        {
            "contact": PII,
            "full_name": ["Ann Lee", "Bo Chen", "Cy Diaz", "Di Eve"],
            "status": ["A", "I", "A", "A"],
        }
    )
    schema = profile_table(df, "x")
    masked = apply_policy(schema, DataPolicy(allowed={DataClass.SCHEMA, DataClass.MASKED_SAMPLES}))
    dumped = masked.model_dump_json()
    for raw in [*PII, "Ann Lee", "Bo Chen"]:
        assert raw not in dumped
    assert masked.field("status").samples  # non-PII categorical values survive
    schema_only = apply_policy(schema, DataPolicy(allowed={DataClass.SCHEMA}))
    assert all(not f.samples and "top_values" not in f.hints for f in schema_only.fields)


def test_mask_helpers() -> None:
    assert shape_mask("Ann Lee 42") == "Xxx Xxx 99"
    assert mask_value("mail me at a@b.co") == "mail me at <email>"


def test_flatten_nested_and_explode() -> None:
    recs = [
        {"id": 1, "owner": {"email": "a@b.c"}, "tags": ["x", 2], "items": [{"sku": "A"}]},
        {"id": 2, "extra": True},
    ]
    flat = flatten(recs)
    assert set(flat.table.columns) == {"id", "owner.email", "tags", "items", "extra"}
    assert flat.table["tags"].to_list()[0] == ["x", "2"]
    assert flat.warnings == ["array_of_objects_serialized:items"]
    exploded = flatten(recs, arrays="explode", source_id="api")
    child = exploded.children["api__items"]
    assert child[PARENT_ID].to_list() == ["0"] and child["sku"].to_list() == ["A"]


json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-5, 5) | st.text(max_size=4),
    lambda inner: (
        st.lists(inner, max_size=3)
        | st.dictionaries(st.text("abc", min_size=1, max_size=3), inner, max_size=3)
    ),
    max_leaves=10,
)


@given(
    st.lists(
        st.dictionaries(st.text("abcd", min_size=1, max_size=3), json_values, max_size=4),
        min_size=1,
        max_size=5,
    )
)
def test_flatten_never_raises(records: list[dict[str, object]]) -> None:
    flat = flatten(records)
    assert flat.table.height in (0, len(records))
    json.dumps(flat.warnings)
