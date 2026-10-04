from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meldr.core.errors import SpecValidationError
from meldr.core.mapping import FieldMapping, Origin, TransformStep
from meldr.transforms.registry import default_registry

REG = default_registry()


def run(values: dict[str, list[Any]], sources: list[str], steps: list[dict[str, Any]]) -> list[Any]:
    m = FieldMapping(
        target="t",
        sources=sources,
        steps=[TransformStep(**s) for s in steps],
        confidence=1,
        origin=Origin.HUMAN,
    )
    args = REG.validate(m)
    exprs = [pl.col(s) for s in sources]
    first = REG.get(m.steps[0].op)
    if first.arity == "multi":
        e, rest = first.build(exprs, args[0]), list(zip(m.steps[1:], args[1:], strict=True))
    else:
        e, rest = exprs[0], list(zip(m.steps, args, strict=True))
    for step, a in rest:
        e = REG.get(step.op).build([e], a)
    return pl.DataFrame(values).select(e.alias("t"))["t"].to_list()


CASES = [
    ("trim", {}, [" a ", None], ["a", None]),
    ("upper", {}, ["a", None], ["A", None]),
    ("title", {}, ["ab cd", None], ["Ab Cd", None]),
    ("nullify", {"values": ["N/A", ""]}, ["n/a", "x", None], [None, "x", None]),
    ("default", {"value": "z"}, [None, "x"], ["z", "x"]),
    ("parse_number", {}, ["$1,234.50", "bad", None], [1234.5, None, None]),
    ("parse_number", {"decimal_sep": ",", "thousands_sep": "."}, ["1.234,5"], [1234.5]),
    ("scale", {"factor": 0.01}, [150, None], [1.5, None]),
    ("unit_convert", {"from_unit": "C", "to_unit": "F"}, [100.0, None], [212.0, None]),
    (
        "parse_date",
        {"formats": ["%m/%d/%Y"]},
        ["03/04/2025", "x", None],
        [date(2025, 3, 4), None, None],
    ),
    ("parse_bool", {}, ["Yes", "n", "maybe", None], [True, False, None, None]),
    (
        "map_values",
        {"mapping": {"A": "active"}, "default": "other"},
        ["a", "q", None],
        ["active", "other", None],
    ),
    ("regex_extract", {"pattern": r"(\d+)"}, ["ab12", "x", None], ["12", None, None]),
    ("regex_replace", {"pattern": r"\s+", "replacement": "-"}, ["a  b", None], ["a-b", None]),
    ("split_part", {"sep": " ", "index": -1}, ["a b c", None], ["c", None]),
]


@pytest.mark.parametrize(("op", "args", "values", "expected"), CASES)
def test_single_ops(op: str, args: dict[str, Any], values: list[Any], expected: list[Any]) -> None:
    assert run({"a": values}, ["a"], [{"op": op, "args": args}]) == expected


def test_unit_convert_linear() -> None:
    assert run(
        {"a": [10.0]}, ["a"], [{"op": "unit_convert", "args": {"from_unit": "ft", "to_unit": "m"}}]
    )[0] == (pytest.approx(3.048))


def test_multi_ops() -> None:
    data = {"a": ["x", None, None], "b": ["y", "z", None]}
    assert run(data, ["a", "b"], [{"op": "concat", "args": {"sep": "-"}}]) == ["x-y", "z", None]
    assert run(data, ["a", "b"], [{"op": "coalesce"}]) == ["x", "z", None]
    assert run(data, ["a", "b"], [{"op": "template", "args": {"template": "{}/{}"}}])[0] == "x/y"


@pytest.mark.parametrize(
    ("sources", "steps", "match"),
    [
        (["a"], [{"op": "nope"}], "Unknown transform op"),
        (["a", "b"], [{"op": "trim"}], "multi-source op"),
        (["a"], [{"op": "trim"}, {"op": "concat"}], "must be the first step"),
        (["a"], [{"op": "scale", "args": {}}], "Invalid args"),
        (["a"], [{"op": "trim", "args": {"x": 1}}], "Invalid args"),
        (["a"], [{"op": "regex_extract", "args": {"pattern": "(("}}], "invalid regex"),
        (
            ["a"],
            [{"op": "unit_convert", "args": {"from_unit": "USD", "to_unit": "EUR"}}],
            "Currency",
        ),
        (["a", "b"], [{"op": "template", "args": {"template": "{}"}}], "placeholders"),
        ([], [], "no sources"),
    ],
)
def test_validation_errors(sources: list[str], steps: list[dict[str, Any]], match: str) -> None:
    with pytest.raises(SpecValidationError, match=match):
        REG.validate(
            FieldMapping(
                target="t",
                sources=sources,
                steps=[TransformStep(**s) for s in steps],
                confidence=1,
                origin=Origin.HUMAN,
            )
        )


@settings(max_examples=40, deadline=None)
@given(st.lists(st.one_of(st.none(), st.text(max_size=12)), min_size=1, max_size=8))
def test_ops_never_raise_on_random_strings(values: list[str | None]) -> None:
    for op, args, _, _ in CASES:
        run({"a": values}, ["a"], [{"op": op, "args": args}])


def test_describe_for_llm_lists_every_op() -> None:
    text = REG.describe_for_llm()
    for name in REG.names():
        assert f"- {name} [" in text
    assert '"enum"' in text  # cast dtype enum is resolved
