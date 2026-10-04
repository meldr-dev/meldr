"""Flatten nested JSON records into one table of dot paths (plus optional child tables)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl

PARENT_ID = "_meldr_parent_id"


@dataclass
class Flattened:
    table: pl.DataFrame
    children: dict[str, pl.DataFrame] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, str | int | float | bool)


def _flatten_obj(
    obj: Mapping[str, Any],
    prefix: str,
    out: dict[str, Any],
    children: dict[str, list[dict[str, Any]]],
    arrays: Literal["json", "explode"],
    parent_id: str,
    json_paths: set[str],
) -> None:
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            if not value:
                out[path] = None
            _flatten_obj(value, f"{path}.", out, children, arrays, parent_id, json_paths)
        elif isinstance(value, list):
            if all(_is_scalar(v) for v in value):
                out[path] = [None if v is None else str(v) for v in value]
            elif arrays == "explode":
                rows = children.setdefault(path, [])
                for item in value:
                    child: dict[str, Any] = {PARENT_ID: parent_id}
                    if isinstance(item, Mapping):
                        _flatten_obj(item, "", child, {}, "json", parent_id, json_paths)
                    else:
                        child["value"] = item
                    rows.append(child)
            else:
                out[path] = json.dumps(value, sort_keys=True, default=str)
                json_paths.add(path)
        else:
            out[path] = value


def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    keys: dict[str, None] = {}
    for r in rows:
        keys.update(dict.fromkeys(r))
    columns: dict[str, list[Any]] = {k: [r.get(k) for r in rows] for k in keys}
    series = []
    for k, vals in columns.items():
        try:
            series.append(pl.Series(k, vals, strict=True))
        except (TypeError, pl.exceptions.PolarsError, OverflowError):
            series.append(
                pl.Series(k, [None if v is None else _to_str(v) for v in vals], dtype=pl.String)
            )
    return pl.DataFrame(series)


def _to_str(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, default=str)


def flatten(
    records: Sequence[Mapping[str, Any]],
    *,
    record_ids: Sequence[str] | None = None,
    arrays: Literal["json", "explode"] = "json",
    source_id: str = "source",
) -> Flattened:
    """Nested objects -> dot paths, scalar arrays -> list columns, object arrays -> json/explode.

    The union of keys across records defines the shape. Missing keys become nulls.
    """
    ids = list(record_ids) if record_ids is not None else [str(i) for i in range(len(records))]
    rows: list[dict[str, Any]] = []
    child_rows: dict[str, list[dict[str, Any]]] = {}
    json_paths: set[str] = set()
    for rid, rec in zip(ids, records, strict=True):
        out: dict[str, Any] = {}
        _flatten_obj(rec, "", out, child_rows, arrays, rid, json_paths)
        rows.append(out)
    result = Flattened(table=_frame(rows))
    if arrays == "explode":
        result.table = result.table.with_columns(pl.Series(PARENT_ID, ids, dtype=pl.String))
    for path, crow in child_rows.items():
        result.children[f"{source_id}__{path}"] = _frame(crow)
    for p in sorted(json_paths):
        result.warnings.append(f"array_of_objects_serialized:{p}")
    return result
