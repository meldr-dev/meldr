"""Profile a table into a SourceSchema (types, patterns, hints, samples, fingerprint)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from meldr.core.dtypes import DataType, as_text, from_polars
from meldr.core.schema import FieldProfile, SourceSchema, shape_fingerprint
from meldr.profiling import dates, patterns
from meldr.profiling.masking import MAX_SAMPLE_LEN
from meldr.transforms.ops import clean_number

INFER_SUCCESS = dates.SUCCESS  # share of values that must parse before the profiler infers a type
CATEGORICAL_MAX_DISTINCT = 50
SAMPLE_SEED = 7  # fixed so the same table always yields the same profile
CATEGORICAL_MAX_RATIO = 0.05
CATEGORICAL_SMALL_RATIO = 0.5
SENTINELS = {"n/a", "na", "-", "--", "null", "none", "nil", "?", "unknown", "#n/a"}
BOOL_WORDS_TRUE = {"true", "t", "yes", "y"}
BOOL_WORDS_FALSE = {"false", "f", "no", "n"}
EMBEDDED_NUMBER = r"^[^\d-]*\d+(\.\d+)?[^\d]*$"  # one unsigned number inside a label
EU_NUMBER = r"^[-+]?[$€£¥]?\s?\d{1,3}((\.\d{3})+(,\d+)?|,\d+)\s?[$€£¥]?$"


def _numbers(values: pl.Series, **seps: str) -> pl.Series:
    return values.to_frame("v").select(clean_number(pl.col("v"), **seps)).to_series()


def _mean(s: pl.Series) -> float:
    value = s.mean()
    return float(value) if isinstance(value, int | float) else 0.0


def _share(parsed: pl.Series, total: int) -> float:
    return 1.0 - parsed.null_count() / max(total, 1)


def _string_inference(s: pl.Series, hints: dict[str, Any]) -> DataType:
    """Infer the semantic type of a string sample and record parsing hints."""
    values = s.drop_nulls().str.strip_chars()
    values = values.filter((values != "") & ~values.str.to_lowercase().is_in(list(SENTINELS)))
    n = values.len()
    if n == 0:
        return DataType.STRING
    lowered = set(values.str.to_lowercase().unique().to_list())
    if lowered <= (BOOL_WORDS_TRUE | BOOL_WORDS_FALSE) and len(lowered) <= 4:
        return DataType.BOOLEAN
    if values.str.contains(r"^0\d+$").any():
        return DataType.STRING  # starts with zero: an identifier (ZIP, code), not a number
    if _share(values.cast(pl.Int64, strict=False), n) >= INFER_SUCCESS:
        date_fmts, _, _ = dates.detect_formats(values)
        if date_fmts:
            hints["date_formats"] = date_fmts
        return DataType.INTEGER
    date_fmts, dt_fmts, ambiguous = dates.detect_formats(values)
    if dt_fmts:
        hints["datetime_formats"] = dt_fmts
        return DataType.DATETIME
    if date_fmts:
        hints["date_formats"] = date_fmts
        if ambiguous:
            hints["ambiguous_date"] = True
        return DataType.DATE
    if _mean(values.str.contains(EU_NUMBER)) >= INFER_SUCCESS:
        eu = _numbers(values, decimal_sep=",", thousands_sep=".")
        if _share(eu, n) >= INFER_SUCCESS:
            hints.update(decimal_sep=",", thousands_sep=".")
            _currency_hint(values, hints)
            return DataType.FLOAT
    if _share(_numbers(values), n) >= INFER_SUCCESS:
        hints.update(decimal_sep=".", thousands_sep=",")
        _currency_hint(values, hints)
        return DataType.FLOAT
    if _mean(values.str.contains(EMBEDDED_NUMBER)) >= INFER_SUCCESS:
        hints["embedded_number"] = True
    return DataType.STRING


def _currency_hint(values: pl.Series, hints: dict[str, Any]) -> None:
    symbols = [sym for sym in "$€£¥" if values.str.contains(sym, literal=True).any()]
    if symbols:
        hints["currency_symbols"] = symbols
    if values.str.contains("%", literal=True).any():
        hints["percent"] = True


def _stats(col: pl.Series, dtype: DataType) -> dict[str, float]:
    if dtype in (DataType.INTEGER, DataType.FLOAT, DataType.DECIMAL):
        values, names = col.cast(pl.Float64, strict=False), ("min", "max", "mean")
    elif dtype == DataType.STRING:
        values, names = col.cast(pl.String).str.len_chars(), ("min_len", "max_len", "avg_len")
    else:
        return {}
    values = values.drop_nulls()
    if values.len() == 0:
        return {}
    v = pl.col("v")
    row = values.to_frame("v").select(v.min(), v.max().alias("max"), v.mean().alias("mean")).row(0)
    return {name: float(x) for name, x in zip(names, row, strict=True)}


def source_columns(df: pl.DataFrame) -> list[str]:
    """Columns that come from the source (not the lineage `_meldr_*` columns)."""
    return [c for c in df.columns if not c.startswith("_meldr_")]


def most_frequent(s: pl.Series, n: int) -> list[tuple[str, int]]:
    """Top `n` non-null values as strings with counts, most frequent first (ties by value)."""
    counts = (
        s.to_frame("v")
        .select(as_text("v", s.dtype))
        .drop_nulls()
        .group_by("v")
        .len()
        .sort(["len", "v"], descending=[True, False])
        .head(n)
    )
    return [(str(v), int(c)) for v, c in counts.iter_rows()]


def profile_table(
    df: pl.DataFrame,
    source_id: str,
    *,
    entity_hint: str | None = None,
    descriptions: Mapping[str, str] | None = None,
    sample_rows: int = 10_000,
    sample_size: int = 20,
) -> SourceSchema:
    """Profile `df`. Samples are raw, and `masking.apply_policy` masks them for LLM payloads."""
    descriptions = descriptions or {}
    total = df.height
    sample = df.sample(n=sample_rows, seed=SAMPLE_SEED) if total > sample_rows else df
    columns = source_columns(df)
    if columns:
        approx = total > 1_000_000
        counts = df.select(
            [pl.col(c).null_count().alias(f"n:{c}") for c in columns]
            + [
                (pl.col(c).approx_n_unique() if approx else pl.col(c).n_unique()).alias(f"d:{c}")
                for c in columns
            ]
        ).row(0, named=True)
    else:
        counts = {}

    fields: list[FieldProfile] = []
    for c in columns:
        physical = from_polars(df.schema[c])
        col = sample[c]
        nulls = int(counts[f"n:{c}"])
        non_null = total - nulls
        distinct = int(counts[f"d:{c}"]) - (1 if nulls else 0)
        hints: dict[str, Any] = {}
        tags: list[str] = []
        inferred = physical
        if physical == DataType.STRING:
            tags = patterns.detect(col)
            inferred = _string_inference(col, hints)
            lowered = col.drop_nulls().str.strip_chars().str.to_lowercase()
            seen = sorted(set(lowered.unique().to_list()) & (SENTINELS | {""}))
            if seen:
                hints["sentinels"] = seen
        elif physical == DataType.INTEGER:
            date_fmts, _, _ = dates.detect_formats(col.cast(pl.String))
            if date_fmts:
                hints["date_formats"] = date_fmts
        if physical in (DataType.STRING, DataType.BOOLEAN, DataType.INTEGER) and non_null > 0:
            ratio = distinct / max(non_null, 1)
            small = distinct <= CATEGORICAL_MAX_DISTINCT and ratio <= CATEGORICAL_SMALL_RATIO
            if small or ratio <= CATEGORICAL_MAX_RATIO or "code_like" in tags:
                hints["categorical"] = True
            if hints.get("categorical") or distinct <= CATEGORICAL_MAX_DISTINCT:
                hints["top_values"] = [
                    {"value": v, "count": n}
                    for v, n in most_frequent(df[c], CATEGORICAL_MAX_DISTINCT)
                ]
        samples = (
            [v[:MAX_SAMPLE_LEN] for v, _ in most_frequent(col, sample_size)]
            if physical != DataType.BINARY
            else []
        )
        if inferred in (DataType.INTEGER, DataType.FLOAT) and "date_formats" not in hints:
            epoch = dates.epoch_hint(samples)
            if epoch is not None:
                hints["epoch"] = epoch
        fields.append(
            FieldProfile(
                path=c,
                name=c.rsplit(".", 1)[-1],
                dtype=physical,
                inferred_dtype=inferred,
                null_ratio=nulls / total if total else 1.0,
                distinct_ratio=distinct / non_null if non_null else 0.0,
                patterns=tags,
                samples=samples,
                stats=_stats(col, physical),
                hints=hints,
                description=descriptions.get(c),
            )
        )
    return SourceSchema(
        source_id=source_id,
        entity_hint=entity_hint,
        fields=fields,
        row_count=total,
        fingerprint=shape_fingerprint(columns),
    )
