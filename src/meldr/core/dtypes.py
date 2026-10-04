"""Logical data types and their Polars equivalents."""

from __future__ import annotations

import polars as pl

from meldr._compat import StrEnum


class DataType(StrEnum):
    """Logical data types shared by profiles, canonical fields, and transforms."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    JSON = "json"
    ARRAY = "array"
    BINARY = "binary"
    UNKNOWN = "unknown"


_NUMERIC = {DataType.INTEGER, DataType.FLOAT, DataType.DECIMAL}


def as_text(column: str, dtype: pl.DataType) -> pl.Expr:
    """`column` as text: nested values (lists, structs) as JSON, everything else cast."""
    if isinstance(dtype, pl.List | pl.Array | pl.Struct):
        encoded = (
            pl.struct(pl.col(column).alias("v")).struct.json_encode().str.slice(5).str.head(-1)
        )
        return pl.when(pl.col(column).is_null()).then(None).otherwise(encoded).alias(column)
    return pl.col(column).cast(pl.String, strict=False)


def to_polars(dtype: DataType) -> pl.DataType:
    """Map a logical type to the Polars dtype used in harmonized output.

    DECIMAL maps to Float64 (see DECISIONS.md: the installed Polars does not support Decimal
    casts from strings the same way on all expression paths).
    """
    mapping: dict[DataType, pl.DataType] = {
        DataType.STRING: pl.String(),
        DataType.INTEGER: pl.Int64(),
        DataType.FLOAT: pl.Float64(),
        DataType.DECIMAL: pl.Float64(),
        DataType.BOOLEAN: pl.Boolean(),
        DataType.DATE: pl.Date(),
        DataType.DATETIME: pl.Datetime("us", "UTC"),
        DataType.JSON: pl.String(),
        DataType.ARRAY: pl.List(pl.String()),
        DataType.BINARY: pl.Binary(),
        DataType.UNKNOWN: pl.String(),
    }
    return mapping[dtype]


def from_polars(dtype: pl.DataType) -> DataType:
    """Map a Polars dtype to the closest logical type."""
    if dtype.is_integer():
        return DataType.INTEGER
    if dtype.is_float():
        return DataType.FLOAT
    if isinstance(dtype, pl.Decimal):
        return DataType.DECIMAL
    if isinstance(dtype, pl.Boolean):
        return DataType.BOOLEAN
    if isinstance(dtype, pl.Date):
        return DataType.DATE
    if isinstance(dtype, pl.Datetime):
        return DataType.DATETIME
    if isinstance(dtype, pl.String | pl.Categorical | pl.Enum):
        return DataType.STRING
    if isinstance(dtype, pl.List | pl.Array):
        return DataType.ARRAY
    if isinstance(dtype, pl.Struct):
        return DataType.JSON
    if isinstance(dtype, pl.Binary):
        return DataType.BINARY
    return DataType.UNKNOWN


def type_compat(src: DataType, tgt: DataType) -> float:
    """Type compatibility score from `src` to `tgt`.

    - 1.0: the same type, or a lossless conversion to a wider type.
    - 0.6: parseable (for example, STRING→DATE).
    - 0.0: incompatible.
    """
    if src == tgt:
        return 1.0
    widening = {
        (DataType.INTEGER, DataType.FLOAT),
        (DataType.INTEGER, DataType.DECIMAL),
        (DataType.FLOAT, DataType.DECIMAL),
        (DataType.DECIMAL, DataType.FLOAT),
        (DataType.DATE, DataType.DATETIME),
        (DataType.JSON, DataType.STRING),
        (DataType.UNKNOWN, DataType.STRING),
    }
    if (src, tgt) in widening:
        return 1.0
    if src in (DataType.STRING, DataType.UNKNOWN):
        return 0.6
    if tgt == DataType.STRING and src != DataType.BINARY:
        return 0.6
    if src in _NUMERIC and tgt in _NUMERIC:
        return 0.6
    if src == DataType.INTEGER and tgt == DataType.BOOLEAN:
        return 0.6
    if src == DataType.DATETIME and tgt == DataType.DATE:
        return 0.6
    return 0.0
