"""Built-in transform ops. Every op is a pure, null-safe Polars expression builder."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Literal, TypeVar, cast

import polars as pl
from pydantic import Field, field_validator, model_validator

from meldr.core.dtypes import DataType, to_polars
from meldr.core.errors import SpecValidationError
from meldr.transforms import units
from meldr.transforms.registry import OpArgs, OpDef

A = TypeVar("A", bound=OpArgs)


def _op(
    name: str,
    args_model: type[A],
    description: str,
    *,
    arity: Literal["single", "multi"] = "single",
    nulls_intentionally: bool = False,
    output_dtype: DataType | None = None,
) -> Callable[[Callable[[list[pl.Expr], A], pl.Expr]], OpDef]:
    def deco(fn: Callable[[list[pl.Expr], A], pl.Expr]) -> OpDef:
        def build(exprs: list[pl.Expr], args: OpArgs) -> pl.Expr:
            return fn(exprs, cast(A, args))

        return OpDef(
            name=name,
            args_model=args_model,
            arity=arity,
            build=build,
            description=description,
            nulls_intentionally=nulls_intentionally,
            output_dtype=output_dtype,
        )

    return deco


def _s(e: pl.Expr) -> pl.Expr:
    return e.cast(pl.String, strict=False)


def has_tz_directive(fmt: str) -> bool:
    """True when a strftime format has its own UTC offset."""
    return any(d in fmt for d in ("%z", "%:z", "%#z", "%+"))


def validate_regex(pattern: str) -> str:
    """Compile with the linear-time Rust engine of Polars, not with the Python `re` module."""
    try:
        pl.select(pl.lit("").str.contains(pattern))
    except Exception as e:
        raise ValueError(f"invalid regex pattern: {e}") from e
    return pattern


class NoArgs(OpArgs):
    pass


class CastArgs(OpArgs):
    dtype: DataType


class NullifyArgs(OpArgs):
    values: list[str]
    case_insensitive: bool = True


class DefaultArgs(OpArgs):
    value: str | int | float | bool


NUMBER_SYMBOLS = "$€£¥%"


def clean_number(
    expr: pl.Expr,
    *,
    decimal_sep: str = ".",
    thousands_sep: str = ",",
    strip_chars: str = NUMBER_SYMBOLS,
) -> pl.Expr:
    """Numeric text as Float64, with whitespace and `strip_chars` removed.

    Thousands separators must be between groups of three digits. "1,234.5" parses, but "1,5" is
    null. "1,5" is a misread decimal, so Meldr reports it as lost and does not make it 15.
    """
    s = expr.cast(pl.String).str.replace_all(r"\s", "")
    for ch in strip_chars:
        s = s.str.replace_all(ch, "", literal=True)
    if thousands_sep:
        sep, dec = re.escape(thousands_sep), re.escape(decimal_sep)
        grouped = s.str.contains(rf"^[+-]?\d{{1,3}}({sep}\d{{3}})+({dec}\d*)?$")
        s = pl.when(grouped | ~s.str.contains(thousands_sep, literal=True)).then(
            s.str.replace_all(thousands_sep, "", literal=True)
        )
    if decimal_sep != ".":
        s = s.str.replace_all(decimal_sep, ".", literal=True)
    return s.cast(pl.Float64, strict=False)


class ParseNumberArgs(OpArgs):
    decimal_sep: str = "."
    thousands_sep: str = ","
    strip_chars: str = NUMBER_SYMBOLS

    @model_validator(mode="after")
    def _seps(self) -> ParseNumberArgs:
        if self.decimal_sep == self.thousands_sep:
            raise ValueError("decimal_sep and thousands_sep must differ")
        return self


class ScaleArgs(OpArgs):
    factor: float
    offset: float = 0.0


class RoundArgs(OpArgs):
    decimals: int = Field(default=0, ge=0, le=12)


class PadArgs(OpArgs):
    width: int = Field(ge=1, le=64)
    fill_char: str = Field(default="0", min_length=1, max_length=1)
    side: Literal["left", "right"] = "left"


class FromEpochArgs(OpArgs):
    unit: Literal["s", "ms", "us", "d"] = "s"
    origin: date = date(1970, 1, 1)  # Excel serial dates: unit "d", origin 1899-12-30


class MapContainsArgs(OpArgs):
    rules: dict[str, str] = Field(min_length=1)  # keyword -> value (the first match wins)
    default: str | None = None
    case_insensitive: bool = True


class UnitConvertArgs(OpArgs):
    from_unit: str
    to_unit: str

    @model_validator(mode="after")
    def _known(self) -> UnitConvertArgs:
        f, t = units.canonical_unit(self.from_unit), units.canonical_unit(self.to_unit)
        if f in units.TEMPERATURE or t in units.TEMPERATURE:
            if not (f in units.TEMPERATURE and t in units.TEMPERATURE):
                raise ValueError("temperature converts only between C, F, and K")
            return self
        try:
            units.linear_factor(self.from_unit, self.to_unit)
        except SpecValidationError as e:
            raise ValueError(str(e)) from e
        return self


class DateArgs(OpArgs):
    formats: list[str] = Field(min_length=1)


class DatetimeArgs(OpArgs):
    formats: list[str] = Field(min_length=1)
    assume_tz: str = "UTC"


class BoolArgs(OpArgs):
    true_values: list[str] = Field(default_factory=lambda: ["true", "t", "yes", "y", "1"])
    false_values: list[str] = Field(default_factory=lambda: ["false", "f", "no", "n", "0"])


class MapValuesArgs(OpArgs):
    mapping: dict[str, str | None]
    default: str | None = None
    case_insensitive: bool = True


class _RegexArgs(OpArgs):
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        return validate_regex(v)


class RegexExtractArgs(_RegexArgs):
    group: int = Field(default=1, ge=0)
    keep_unmatched: bool = False


class RegexReplaceArgs(_RegexArgs):
    replacement: str


class SplitPartArgs(OpArgs):
    sep: str = Field(min_length=1)
    index: int


class ConcatArgs(OpArgs):
    sep: str = " "


class TemplateArgs(OpArgs):
    template: str

    def check_sources(self, n: int) -> None:
        count = self.template.count("{}")
        if count != n:
            raise SpecValidationError(
                f"template has {count} '{{}}' placeholders but the mapping has {n} sources"
            )


@_op("cast", CastArgs, "Non-strict cast to a logical dtype; failures become null.")
def _cast(e: list[pl.Expr], a: CastArgs) -> pl.Expr:
    return e[0].cast(to_polars(a.dtype), strict=False)


@_op("trim", NoArgs, "Strip leading/trailing whitespace.", output_dtype=DataType.STRING)
def _trim(e: list[pl.Expr], a: NoArgs) -> pl.Expr:
    return _s(e[0]).str.strip_chars()


@_op("lower", NoArgs, "Lowercase.", output_dtype=DataType.STRING)
def _lower(e: list[pl.Expr], a: NoArgs) -> pl.Expr:
    return _s(e[0]).str.to_lowercase()


@_op("upper", NoArgs, "Uppercase.", output_dtype=DataType.STRING)
def _upper(e: list[pl.Expr], a: NoArgs) -> pl.Expr:
    return _s(e[0]).str.to_uppercase()


@_op("title", NoArgs, "Title Case.", output_dtype=DataType.STRING)
def _title(e: list[pl.Expr], a: NoArgs) -> pl.Expr:
    return _s(e[0]).str.to_titlecase()


@_op(
    "nullify",
    NullifyArgs,
    "Turn sentinel values (e.g. 'N/A', '-', '') into null.",
    nulls_intentionally=True,
)
def _nullify(e: list[pl.Expr], a: NullifyArgs) -> pl.Expr:
    key = _s(e[0]).str.strip_chars()
    values = [v.strip() for v in a.values]
    if a.case_insensitive:
        key = key.str.to_lowercase()
        values = [v.lower() for v in values]
    return pl.when(key.is_in(values)).then(None).otherwise(e[0])


@_op("default", DefaultArgs, "Fill nulls with a constant.")
def _default(e: list[pl.Expr], a: DefaultArgs) -> pl.Expr:
    return e[0].fill_null(pl.lit(a.value))


@_op(
    "parse_number",
    ParseNumberArgs,
    "Parse a numeric string (currency symbols, thousands separators) to float.",
    output_dtype=DataType.FLOAT,
)
def _parse_number(e: list[pl.Expr], a: ParseNumberArgs) -> pl.Expr:
    return clean_number(
        e[0], decimal_sep=a.decimal_sep, thousands_sep=a.thousands_sep, strip_chars=a.strip_chars
    )


@_op(
    "scale",
    ScaleArgs,
    "Linear formula factor * x + offset (e.g. cents to dollars: factor 0.01).",
    output_dtype=DataType.FLOAT,
)
def _scale(e: list[pl.Expr], a: ScaleArgs) -> pl.Expr:
    return e[0].cast(pl.Float64, strict=False) * a.factor + a.offset


@_op("round", RoundArgs, "Round a number to `decimals` places.", output_dtype=DataType.FLOAT)
def _round(e: list[pl.Expr], a: RoundArgs) -> pl.Expr:
    return e[0].cast(pl.Float64, strict=False).round(a.decimals)


@_op(
    "pad",
    PadArgs,
    "Pad to a fixed width (e.g. zero-padded codes: 42 -> 0042).",
    output_dtype=DataType.STRING,
)
def _pad(e: list[pl.Expr], a: PadArgs) -> pl.Expr:
    s = _s(e[0]).str.strip_chars()
    if a.side == "left":
        return s.str.pad_start(a.width, a.fill_char)
    return s.str.pad_end(a.width, a.fill_char)


_MS_PER_UNIT = {"s": 1_000, "ms": 1, "us": 0.001, "d": 86_400_000}


@_op(
    "from_epoch",
    FromEpochArgs,
    "Count of `unit`s since `origin` -> UTC datetime (Unix time; Excel dates: d, 1899-12-30).",
    output_dtype=DataType.DATETIME,
)
def _from_epoch(e: list[pl.Expr], a: FromEpochArgs) -> pl.Expr:
    ms = (e[0].cast(pl.Float64, strict=False) * _MS_PER_UNIT[a.unit]).round(0).cast(pl.Int64)
    origin = datetime(a.origin.year, a.origin.month, a.origin.day)
    return (pl.lit(origin) + pl.duration(milliseconds=ms)).dt.replace_time_zone("UTC")


@_op(
    "map_contains",
    MapContainsArgs,
    "Map by keyword: the first rule whose substring occurs in the value wins, else `default`.",
    output_dtype=DataType.STRING,
)
def _map_contains(e: list[pl.Expr], a: MapContainsArgs) -> pl.Expr:
    text = _s(e[0])
    if a.case_insensitive:
        text = text.str.to_lowercase()
    out: pl.Expr = pl.lit(a.default, dtype=pl.String)
    for needle, value in reversed(list(a.rules.items())):
        key = needle.lower() if a.case_insensitive else needle
        out = pl.when(text.str.contains(key, literal=True)).then(pl.lit(value)).otherwise(out)
    return pl.when(e[0].is_null()).then(pl.lit(None, dtype=pl.String)).otherwise(out)


@_op(
    "unit_convert",
    UnitConvertArgs,
    "Convert between units of length, area, volume, mass, duration, or temperature (C/F/K).",
    output_dtype=DataType.FLOAT,
)
def _unit_convert(e: list[pl.Expr], a: UnitConvertArgs) -> pl.Expr:
    x = e[0].cast(pl.Float64, strict=False)
    f, t = units.canonical_unit(a.from_unit), units.canonical_unit(a.to_unit)
    if f in units.TEMPERATURE:
        kelvin = {"C": x + 273.15, "F": (x - 32) * 5 / 9 + 273.15, "K": x}[f]
        return {"C": kelvin - 273.15, "F": (kelvin - 273.15) * 9 / 5 + 32, "K": kelvin}[t]
    return x * units.linear_factor(f, t)


@_op(
    "parse_date",
    DateArgs,
    "Parse a date string; the first strftime format that parses wins.",
    output_dtype=DataType.DATE,
)
def _parse_date(e: list[pl.Expr], a: DateArgs) -> pl.Expr:
    s = _s(e[0]).str.strip_chars()
    return pl.coalesce([s.str.strptime(pl.Date, f, strict=False) for f in a.formats])


@_op(
    "parse_datetime",
    DatetimeArgs,
    "Parse a datetime string with strftime formats, assume a timezone if none, output UTC.",
    output_dtype=DataType.DATETIME,
)
def _parse_datetime(e: list[pl.Expr], a: DatetimeArgs) -> pl.Expr:
    s = _s(e[0]).str.strip_chars()
    parts: list[pl.Expr] = []
    for f in a.formats:
        if has_tz_directive(f):
            parsed = s.str.strptime(pl.Datetime("us", "UTC"), f, strict=False)
        else:
            parsed = (
                s.str.strptime(pl.Datetime("us"), f, strict=False)
                .dt.replace_time_zone(a.assume_tz, ambiguous="null", non_existent="null")
                .dt.convert_time_zone("UTC")
            )
        parts.append(parsed)
    return pl.coalesce(parts)


@_op("parse_bool", BoolArgs, "Case-insensitive boolean parsing.", output_dtype=DataType.BOOLEAN)
def _parse_bool(e: list[pl.Expr], a: BoolArgs) -> pl.Expr:
    key = _s(e[0]).str.strip_chars().str.to_lowercase()
    return (
        pl.when(key.is_in([v.lower() for v in a.true_values]))
        .then(pl.lit(True))
        .when(key.is_in([v.lower() for v in a.false_values]))
        .then(pl.lit(False))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
    )


@_op(
    "map_values",
    MapValuesArgs,
    "Map source codes to canonical vocabulary values; unmapped values become `default`.",
    output_dtype=DataType.STRING,
)
def _map_values(e: list[pl.Expr], a: MapValuesArgs) -> pl.Expr:
    key = _s(e[0]).str.strip_chars()
    mapping: dict[str, Any] = dict(a.mapping)
    if a.case_insensitive:
        key = key.str.to_lowercase()
        mapping = {k.strip().lower(): v for k, v in mapping.items()}
    else:
        mapping = {k.strip(): v for k, v in mapping.items()}
    mapped = key.replace_strict(
        list(mapping.keys()),
        list(mapping.values()),
        default=pl.lit(a.default, dtype=pl.String),
        return_dtype=pl.String,
    )
    return pl.when(e[0].is_null()).then(pl.lit(None, dtype=pl.String)).otherwise(mapped)


@_op(
    "regex_extract",
    RegexExtractArgs,
    "Extract a capture group with a regex.",
    output_dtype=DataType.STRING,
)
def _regex_extract(e: list[pl.Expr], a: RegexExtractArgs) -> pl.Expr:
    extracted = _s(e[0]).str.extract(a.pattern, a.group)
    return pl.coalesce(extracted, _s(e[0])) if a.keep_unmatched else extracted


@_op(
    "regex_replace",
    RegexReplaceArgs,
    "Replace all regex matches ($1 for groups).",
    output_dtype=DataType.STRING,
)
def _regex_replace(e: list[pl.Expr], a: RegexReplaceArgs) -> pl.Expr:
    return _s(e[0]).str.replace_all(a.pattern, a.replacement)


@_op(
    "split_part",
    SplitPartArgs,
    "Split on `sep` and take part `index` (0-based; negative from end).",
    output_dtype=DataType.STRING,
)
def _split_part(e: list[pl.Expr], a: SplitPartArgs) -> pl.Expr:
    return _s(e[0]).str.split(a.sep, inclusive=False).list.get(a.index, null_on_oob=True)


@_op(
    "concat",
    ConcatArgs,
    "Join several sources with `sep`, skipping nulls.",
    arity="multi",
    output_dtype=DataType.STRING,
)
def _concat(e: list[pl.Expr], a: ConcatArgs) -> pl.Expr:
    parts = [_s(x) for x in e]
    joined = pl.concat_str(parts, separator=a.sep, ignore_nulls=True)
    return pl.when(pl.all_horizontal([x.is_null() for x in e])).then(None).otherwise(joined)


@_op("coalesce", NoArgs, "First non-null of several sources.", arity="multi")
def _coalesce(e: list[pl.Expr], a: NoArgs) -> pl.Expr:
    return pl.coalesce(e)


@_op(
    "template",
    TemplateArgs,
    "Format sources into a string with '{}' placeholders in `sources` order.",
    arity="multi",
    output_dtype=DataType.STRING,
)
def _template(e: list[pl.Expr], a: TemplateArgs) -> pl.Expr:
    return pl.format(a.template, *[_s(x) for x in e])


BUILTIN_OPS: list[OpDef] = [
    _cast,
    _trim,
    _lower,
    _upper,
    _title,
    _nullify,
    _default,
    _parse_number,
    _scale,
    _round,
    _pad,
    _from_epoch,
    _unit_convert,
    _parse_date,
    _parse_datetime,
    _parse_bool,
    _map_values,
    _map_contains,
    _regex_extract,
    _regex_replace,
    _split_part,
    _concat,
    _coalesce,
    _template,
]
