"""Date/datetime format detection."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from meldr.transforms.ops import has_tz_directive

DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%Y%m%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%b %d %Y",
    "%b %d, %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%d %B %Y",
]

DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%#z",
    "%Y-%m-%dT%H:%M:%S%.f%#z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
]

SUCCESS = 0.95

# (unit, low, high, origin): plausible 1950-2100 ranges for numbers that encode timestamps.
EPOCH_RANGES = [
    ("s", 1e8, 4.1e9, "1970-01-01"),
    ("ms", 1e11, 4.1e12, "1970-01-01"),
    ("d", 18_264, 73_051, "1899-12-30"),  # Excel / Lotus serial dates
]


def epoch_hint(values: Sequence[str]) -> dict[str, str] | None:
    """Unit and origin when every value looks like an epoch timestamp, else None.

    Meldr uses this hint only for date targets, so counts that fall in a range are harmless.
    """
    numbers = []
    for v in values:
        try:
            numbers.append(float(v))
        except ValueError:
            return None
    if not numbers:
        return None
    for unit, low, high, origin in EPOCH_RANGES:
        if all(low <= x <= high for x in numbers):
            return {"unit": unit, "origin": origin}
    return None


def _parsed_share(fmt: str, dtype: pl.DataType) -> pl.Expr:
    return pl.col("v").str.strptime(dtype, fmt, strict=False).is_not_null().mean()  # type: ignore[arg-type]


def detect_formats(values: pl.Series) -> tuple[list[str], list[str], bool]:
    """Return (date_formats, datetime_formats, ambiguous) with ≥95% parse success.

    One Polars query tries every format on the non-empty values.
    """
    s = values.drop_nulls().cast(pl.String).str.strip_chars()
    s = s.filter(s != "")
    if s.len() == 0:
        return [], [], False
    checks = [(f, pl.Date()) for f in DATE_FORMATS] + [
        (f, pl.Datetime("us", "UTC") if has_tz_directive(f) else pl.Datetime("us"))
        for f in DATETIME_FORMATS
    ]
    shares = s.to_frame("v").select(
        _parsed_share(f, dtype).alias(str(i)) for i, (f, dtype) in enumerate(checks)
    )
    ok = [f for (f, _), share in zip(checks, shares.row(0), strict=True) if share >= SUCCESS]
    date_fmts = [f for f in ok if f in DATE_FORMATS]
    dt_fmts = [f for f in ok if f in DATETIME_FORMATS]
    ambiguous = "%m/%d/%Y" in date_fmts and "%d/%m/%Y" in date_fmts
    ambiguous = ambiguous or ("%m-%d-%Y" in date_fmts and "%d-%m-%Y" in date_fmts)
    return date_fmts, dt_fmts, ambiguous
