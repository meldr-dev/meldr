"""Value patterns detected by the profiler (Rust regex syntax, evaluated by Polars)."""

from __future__ import annotations

import polars as pl

PATTERNS: dict[str, str] = {
    "email": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    "url": r"^https?://\S+$",
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    "ipv4": r"^(\d{1,3}\.){3}\d{1,3}$",
    "postal_code_us": r"^\d{5}(-\d{4})?$",
    "phone": r"^\+?\(?\d[\d\s\-().]{6,18}\d$",
    "currency_amount": r"^[-+]?\s?[$€£¥]\s?\d[\d,.\s]*$|^[-+]?\d[\d,.\s]*\s?(USD|EUR|GBP)$",
    "percentage": r"^[-+]?\d+([.,]\d+)?\s?%$",
    "iso_date": r"^\d{4}-\d{2}-\d{2}$",
    "iso_datetime": r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$",
    "integer_like": r"^[-+]?\d+$",
    "float_like": r"^[-+]?\d*[.,]\d+([eE][-+]?\d+)?$",
    "boolean_like": r"(?i)^(true|false|yes|no|y|n|t|f)$",
    "code_like": r"^[A-Z0-9_\-]{1,6}$",
}

PATTERN_THRESHOLD = 0.8

# Patterns that a canonical field implies (or that conflict with it), keyed by name tokens / dtype.
TARGET_PATTERN_HINTS: dict[str, set[str]] = {
    "email": {"email"},
    "phone": {"phone"},
    "url": {"url"},
    "website": {"url"},
    "uuid": {"uuid"},
    "zip": {"postal_code_us"},
    "postal": {"postal_code_us"},
}

# Patterns that are wrong for a target with this name token.
CONFLICTS: dict[str, set[str]] = {
    "email": {"phone", "url", "iso_date", "integer_like"},
    "phone": {"email", "url", "iso_date"},
    "date": {"email", "url", "phone"},
    "name": {"email", "url", "uuid", "iso_date", "integer_like"},
}


def detect(values: pl.Series) -> list[str]:
    """Pattern tags matched by at least 80% of non-null string values."""
    s = values.drop_nulls().cast(pl.String).str.strip_chars()
    s = s.filter(s != "")
    if s.len() == 0:
        return []
    shares = pl.DataFrame({"v": s}).select(
        [pl.col("v").str.contains(p).mean().alias(name) for name, p in PATTERNS.items()]
    )
    row = shares.row(0, named=True)
    tags = [name for name, share in row.items() if share is not None and share >= PATTERN_THRESHOLD]
    if "code_like" in tags and s.n_unique() > 50:
        tags.remove("code_like")
    if "phone" in tags and {"iso_date", "iso_datetime"} & set(tags):
        tags.remove("phone")  # "2023-01-01" also fits the loose phone shape
    return tags
