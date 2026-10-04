"""Apply a data policy to a SourceSchema before anything leaves the process.

Masking uses fixed, library-owned regexes (not LLM-supplied patterns).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from meldr.core.schema import FieldProfile, SourceSchema
from meldr.llm.policy import DataClass, DataPolicy

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE = re.compile(r"\+?\(?\d[\d\s\-().]{6,18}\d")

PII_NAME_TOKENS = {
    "name",
    "names",
    "first",
    "last",
    "surname",
    "address",
    "addr",
    "street",
    "dob",
    "birth",
    "birthday",
    "ssn",
    "tax",
    "tin",
    "passport",
    "email",
    "mail",
    "phone",
    "tel",
    "mobile",
    "cell",
    "fax",
    "contact",
    "person",
    "owner",
    "iban",
    "account",
    "card",
    "license",
    "ip",
}

MAX_SAMPLE_LEN = 80
_TOKEN = re.compile(r"[a-z]+")


def _name_tokens(path: str) -> set[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", path)
    return set(_TOKEN.findall(spaced.casefold()))


def is_pii_name(path: str) -> bool:
    """True when the field name suggests personal data."""
    return bool(_name_tokens(path) & PII_NAME_TOKENS)


def shape_mask(value: str) -> str:
    """Replace letters with X/x and digits with 9, keep punctuation: 'Ann Lee' -> 'Xxx Xxx'."""
    return "".join(
        ("X" if c.isupper() else "x") if c.isalpha() else ("9" if c.isdigit() else c) for c in value
    )


def mask_value(value: str) -> str:
    """Replace PII-looking substrings with shape tokens (<email>, <phone:10d>, ...)."""
    out = _EMAIL.sub("<email>", value)
    out = _SSN.sub("<ssn>", out)
    out = _CARD.sub(
        lambda m: "<card>" if sum(c.isdigit() for c in m.group()) >= 13 else m.group(), out
    )
    out = _IP.sub("<ip>", out)

    def phone(m: re.Match[str]) -> str:
        digits = sum(c.isdigit() for c in m.group())
        return f"<phone:{digits}d>" if 7 <= digits <= 15 else m.group()

    return _PHONE.sub(phone, out)


def _mask_field(f: FieldProfile, *, pii: bool) -> FieldProfile:
    masker = shape_mask if pii else mask_value
    hints: dict[str, Any] = dict(f.hints)
    if "top_values" in hints:
        hints["top_values"] = [
            {"value": masker(str(tv["value"]))[:MAX_SAMPLE_LEN], "count": tv["count"]}
            for tv in hints["top_values"]
        ]
    return f.model_copy(
        update={"samples": [masker(s)[:MAX_SAMPLE_LEN] for s in f.samples], "hints": hints}
    )


_SCHEMA_STATS = {"avg_len", "min_len", "max_len"}


def _schema_only(f: FieldProfile) -> FieldProfile:
    hints = {k: v for k, v in f.hints.items() if k not in ("top_values", "examples")}
    stats = {k: v for k, v in f.stats.items() if k in _SCHEMA_STATS}
    return f.model_copy(update={"samples": [], "hints": hints, "stats": stats})


def apply_policy(
    schema: SourceSchema, policy: DataPolicy, *, pii_paths: Iterable[str] = ()
) -> SourceSchema:
    """Return a copy of `schema` that contains only the data that `policy` lets Meldr send.

    `pii_paths` are source fields that feed (or can feed) PII targets. `shape_mask` masks them.
    """
    pii_set = set(pii_paths)
    if policy.allows(DataClass.SAMPLES):
        fields = [
            f.model_copy(update={"samples": [v[:MAX_SAMPLE_LEN] for v in f.samples]})
            for f in schema.fields
        ]
    elif policy.allows(DataClass.MASKED_SAMPLES):
        fields = [
            _mask_field(f, pii=f.path in pii_set or is_pii_name(f.path)) for f in schema.fields
        ]
    else:
        fields = [_schema_only(f) for f in schema.fields]
    return schema.model_copy(update={"fields": fields})


def samples_class(policy: DataPolicy) -> DataClass | None:
    """The data class that a payload made with `policy` contains (besides SCHEMA)."""
    if policy.allows(DataClass.SAMPLES):
        return DataClass.SAMPLES
    if policy.allows(DataClass.MASKED_SAMPLES):
        return DataClass.MASKED_SAMPLES
    return None
