"""Field-name normalization: tokenize, expand abbreviations, drop noise."""

from __future__ import annotations

import re
from functools import cache
from importlib import resources

import yaml

NOISE = {"tbl", "fld", "col", "src", "the", "of", "a", "an", "f", "c", "x"}
_CAMEL = re.compile(r"([a-z0-9])([A-Z])|([A-Z]+)([A-Z][a-z])")
_SPLIT = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])")


@cache
def abbreviations() -> dict[str, str]:
    """The bundled abbreviation table."""
    text = resources.files("meldr.mapping").joinpath("abbreviations.yaml").read_text()
    data = yaml.safe_load(text) or {}
    return {str(k).casefold(): str(v).casefold() for k, v in data.items()}


def _camel(s: str) -> str:
    return _CAMEL.sub(lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}", s)


@cache
def tokens(name: str) -> tuple[str, ...]:
    """Name tokens: split, casefold, expand, and drop noise.

    The split is on camelCase, snake_case, kebab-case, dots, and digits.
    """
    parts = [p for p in _SPLIT.split(_camel(name)) if p]
    abbr = abbreviations()
    out: list[str] = []
    for p in parts:
        low = p.casefold()
        if low.isdigit():
            continue
        expanded = abbr.get(low, low)
        out.extend(t for t in expanded.split() if t not in NOISE)
    return tuple(out)


def normalize(name: str) -> str:
    """Normalized, space-separated token string."""
    return " ".join(tokens(name))
