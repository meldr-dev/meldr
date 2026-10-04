"""Versioned prompt files."""

from __future__ import annotations

from functools import cache
from importlib import resources


@cache
def load(name: str) -> str:
    """Return the text of `<name>.md` from this package."""
    return resources.files(__name__).joinpath(f"{name}.md").read_text()
