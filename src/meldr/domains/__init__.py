"""Bundled canonical entity packs."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def pack_path(name: str = "generic") -> Path:
    """Filesystem path of a bundled domain pack."""
    return Path(str(resources.files(__name__).joinpath(name)))
