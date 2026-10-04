"""YAML parsing with the libyaml C loader when it is available (same results, much faster)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from meldr.core.errors import ConfigError

_Loader: Any = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load(text: str) -> Any:
    """`yaml.safe_load` semantics."""
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass


def read_mapping(path: str | Path, what: str) -> dict[str, Any]:
    """Parse `path`. Errors name the file and the expected content (`what`)."""
    p = Path(path)
    try:
        data = load(p.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"Cannot read {what} {p}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{what.capitalize()} {p} must be a YAML mapping at the top level.")
    return data
