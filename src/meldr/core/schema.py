"""Source profiles and the shape fingerprint."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from meldr.core.dtypes import DataType


class FieldProfile(BaseModel):
    """Profile of one source field (a column or a flattened JSON path)."""

    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    dtype: DataType
    inferred_dtype: DataType
    null_ratio: float
    distinct_ratio: float
    patterns: list[str] = Field(default_factory=list)
    samples: list[str] = Field(default_factory=list)
    stats: dict[str, float] = Field(default_factory=dict)
    hints: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class SourceSchema(BaseModel):
    """Profile of a full source batch."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    entity_hint: str | None = None
    fields: list[FieldProfile]
    row_count: int
    fingerprint: str

    def type_signature(self) -> dict[str, DataType]:
        """Map path -> inferred dtype, used for drift detection."""
        return {f.path: f.inferred_dtype for f in self.fields}

    def field(self, path: str) -> FieldProfile:
        """Return the profile for `path` or raise KeyError."""
        for f in self.fields:
            if f.path == path:
                return f
        raise KeyError(f"Source field '{path}' not found in schema '{self.source_id}'.")

    def paths(self) -> list[str]:
        """All field paths in order."""
        return [f.path for f in self.fields]


def shape_fingerprint(paths: Iterable[str]) -> str:
    """sha256 of the sorted, casefolded, stripped paths.

    Row content and column order do not change the fingerprint. It deliberately excludes types,
    because with types, null-only batches can thrash the cache. `type_signature()` detects type
    drift separately.
    """
    normalized = sorted({p.strip().casefold() for p in paths})
    return hashlib.sha256("\n".join(normalized).encode()).hexdigest()
