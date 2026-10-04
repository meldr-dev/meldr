"""Content segments (documents, media) and extracted records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from meldr.core.types import Modality


class Locator(BaseModel):
    """Where a segment lives in its source: page, sheet, time window, or character span."""

    model_config = ConfigDict(extra="forbid")

    page: int | None = None
    sheet: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    char_start: int | None = None
    char_end: int | None = None


def segment_id(source_id: str, record_id: str, index: int) -> str:
    """Unique across sources, so one vector collection can hold every source's segments."""
    return f"{source_id}:{record_id}:{index}"


class ContentSegment(BaseModel):
    """A chunk of text from a document or media asset."""

    segment_id: str
    source_id: str
    record_id: str
    modality: Modality
    text: str
    locator: Locator
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """A quote that supports one extracted value."""

    segment_id: str
    quote: str = Field(max_length=300)
    verified: bool = False


class Relation(BaseModel):
    """A permitted relationship found in source content."""

    model_config = ConfigDict(extra="forbid")

    subject_id: str
    predicate: str
    object_id: str
    evidence: Evidence | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedRecord(BaseModel):
    """One entity instance extracted from unstructured content."""

    entity: str
    entity_version: str
    record_id: str
    values: dict[str, Any]
    evidence: dict[str, Evidence]
    confidence: dict[str, float]
    relations: list[Relation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
