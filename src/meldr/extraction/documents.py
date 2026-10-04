"""Turn document records into ContentSegments."""

from __future__ import annotations

from pathlib import Path

from meldr._fsspec import storage_path
from meldr.core.errors import ConnectorError
from meldr.core.segments import ContentSegment
from meldr.core.types import RawRecord
from meldr.extraction import parsers
from meldr.extraction.chunker import chunk


def record_bytes(record: RawRecord) -> bytes:
    if record.content is not None:
        return record.content
    if record.uri:
        path = storage_path(record.uri, feature="document reads")
        if path.is_file():
            return path.read_bytes()
    raise ConnectorError(f"Record {record.record_id} has neither content nor a readable uri.")


def segment_record(
    record: RawRecord, *, target_tokens: int = 800, overlap_tokens: int = 100
) -> tuple[list[ContentSegment], list[str]]:
    """Parse and chunk one document record. Returns (segments, warnings)."""
    name = str(record.metadata.get("filename") or record.uri or record.record_id)
    elements, warnings = parsers.parse(name, record_bytes(record))
    title = next((e.text for e in elements if e.kind == "heading"), Path(name).stem)
    segments = chunk(
        elements,
        record_id=record.record_id,
        source_id=record.source_id,
        modality=record.modality,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        title=title,
    )
    return segments, warnings
