"""Heading-aware chunking into ContentSegments (≈4 characters for each token)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from meldr.core.segments import ContentSegment, Locator, segment_id
from meldr.core.types import Modality
from meldr.extraction.parsers import DocElement

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class _Chunk:
    title: str | None
    parts: list[str] = field(default_factory=list)
    page: int | None = None
    sheet: str | None = None
    body: bool = False

    @property
    def size(self) -> int:
        return sum(len(p) + 2 for p in self.parts)


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    buf = ""
    for w in text.split():
        if buf and len(buf) + len(w) + 1 > limit:
            out.append(buf)
            buf = w
        else:
            buf = f"{buf} {w}" if buf else w
    if buf:
        out.append(buf)
    return out


def chunk(
    elements: Sequence[DocElement],
    *,
    record_id: str,
    source_id: str,
    modality: Modality = Modality.DOCUMENT,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
    title: str | None = None,
) -> list[ContentSegment]:
    """New chunk at every heading and when the size target is reached (with overlap)."""
    limit = target_tokens * CHARS_PER_TOKEN
    overlap = overlap_tokens * CHARS_PER_TOKEN
    chunks: list[_Chunk] = []
    cur = _Chunk(title=title)

    for el in elements:
        if el.kind == "heading":
            if cur.parts:
                chunks.append(cur)
            cur = _Chunk(title=el.text, parts=[el.text])
            continue
        for piece in _split_long(el.text, limit):
            if cur.body and cur.size + len(piece) > limit:
                chunks.append(cur)
                tail = cur.parts[-1][-overlap:] if overlap else ""
                cur = _Chunk(title=cur.title, parts=[tail] if tail else [])
            cur.page = cur.page if cur.page is not None else el.page
            cur.sheet = cur.sheet or el.sheet
            cur.parts.append(piece)
            cur.body = True
    if cur.parts:
        chunks.append(cur)

    return [
        ContentSegment(
            segment_id=segment_id(source_id, record_id, i),
            source_id=source_id,
            record_id=record_id,
            modality=modality,
            text="\n\n".join(c.parts),
            title=c.title,
            locator=Locator(page=c.page, sheet=c.sheet),
        )
        for i, c in enumerate(chunks)
    ]
