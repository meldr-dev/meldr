"""Plain text and Markdown parsers."""

from __future__ import annotations

import re

from meldr.extraction.parsers import DocElement

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")  # pragma: no cover


def _paragraphs(text: str) -> list[DocElement]:
    blocks = re.split(r"\n\s*\n", text)
    return [DocElement(kind="paragraph", text=b.strip()) for b in blocks if b.strip()]


def parse_text(data: bytes) -> list[DocElement]:
    return _paragraphs(_decode(data))


def parse_markdown(data: bytes) -> list[DocElement]:
    out: list[DocElement] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.extend(_paragraphs("\n".join(buf)))
            buf.clear()

    for line in _decode(data).splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            out.append(DocElement(kind="heading", text=m.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return out
