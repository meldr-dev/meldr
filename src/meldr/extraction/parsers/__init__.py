"""Document parsers keyed by kind, with detection by magic bytes, then by file extension."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from meldr.core.errors import MeldrError

ElementKind = Literal["heading", "paragraph", "table"]


@dataclass
class DocElement:
    kind: ElementKind
    text: str
    page: int | None = None
    sheet: str | None = None


Parser = Callable[[bytes], list[DocElement]]
_PARSERS: dict[str, Parser] = {}

_EXT_KIND = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}


def register_parser(kind: str, parser: Parser) -> None:
    """Register (or replace) the parser for a document kind, for example a Docling-based 'pdf'."""
    _PARSERS[kind] = parser


def detect_kind(name: str, data: bytes) -> str:
    head = data[:512].lstrip()
    if head.startswith(b"%PDF"):
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile:
            names = []
        if any(n.startswith("word/") for n in names):
            return "docx"
        if any(n.startswith("xl/") for n in names):
            return "xlsx"
    low = head[:200].lower()
    if low.startswith((b"<!doctype html", b"<html")):
        return "html"
    return _EXT_KIND.get(Path(name).suffix.lower(), "text")


def parse(name: str, data: bytes) -> tuple[list[DocElement], list[str]]:
    """Parse bytes into elements. Returns (elements, warnings)."""
    _ensure_builtins()
    kind = detect_kind(name, data)
    parser = _PARSERS.get(kind)
    if parser is None:
        raise MeldrError(f"No parser registered for '{kind}' ({name}).")
    elements = parser(data)
    warnings: list[str] = []
    if kind == "pdf" and not any(e.text.strip() for e in elements):
        warnings.append(f"no_text_layer:{name}")
    return elements, warnings


def _ensure_builtins() -> None:
    if _PARSERS.keys() >= {"pdf", "docx", "xlsx", "html", "markdown", "text"}:
        return
    from meldr.extraction.parsers import office, text, web

    _PARSERS.setdefault("pdf", office.parse_pdf)
    _PARSERS.setdefault("docx", office.parse_docx)
    _PARSERS.setdefault("xlsx", office.parse_xlsx)
    _PARSERS.setdefault("html", web.parse_html)
    _PARSERS.setdefault("markdown", text.parse_markdown)
    _PARSERS.setdefault("text", text.parse_text)
