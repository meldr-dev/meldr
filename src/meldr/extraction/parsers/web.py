"""HTML parser (BeautifulSoup) — extra `docs`."""

from __future__ import annotations

from meldr.core.errors import MissingExtraError
from meldr.extraction.parsers import DocElement

_BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "pre", "blockquote"]


def parse_html(data: bytes) -> list[DocElement]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise MissingExtraError("docs", "Parsing HTML files") from e
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    out: list[DocElement] = []
    for el in soup.find_all(_BLOCKS):
        if el.find_parent(_BLOCKS) is not None:
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        name = el.name or ""
        if name.startswith("h") and name[1:].isdigit():
            out.append(DocElement(kind="heading", text=text))
        elif name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            out.append(DocElement(kind="table", text=" | ".join(cells)))
        else:
            out.append(DocElement(kind="paragraph", text=text))
    if not out:
        text = " ".join(soup.get_text(" ", strip=True).split())
        if text:
            out.append(DocElement(kind="paragraph", text=text))
    return out
