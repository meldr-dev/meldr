"""Sandboxed rendering of entity embedding templates."""

from __future__ import annotations

import logging
from functools import cache, lru_cache
from typing import Any

from jinja2 import Template, TemplateError
from jinja2.sandbox import SandboxedEnvironment

from meldr.observability import safe_log


def _finalize(value: Any) -> Any:
    return "" if value is None else value


@cache
def _env() -> SandboxedEnvironment:
    return SandboxedEnvironment(autoescape=False, finalize=_finalize)


@lru_cache(maxsize=128)
def _template(source: str) -> Template | None:
    """Compile each source one time, or log one warning and return None for a broken template."""
    try:
        return _env().from_string(source)
    except TemplateError as e:
        safe_log("render.template_invalid", level=logging.WARNING, reason=type(e).__name__)
        return None


def render(template: str | None, values: dict[str, Any], fields: list[str]) -> str:
    """Render `template` with `values`, with `key: value` lines for non-null fields as fallback."""
    compiled = _template(template) if template else None
    if compiled is not None:
        try:
            text = compiled.render(**values)
            if text.strip():
                return " ".join(text.split())
        except TemplateError:
            pass  # a value the sandbox refuses: use the plain rendering for this row
    return "\n".join(f"{k}: {values[k]}" for k in fields if values.get(k) is not None)
