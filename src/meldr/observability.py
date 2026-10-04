"""Structured logging that does not include payload values, plus the RunReport."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from meldr._compat import UTC

logger = logging.getLogger("meldr")

_SAFE_TYPES = (int, float, bool, type(None))
_REQUEST_ID: ContextVar[str | None] = ContextVar("meldr_request_id", default=None)


class RunReport(BaseModel):
    """Summary of one meld or extract run. It contains counts and IDs, not values."""

    run_id: str
    source_id: str
    target: str | None
    started_at: datetime
    duration_s: float = 0.0
    rows_in: int = 0
    rows_out: int = 0
    rows_quarantined: int = 0
    fields_mapped: int = 0
    fields_pending: int = 0
    values_lost: dict[str, int] = Field(default_factory=dict)
    spec_id: str | None = None
    spec_version: int | None = None
    spec_cache_hit: bool = False
    llm_calls: int = 0
    llm_cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float | None = None
    # An escalation to a stronger model has a separate entry here.
    tokens_by_model: dict[str, dict[str, int]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object for each log line, with the `mr` extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra = getattr(record, "mr", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO, *, json_output: bool = True) -> None:
    """Send `meldr` logs to stderr, by default one JSON object for each line (idempotent)."""
    if any(h.get_name() == "meldr" for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.set_name("meldr")
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


# Only keys with these suffixes can contain strings (identifiers, not payload values).
_ID_SUFFIXES = (
    "id",
    "ids",
    "name",
    "names",
    "field",
    "fields",
    "path",
    "paths",
    "purpose",
    "provider",
    "model",
    "op",
    "target",
    "targets",
    "entity",
    "status",
    "reason",
    "kind",
    "version",
    "type",
    "mode",
    "collection",
    "classes",
)


def _safe_value(key: str, value: Any) -> Any:
    if isinstance(value, _SAFE_TYPES):
        return value
    if isinstance(value, list | tuple | set):
        items = list(value)
        if all(isinstance(i, _SAFE_TYPES) for i in items):
            return items
        if all(isinstance(i, str) for i in items) and key.endswith(_ID_SUFFIXES):
            return items
        return f"<{len(items)} items>"
    if isinstance(value, str) and key.endswith(_ID_SUFFIXES):
        return value
    if isinstance(value, dict):
        return f"<{len(value)} keys>"
    return "<redacted>"


@contextmanager
def request_id_context(request_id: str | None) -> Iterator[None]:
    """Set a request ID for logs made in this context."""
    token = _REQUEST_ID.set(request_id)
    try:
        yield
    finally:
        _REQUEST_ID.reset(token)


def _span_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is not None and isinstance(value, (*_SAFE_TYPES, str)):
            safe[f"meldr.{key}"] = value
    request_id = _REQUEST_ID.get()
    if request_id is not None:
        safe["meldr.request_id"] = request_id
    return safe


@contextmanager
def otel_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Make an OpenTelemetry span when the optional extra is installed."""
    try:
        from opentelemetry import trace
    except ImportError:
        yield None
        return
    tracer = trace.get_tracer("meldr")
    with tracer.start_as_current_span(name, attributes=_span_attributes(attributes)) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(type(e).__name__)))
            raise


def set_span_attributes(span: Any, **attributes: Any) -> None:
    """Add safe scalar attributes to an OpenTelemetry span."""
    if span is None:
        return
    for key, value in _span_attributes(attributes).items():
        span.set_attribute(key, value)


def safe_log(event: str, *, level: int = logging.INFO, **fields: Any) -> dict[str, Any]:
    """Log `event` with only field names, counts, and IDs, and redact all other values.

    Returns the sanitized payload (useful in tests).
    """
    clean = {k: _safe_value(k, v) for k, v in fields.items()}
    request_id = _REQUEST_ID.get()
    if request_id is not None:
        clean["request_id"] = request_id
    logger.log(level, event, extra={"mr": clean})
    return clean
