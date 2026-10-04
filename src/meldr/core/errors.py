"""Error hierarchy. Every message says what happened and how to fix it."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import ValidationError


def validation_details(e: ValidationError, limit: int = 20) -> str:
    """`loc: msg; ...` for the first `limit` errors of a pydantic ValidationError."""
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in e.errors()[:limit]
    )


class MeldrError(Exception):
    """Base class for all Meldr errors."""


class ConfigError(MeldrError):
    """Configuration is invalid or incomplete."""


class SecretReferenceError(MeldrError):
    """Meldr cannot resolve a secret reference."""


class MissingExtraError(MeldrError, ImportError):
    """An optional dependency is not installed."""

    def __init__(self, extra: str, feature: str | None = None) -> None:
        what = feature or f"this feature ({extra})"
        super().__init__(
            f"{what} uses an optional extra that is not installed. "
            f"Install it with: pip install 'meldr[{extra}]'"
        )
        self.extra = extra


class SpecValidationError(MeldrError):
    """A mapping spec or transform step is invalid."""


class SpecConflictError(MeldrError):
    """A mapping spec changed before this save."""


class PolicyViolation(MeldrError):
    """A request contains data that the effective data policy forbids."""


class LLMError(MeldrError):
    """Base class for LLM failures."""


class LLMTransientError(LLMError):
    """A retryable provider failure (rate limit, 5xx, timeout, connection)."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMOutputError(LLMError):
    """After repairs, the model output still fails parsing or validation."""


class ConnectorError(MeldrError):
    """A connector failed to read from its source."""


class SinkError(MeldrError):
    """A sink failed to write or read."""


class DriftBlockedError(MeldrError):
    """A source shape change removed inputs that feed required targets."""


class ExternalToolError(MeldrError):
    """A necessary system binary (for example, ffmpeg) is missing or failed."""
