"""Tenacity wait strategy shared by the LLM gateway and the REST connector."""

from __future__ import annotations

import random
from typing import Any

from tenacity.wait import wait_base


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header (the HTTP-date form gives None)."""
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class RetryAfterWait(wait_base):
    """Exponential backoff with jitter. Uses the `retry_after` seconds of the exception, if set."""

    def __init__(self, base: float, cap: float = 30.0) -> None:
        self.base = base
        self.cap = cap

    def __call__(self, retry_state: Any) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            return min(float(retry_after), self.cap)
        delay: float = min(self.cap, self.base * 2 ** (retry_state.attempt_number - 1))
        return delay * (0.5 + random.random() / 2)  # noqa: S311 - jitter, not crypto
