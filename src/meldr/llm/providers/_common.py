"""Helpers shared by SDK-backed providers."""

from __future__ import annotations

import asyncio
import inspect
import threading
import weakref
from collections.abc import Sequence
from typing import Any

from meldr._retry import parse_retry_after
from meldr.core.errors import LLMError, LLMTransientError
from meldr.llm.base import Completion, Message, Usage

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class LoopClientPool:
    """Keep one async SDK client for each event loop."""

    def __init__(self, factory: Any, client: Any = None) -> None:
        self._factory = factory
        self._provided = client
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.Lock()

    def get(self) -> Any:
        """Return the client for the current event loop."""
        if self._provided is not None:
            return self._provided
        loop = asyncio.get_running_loop()
        with self._lock:
            client = self._clients.get(loop)
            if client is None:
                client = self._factory()
                self._clients[loop] = client
            return client

    async def aclose(self) -> None:
        """Close clients made by this pool."""
        if self._provided is not None:
            return
        with self._lock:
            clients = list(self._clients.values())
            self._clients = weakref.WeakKeyDictionary()
        for client in clients:
            close = getattr(client, "close", None)
            if close is None:
                continue
            if inspect.iscoroutinefunction(close):
                await close()
            else:
                result = await asyncio.to_thread(close)
                if inspect.isawaitable(result):
                    await result


def retry_after_from(exc: Any) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    return parse_retry_after(headers.get("retry-after")) if headers else None


def translate_sdk_error(exc: Exception, provider: str) -> Exception:
    """Map an SDK exception to LLMTransientError (retryable) or LLMError."""
    status = getattr(exc, "status_code", None)
    cls = type(exc).__name__
    transient = status in TRANSIENT_STATUS or cls in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
        "Timeout",
    }
    if transient:
        return LLMTransientError(
            f"{provider} transient error ({cls}, status={status})",
            retry_after=retry_after_from(exc),
        )
    return LLMError(f"{provider} request failed ({cls}, status={status}). Check model and key.")


def openai_completion(resp: Any, fallback_model: str) -> Completion:
    """Completion from an OpenAI-style chat response (OpenAI SDK, litellm)."""
    usage = getattr(resp, "usage", None)
    return Completion(
        text=resp.choices[0].message.content or "",
        usage=Usage(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        ),
        model=str(getattr(resp, "model", None) or fallback_model),
    )


def openai_messages(messages: Sequence[Message], system: str | None) -> list[dict[str, Any]]:
    """Convert to the OpenAI chat format."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if isinstance(m.content, str):
            out.append({"role": m.role, "content": m.content})
            continue
        parts: list[dict[str, Any]] = []
        for p in m.content:
            if p.type == "text":
                parts.append({"type": "text", "text": p.text or ""})
            else:
                url = f"data:{p.media_type or 'image/jpeg'};base64,{p.image_b64}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
        out.append({"role": m.role, "content": parts})
    return out
