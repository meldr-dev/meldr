"""Anthropic Claude provider (structured output through forced tool use)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from meldr.core.errors import LLMOutputError, MissingExtraError
from meldr.llm.base import Capability, Completion, Message, Usage
from meldr.llm.providers._common import LoopClientPool, translate_sdk_error

EMIT_TOOL = "emit"


def _anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m.content, str):
            out.append({"role": m.role, "content": m.content})
            continue
        blocks: list[dict[str, Any]] = []
        for p in m.content:
            if p.type == "text":
                blocks.append({"type": "text", "text": p.text or ""})
            else:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": p.media_type or "image/jpeg",
                            "data": p.image_b64 or "",
                        },
                    }
                )
        out.append({"role": m.role, "content": blocks})
    return out


class AnthropicProvider:
    """Claude through the official async SDK.

    SDK 1.x `messages.create` exposes no sampling parameters, so the provider does not send
    `temperature`. Determinism comes from the response cache (see DECISIONS.md).
    """

    name = "anthropic"
    is_local = False
    capabilities = frozenset({Capability.NATIVE_STRUCTURED, Capability.VISION})

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any = None,
        client: Any = None,
    ) -> None:
        self.model = model
        if client is not None:
            self._clients = LoopClientPool(lambda: client, client=client)
            return
        try:
            import anthropic
        except ImportError as e:
            raise MissingExtraError("anthropic", "The anthropic provider") from e
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._clients = LoopClientPool(lambda: anthropic.AsyncAnthropic(**kwargs))

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": _anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if json_schema is not None:
            kwargs["tools"] = [
                {
                    "name": EMIT_TOOL,
                    "description": "Return the result as structured data.",
                    "input_schema": json_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": EMIT_TOOL}
        try:
            resp = await self._clients.get().messages.create(**kwargs)
        except Exception as e:
            raise translate_sdk_error(e, self.name) from e
        usage = Usage(
            input_tokens=int(getattr(resp.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(resp.usage, "output_tokens", 0) or 0),
        )
        text = ""
        for block in resp.content:
            if json_schema is not None and getattr(block, "type", None) == "tool_use":
                text = json.dumps(block.input)
                break
            if getattr(block, "type", None) == "text":
                text += block.text
        if json_schema is not None and not text:
            raise LLMOutputError(
                "Claude did not call the emit tool, so there is no structured output."
            )
        return Completion(text=text, usage=usage, model=str(getattr(resp, "model", self.model)))

    async def aclose(self) -> None:
        """Close SDK clients made by this provider."""
        await self._clients.aclose()
