"""OpenAI provider (response_format json_schema, strict off)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from meldr.core.errors import MissingExtraError
from meldr.llm.base import Capability, Completion, Message
from meldr.llm.providers._common import (
    LoopClientPool,
    openai_completion,
    openai_messages,
    translate_sdk_error,
)


def make_openai_client(
    *, api_key: str | None, base_url: str | None, http_client: Any, feature: str
) -> Any:
    try:
        import openai
    except ImportError as e:
        raise MissingExtraError("openai", feature) from e
    kwargs: dict[str, Any] = {"api_key": api_key or "not-needed", "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
    if http_client is not None:
        kwargs["http_client"] = http_client
    return openai.AsyncOpenAI(**kwargs)


def response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": False},
    }


class OpenAIProvider:
    """OpenAI chat completions through the official async SDK."""

    name = "openai"
    is_local = False
    capabilities = frozenset({Capability.NATIVE_STRUCTURED, Capability.VISION})
    max_tokens_param = "max_completion_tokens"

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
        self._clients = LoopClientPool(
            lambda: make_openai_client(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
                feature="OpenAI provider",
            ),
            client=client,
        )

    async def _create(self, **kwargs: Any) -> Any:
        try:
            return await self._clients.get().chat.completions.create(**kwargs)
        except Exception as e:
            raise translate_sdk_error(e, self.name) from e

    def _kwargs(
        self, messages: Sequence[Message], system: str | None, temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": openai_messages(messages, system),
            "temperature": temperature,
            self.max_tokens_param: max_tokens,
        }

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        kwargs = self._kwargs(messages, system, temperature, max_tokens)
        if json_schema is not None:
            kwargs["response_format"] = response_format(json_schema)
        return openai_completion(await self._create(**kwargs), self.model)

    async def aclose(self) -> None:
        """Close SDK clients made by this provider."""
        await self._clients.aclose()
