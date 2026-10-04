"""OpenAI-compatible servers (vLLM, Ollama, LM Studio, TGI) with JSON fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from meldr.llm.base import Completion, Message, json_instruction
from meldr.llm.providers._common import openai_completion, translate_sdk_error
from meldr.llm.providers.openai import OpenAIProvider, response_format


class OpenAICompatibleProvider(OpenAIProvider):
    """Self-hosted models behind an OpenAI-compatible API. Local by default."""

    name = "openai_compatible"
    max_tokens_param = "max_tokens"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        http_client: Any = None,
        client: Any = None,
        is_local: bool = True,
    ) -> None:
        super().__init__(
            model, api_key=api_key, base_url=base_url, http_client=http_client, client=client
        )
        self.base_url = base_url
        self.is_local = is_local
        self._native_schema = True  # until the first time the server rejects response_format

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        if json_schema is not None and self._native_schema:
            kwargs = self._kwargs(messages, system, temperature, max_tokens)
            kwargs["response_format"] = response_format(json_schema)
            try:
                resp = await self._clients.get().chat.completions.create(**kwargs)
            except Exception as e:
                if getattr(e, "status_code", None) not in (400, 422):
                    raise translate_sdk_error(e, self.name) from e
                self._native_schema = False  # the server rejected response_format: use the prompt
            else:
                return openai_completion(resp, self.model)
        if json_schema is not None:
            system = (system or "") + json_instruction(json_schema)
        kwargs = self._kwargs(messages, system, temperature, max_tokens)
        return openai_completion(await self._create(**kwargs), self.model)
