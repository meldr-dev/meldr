"""Optional adapter for litellm.acompletion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from meldr.core.errors import MissingExtraError
from meldr.llm.base import Capability, Completion, Message
from meldr.llm.providers._common import (
    openai_completion,
    openai_messages,
    translate_sdk_error,
)
from meldr.llm.providers.openai import response_format


class LiteLLMProvider:
    """Provider for all models that litellm supports (a vendor provider unless `is_local`)."""

    name = "litellm"
    capabilities = frozenset({Capability.NATIVE_STRUCTURED})

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        is_local: bool = False,
        acompletion: Any = None,
    ) -> None:
        self.model = model
        self.is_local = is_local
        self._api_key = api_key
        self._base_url = base_url
        if acompletion is None:
            try:
                import litellm
            except ImportError as e:
                raise MissingExtraError("litellm", "The litellm provider") from e
            acompletion = litellm.acompletion
        self._acompletion = acompletion

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
            "messages": openai_messages(messages, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["api_base"] = self._base_url
        if json_schema is not None:
            kwargs["response_format"] = response_format(json_schema)
        try:
            resp = await self._acompletion(**kwargs)
        except Exception as e:
            raise translate_sdk_error(e, self.name) from e
        return openai_completion(resp, self.model)
