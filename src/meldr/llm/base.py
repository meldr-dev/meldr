"""Provider-neutral LLM types."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from meldr._compat import StrEnum


class Capability(StrEnum):
    NATIVE_STRUCTURED = "native_structured"
    VISION = "vision"


class ContentPart(BaseModel):
    type: Literal["text", "image"]
    text: str | None = None
    image_b64: str | None = None
    media_type: str | None = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentPart]

    def text_parts(self) -> list[str]:
        """All text in this message."""
        if isinstance(self.content, str):
            return [self.content]
        return [p.text or "" for p in self.content if p.type == "text"]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Completion(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)
    model: str


@runtime_checkable
class LLMProvider(Protocol):
    """A chat model behind a provider SDK."""

    name: str
    model: str
    capabilities: frozenset[Capability]
    is_local: bool

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> Completion: ...


T = TypeVar("T", bound=BaseModel)


class LLMResult(BaseModel, Generic[T]):
    parsed: T
    text: str
    usage: Usage
    provider: str
    model: str
    cached: bool
    attempts: int


@runtime_checkable
class Embedder(Protocol):
    """Text embedding model."""

    name: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def json_instruction(schema: dict[str, Any]) -> str:
    """System-prompt suffix for providers without native structured output."""
    return (
        "\n\nRespond with a single JSON object only (no prose, no code fences) that validates "
        "against this JSON schema:\n" + json.dumps(schema, separators=(",", ":"))
    )
