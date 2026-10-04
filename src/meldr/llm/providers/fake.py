"""Deterministic fake provider: scripted, replay (cassettes), and oracle modes."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from meldr.core.errors import LLMError, LLMTransientError
from meldr.core.types import sha256_json
from meldr.llm.base import Capability, Completion, LLMProvider, Message, Usage

Response = str | dict[str, Any] | BaseModel


@dataclass
class FakeRequest:
    """What the fake provider received (inspect `FakeProvider.calls` in tests)."""

    messages: list[Message]
    system: str | None
    json_schema: dict[str, Any] | None
    temperature: float
    max_tokens: int

    @property
    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return "\n".join(m.text_parts())
        return ""

    def payload(self) -> dict[str, Any] | None:
        """The JSON payload in the first user message, if it has one."""
        for m in self.messages:
            if m.role != "user":
                continue
            for text in m.text_parts():
                start = text.find("{")
                if start >= 0:
                    try:
                        obj = json.loads(text[start:])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        return obj
        return None

    def hash(self) -> str:
        request = {
            "messages": [m.model_dump(mode="json") for m in self.messages],
            "system": self.system,
            "schema": self.json_schema,
        }
        return sha256_json(request)[:32]


Oracle = Callable[[FakeRequest], Response]


def _to_text(r: Response) -> str:
    if isinstance(r, BaseModel):
        return r.model_dump_json()
    if isinstance(r, dict):
        return json.dumps(r)
    return r


class FakeProvider:
    """A test double that does not touch the network (except when it records a cassette)."""

    name = "fake"

    def __init__(
        self,
        *,
        mode: Literal["scripted", "replay", "oracle"] = "scripted",
        responses: Sequence[Response] = (),
        oracle: Oracle | None = None,
        cassette_dir: str | Path | None = None,
        record_with: LLMProvider | None = None,
        fail_times: int = 0,
        fail_retry_after: float | None = None,
        invalid_json_times: int = 0,
        model: str = "fake-model",
        is_local: bool = True,
        capabilities: frozenset[Capability] = frozenset(
            {Capability.NATIVE_STRUCTURED, Capability.VISION}
        ),
        usage: Usage | None = None,
    ) -> None:
        if mode == "oracle" and oracle is None:
            raise LLMError("FakeProvider(mode='oracle') must have an `oracle` callable.")
        if mode == "replay" and cassette_dir is None:
            raise LLMError("FakeProvider(mode='replay') must have `cassette_dir`.")
        self.mode = mode
        self.model = model
        self.is_local = is_local
        self.capabilities = capabilities
        self._responses: deque[Response] = deque(responses)
        self._oracle = oracle
        self._cassettes = Path(cassette_dir) if cassette_dir else None
        self._record_with = record_with
        self._fail_times = fail_times
        self._fail_retry_after = fail_retry_after
        self._invalid_json_times = invalid_json_times
        self._usage = usage or Usage(input_tokens=10, output_tokens=5)
        self.calls: list[FakeRequest] = []

    def push(self, *responses: Response) -> None:
        """Queue more scripted responses."""
        self._responses.extend(responses)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        req = FakeRequest(list(messages), system, json_schema, temperature, max_tokens)
        self.calls.append(req)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise LLMTransientError("fake transient failure", retry_after=self._fail_retry_after)
        if self._invalid_json_times > 0:
            self._invalid_json_times -= 1
            return Completion(text="not json {", usage=self._usage, model=self.model)
        text = await self._respond(req)
        return Completion(text=text, usage=self._usage, model=self.model)

    async def _respond(self, req: FakeRequest) -> str:
        if self.mode == "scripted":
            if not self._responses:
                raise LLMError(
                    "FakeProvider ran out of scripted responses. Push more with .push() or use "
                    "oracle mode."
                )
            return _to_text(self._responses.popleft())
        if self._oracle is not None:
            return _to_text(self._oracle(req))
        if self._cassettes is None:  # pragma: no cover - guarded in __init__
            raise LLMError("FakeProvider replay mode must have `cassette_dir`.")
        path = self._cassettes / f"{req.hash()}.json"
        if path.exists():
            return str(json.loads(path.read_text())["text"])
        if os.environ.get("MELDR_RECORD") == "1" and self._record_with is not None:
            real = await self._record_with.complete(
                req.messages,
                system=req.system,
                json_schema=req.json_schema,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            self._cassettes.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": real.text, "model": real.model}))
            return real.text
        raise LLMError(
            f"No cassette {path.name} in {self._cassettes}. Record it with MELDR_RECORD=1 and "
            "a live provider."
        )
