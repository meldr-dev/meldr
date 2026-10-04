"""LLMGateway: policy, redaction, cache, retries, structured output with repair, accounting."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from meldr._retry import RetryAfterWait
from meldr.core.errors import LLMOutputError, LLMTransientError, validation_details
from meldr.core.types import sha256_json
from meldr.llm import policy as policy_mod
from meldr.llm.base import (
    Capability,
    Completion,
    ContentPart,
    LLMProvider,
    LLMResult,
    Message,
    Usage,
    json_instruction,
)
from meldr.llm.cache import ResponseCache
from meldr.llm.policy import DataClass, DataPolicy, SourcePolicy
from meldr.llm.pricing import Pricing, UsageTotals
from meldr.observability import otel_span, safe_log, set_span_attributes

T = TypeVar("T", bound=BaseModel)
_USAGE_STACK: contextvars.ContextVar[tuple[UsageTotals, ...]] = contextvars.ContextVar(
    "meldr_usage_stack", default=()
)


@contextmanager
def usage_scope() -> Iterator[UsageTotals]:
    """Collect gateway usage for the current async call."""
    totals = UsageTotals()
    token = _USAGE_STACK.set((*_USAGE_STACK.get(), totals))
    try:
        yield totals
    finally:
        _USAGE_STACK.reset(token)


def extract_json(text: str) -> str:
    """Strip code fences and return the first balanced JSON object in `text`."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rsplit("```", 1)[0]
    start = s.find("{")
    if start < 0:
        return s
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s[start:]


class LLMGateway:
    """The only path from Meldr to an LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        policy: DataPolicy | None = None,
        cache: ResponseCache | None = None,
        redactor: Callable[[str], str] | None = None,
        max_concurrency: int = 4,
        max_retries: int = 4,
        max_repairs: int = 2,
        timeout_s: float = 120.0,
        pricing: Pricing | None = None,
        retry_base_s: float = 1.0,
    ) -> None:
        self.provider = provider
        self.policy = policy or DataPolicy.default_for(is_local=provider.is_local)
        self.cache = cache
        self.redactor = redactor
        self.max_retries = max_retries
        self.max_repairs = max_repairs
        self.timeout_s = timeout_s
        self.pricing = pricing or Pricing()
        self.retry_base_s = retry_base_s
        self._max_concurrency = max_concurrency
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()
        self._semaphore_lock = threading.Lock()
        self._totals = UsageTotals()
        self._totals_lock = threading.Lock()

    @property
    def totals(self) -> UsageTotals:
        """Calls, cache hits, tokens, and estimated cost so far."""
        with self._totals_lock:
            return self._totals.model_copy(deep=True)

    def effective_policy(self, source_policy: SourcePolicy | None = None) -> DataPolicy:
        return policy_mod.effective_policy(
            self.policy, source_policy, is_local=self.provider.is_local
        )

    def _semaphore(self) -> asyncio.Semaphore:
        """Return one semaphore for the current event loop."""
        loop = asyncio.get_running_loop()
        with self._semaphore_lock:
            sem = self._semaphores.get(loop)
            if sem is None:
                sem = asyncio.Semaphore(self._max_concurrency)
                self._semaphores[loop] = sem
            return sem

    def _redact(self, messages: Sequence[Message]) -> list[Message]:
        if self.redactor is None:
            return list(messages)
        red = self.redactor
        out: list[Message] = []
        for m in messages:
            if isinstance(m.content, str):
                out.append(Message(role=m.role, content=red(m.content)))
            else:
                parts = [
                    ContentPart(**{**p.model_dump(), "text": red(p.text)})
                    if p.type == "text" and p.text is not None
                    else p
                    for p in m.content
                ]
                out.append(Message(role=m.role, content=parts))
        return out

    def _cache_key(
        self,
        system: str | None,
        messages: Sequence[Message],
        schema: dict[str, Any],
        prompt_version: str,
    ) -> str:
        return sha256_json(
            {
                "provider": self.provider.name,
                "model": self.provider.model,
                "system": system,
                "messages": [m.model_dump(mode="json") for m in messages],
                "schema": schema,
                "prompt_version": prompt_version,
            }
        )

    async def _call(
        self,
        messages: Sequence[Message],
        *,
        system: str | None,
        schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(LLMTransientError),
            stop=stop_after_attempt(self.max_retries + 1),
            wait=RetryAfterWait(self.retry_base_s),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                async with self._semaphore():
                    try:
                        return await asyncio.wait_for(
                            self.provider.complete(
                                messages,
                                system=system,
                                json_schema=schema,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            ),
                            timeout=self.timeout_s,
                        )
                    except TimeoutError as e:
                        raise LLMTransientError(
                            f"{self.provider.name} timed out after {self.timeout_s}s"
                        ) from e
        raise AssertionError("unreachable")  # pragma: no cover

    def _account(self, completion: Completion) -> None:
        cost = self.pricing.estimate(completion.model, completion.usage)
        with self._totals_lock:
            self._totals.add(completion.model, completion.usage, cost)
            for totals in _USAGE_STACK.get():
                totals.add(completion.model, completion.usage, cost)

    def _account_cache_hit(self) -> None:
        with self._totals_lock:
            self._totals.cache_hits += 1
            for totals in _USAGE_STACK.get():
                totals.cache_hits += 1

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        response_model: type[T],
        contains: set[DataClass],
        purpose: str,
        prompt_version: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        source_policy: SourcePolicy | None = None,
    ) -> LLMResult[T]:
        """Validated structured generation with an optional tracing span."""
        with otel_span(
            "meldr.llm.generate",
            provider=self.provider.name,
            model=self.provider.model,
            purpose=purpose,
        ) as span:
            result = await self._generate(
                messages,
                response_model=response_model,
                contains=contains,
                purpose=purpose,
                prompt_version=prompt_version,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                source_policy=source_policy,
            )
            set_span_attributes(
                span,
                cached=result.cached,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                attempts=result.attempts,
            )
            return result

    async def _generate(
        self,
        messages: Sequence[Message],
        *,
        response_model: type[T],
        contains: set[DataClass],
        purpose: str,
        prompt_version: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        source_policy: SourcePolicy | None = None,
    ) -> LLMResult[T]:
        """Validated structured generation.

        The policy check runs before all I/O and can raise PolicyViolation.
        """
        policy_mod.check(
            self.effective_policy(source_policy), set(contains), provider=self.provider.name
        )
        started = time.monotonic()
        msgs = self._redact(messages)
        schema = response_model.model_json_schema()
        sys_prompt = system
        native = Capability.NATIVE_STRUCTURED in self.provider.capabilities
        if not native:
            sys_prompt = (system or "") + json_instruction(schema)

        key = None
        if self.cache is not None and temperature == 0:
            key = self._cache_key(sys_prompt, msgs, schema, prompt_version)
            hit = await asyncio.to_thread(self.cache.get, key)
            if hit is not None:
                try:
                    parsed = response_model.model_validate_json(hit)
                except ValidationError:
                    parsed = None
                if parsed is not None:
                    self._account_cache_hit()
                    self._audit(purpose, contains, Usage(), cached=True, started=started)
                    return LLMResult(
                        parsed=parsed,
                        text=hit,
                        usage=Usage(),
                        provider=self.provider.name,
                        model=self.provider.model,
                        cached=True,
                        attempts=0,
                    )

        conversation = list(msgs)
        usage = Usage()
        attempts = 0
        last_error = ""
        for _ in range(self.max_repairs + 1):
            attempts += 1
            completion = await self._call(
                conversation,
                system=sys_prompt,
                schema=schema if native else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._account(completion)
            usage = usage + completion.usage
            raw = extract_json(completion.text)
            try:
                parsed = response_model.model_validate_json(raw)
            except ValidationError as e:
                last_error = validation_details(e)
                conversation += [
                    Message(role="assistant", content=completion.text),
                    Message(
                        role="user",
                        content="Your previous output did not validate against the schema:\n"
                        f"{last_error}\nReturn a corrected JSON object only.",
                    ),
                ]
                continue
            if key is not None and self.cache is not None:
                await asyncio.to_thread(self.cache.set, key, parsed.model_dump_json())
            self._audit(purpose, contains, usage, cached=False, started=started)
            return LLMResult(
                parsed=parsed,
                text=raw,
                usage=usage,
                provider=self.provider.name,
                model=completion.model,
                cached=False,
                attempts=attempts,
            )
        self._audit(purpose, contains, usage, cached=False, started=started, failed=True)
        raise LLMOutputError(
            f"{self.provider.name}/{self.provider.model} returned invalid output for '{purpose}' "
            f"after {attempts} tries. Last errors: {last_error}. Try a stronger model or "
            "check that the provider supports JSON output."
        )

    def _audit(
        self,
        purpose: str,
        contains: set[DataClass],
        usage: Usage,
        *,
        cached: bool,
        started: float,
        failed: bool = False,
    ) -> None:
        safe_log(
            "llm.generate",
            purpose=purpose,
            provider=self.provider.name,
            model=self.provider.model,
            data_classes=sorted(str(c) for c in contains),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached=cached,
            failed=failed,
            latency_ms=round((time.monotonic() - started) * 1000, 1),
        )

    async def aclose(self) -> None:
        """Close provider clients made by this gateway."""
        close = getattr(self.provider, "aclose", None)
        if close is None:
            return
        if inspect.iscoroutinefunction(close):
            await close()
        else:
            await asyncio.to_thread(close)
