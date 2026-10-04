from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from meldr._async import run_sync
from meldr.core.errors import LLMError, LLMOutputError, LLMTransientError, PolicyViolation
from meldr.indexing.indexer import Indexer
from meldr.indexing.sinks.memory import MemoryVectorSink
from meldr.llm.base import Message
from meldr.llm.cache import MemoryCache, SQLiteResponseCache, SQLiteVectorCache
from meldr.llm.embedders.fake import FakeEmbedder
from meldr.llm.gateway import LLMGateway, extract_json
from meldr.llm.policy import DataClass, DataPolicy, SplitPolicy, effective_policy
from meldr.llm.pricing import Pricing
from meldr.llm.providers.fake import FakeProvider
from meldr.observability import safe_log


class Answer(BaseModel):
    value: int


MSG = [Message(role="user", content="secret payload 123")]


def _gw(provider: FakeProvider, **kw: object) -> LLMGateway:
    return LLMGateway(provider, retry_base_s=0.0, **kw)  # type: ignore[arg-type]


async def _gen(gw: LLMGateway, contains: set[DataClass] | None = None) -> int:
    res = await gw.generate(
        MSG,
        response_model=Answer,
        contains=contains or {DataClass.SCHEMA},
        purpose="test",
        prompt_version="v1",
    )
    return res.parsed.value


async def test_policy_blocks_before_any_io() -> None:
    fake = FakeProvider(responses=[{"value": 1}], is_local=False)
    gw = _gw(fake, policy=DataPolicy(allowed={DataClass.SCHEMA}))
    with pytest.raises(PolicyViolation, match="samples"):
        await _gen(gw, {DataClass.SCHEMA, DataClass.SAMPLES})
    assert fake.calls == []


def test_effective_policy_split() -> None:
    vendor = DataPolicy.default_for(is_local=False)
    split = SplitPolicy(vendor=[DataClass.SCHEMA], local=list(DataClass))
    assert effective_policy(vendor, split, is_local=False).allowed == {DataClass.SCHEMA}
    local = DataPolicy.default_for(is_local=True)
    assert DataClass.CONTENT in effective_policy(local, split, is_local=True).allowed


async def test_cache_retry_repair_and_totals(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="meldr")
    fake = FakeProvider(responses=[{"value": 7}], fail_times=2, invalid_json_times=1)
    pricing = Pricing(per_million={"fake-model": (1.0, 2.0)})
    gw = _gw(fake, cache=MemoryCache(), pricing=pricing)
    fake.push({"value": 7})
    assert await _gen(gw) == 7
    assert len(fake.calls) == 4  # 2 transient failures, 1 invalid JSON, 1 good
    assert "did not validate" in fake.calls[-1].last_user_text
    assert await _gen(gw) == 7  # cache hit: no new call
    assert len(fake.calls) == 4
    totals = gw.totals
    assert totals.cache_hits == 1 and totals.calls == 2 and totals.est_cost_usd
    events = [r.mr for r in caplog.records if r.getMessage() == "llm.generate"]  # type: ignore[attr-defined]
    assert len(events) == 2 and all("secret" not in json.dumps(e) for e in events)


async def test_repair_exhaustion_raises() -> None:
    gw = _gw(FakeProvider(responses=["nope"] * 5), max_repairs=1)
    with pytest.raises(LLMOutputError):
        await _gen(gw)


async def test_redactor_applied() -> None:
    fake = FakeProvider(responses=[{"value": 1}])
    gw = _gw(fake, redactor=lambda s: s.replace("123", "<n>"))
    await _gen(gw)
    assert "123" not in fake.calls[0].last_user_text


def test_extract_json() -> None:
    assert extract_json('```json\n{"a": "}"}\n```') == '{"a": "}"}'
    assert extract_json('prose {"a": {"b": 1}} tail') == '{"a": {"b": 1}}'


def test_sqlite_cache(tmp_path: Path) -> None:
    c = SQLiteResponseCache(tmp_path / "c.sqlite")
    c.set("k", "v")
    assert c.get("k") == "v" and c.get("x") is None
    vectors = SQLiteVectorCache(tmp_path / "e.sqlite")
    vectors.set_many({"a": [0.1, 1 / 3]})
    assert vectors.get_many(["a", "b"]) == {"a": [0.1, 1 / 3]}


def test_indexer_retries_dedupes_and_caches(tmp_path: Path) -> None:
    class Flaky(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 1
            self.batches: list[int] = []

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            if self.failures:
                self.failures -= 1
                raise LLMTransientError("rate limited", retry_after=0)
            self.batches.append(len(texts))
            return await super().embed(texts)

    embedder = Flaky()
    cache = SQLiteVectorCache(tmp_path / "e.sqlite")
    indexer = Indexer(embedder, MemoryVectorSink(), cache=cache)
    first = run_sync(indexer.embed(["a b", "c", "a b"]))
    assert first[0] == first[2] and embedder.batches == [2]
    assert run_sync(indexer.embed(["c", "a b"])) == [first[1], first[0]]
    assert embedder.batches == [2]


def test_fake_replay_and_oracle(tmp_path: Path) -> None:
    with pytest.raises(LLMError):
        FakeProvider(mode="oracle")
    replay = FakeProvider(mode="replay", cassette_dir=tmp_path)
    with pytest.raises(LLMError, match="No cassette"):
        run_sync(replay.complete(MSG, system=None, json_schema=None, temperature=0, max_tokens=1))
    oracle = FakeProvider(mode="oracle", oracle=lambda r: {"value": len(r.messages)})
    assert run_sync(_gen(_gw(oracle))) == 1


def test_cassettes_record_then_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = FakeProvider(responses=[{"value": 42}])
    recorder = FakeProvider(mode="replay", cassette_dir=tmp_path, record_with=real)
    monkeypatch.setenv("MELDR_RECORD", "1")
    assert run_sync(_gen(_gw(recorder))) == 42
    monkeypatch.delenv("MELDR_RECORD")
    replay = FakeProvider(mode="replay", cassette_dir=tmp_path)
    assert run_sync(_gen(_gw(replay))) == 42 and len(list(tmp_path.glob("*.json"))) == 1


async def test_fake_embedder_deterministic() -> None:
    e = FakeEmbedder(dim=16)
    a, b, c = await e.embed(["red truck", "red truck", "blue sky"])
    assert a == b and a != c and len(a) == 16


async def test_run_sync_inside_running_loop() -> None:
    async def seven() -> int:
        await asyncio.sleep(0)
        return 7

    assert run_sync(seven()) == 7


def test_safe_log_redacts_values() -> None:
    out = safe_log("x", source_id="erp", value="alice@example.com", rows=3, field_names=["a"])
    assert out == {"source_id": "erp", "value": "<redacted>", "rows": 3, "field_names": ["a"]}


def _anthropic_response(request: Any) -> Any:
    import httpx2

    body = json.loads(request.content)
    assert body["tool_choice"] == {"type": "tool", "name": "emit"}
    return httpx2.Response(
        200,
        json={
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": [{"type": "tool_use", "id": "t", "name": "emit", "input": {"value": 3}}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    )


async def test_anthropic_provider_mocked() -> None:
    httpx2 = pytest.importorskip("httpx2")  # the anthropic SDK's HTTP client
    from meldr.llm.providers.anthropic import AnthropicProvider

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(_anthropic_response))
    p = AnthropicProvider("claude-x", api_key="test", http_client=client)
    assert await _gen(_gw(p, policy=DataPolicy(allowed={DataClass.SCHEMA}))) == 3  # type: ignore[arg-type]


def _openai_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if "response_format" in body and "compat" in str(request.url):
        return httpx.Response(400, json={"error": {"message": "unsupported"}})
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"value": 4}'},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    )


async def test_openai_and_compatible_fallback() -> None:
    from meldr.llm.providers.openai import OpenAIProvider
    from meldr.llm.providers.openai_compatible import OpenAICompatibleProvider

    client = httpx.AsyncClient(transport=httpx.MockTransport(_openai_response))
    oa = OpenAIProvider("gpt-x", api_key="test", http_client=client)
    assert await _gen(_gw(oa)) == 4  # type: ignore[arg-type]
    compat = OpenAICompatibleProvider(
        "local", base_url="http://compat.local/v1", http_client=client
    )
    assert compat.is_local
    assert await _gen(_gw(compat)) == 4  # type: ignore[arg-type]


@pytest.mark.live_llm
async def test_live_anthropic_smoke() -> None:
    import os

    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("MELDR_LIVE_MODEL"):
        pytest.skip("set ANTHROPIC_API_KEY and MELDR_LIVE_MODEL")
    from meldr.llm.providers.anthropic import AnthropicProvider

    p = AnthropicProvider(os.environ["MELDR_LIVE_MODEL"], api_key=os.environ["ANTHROPIC_API_KEY"])
    gw = LLMGateway(p)
    res = await gw.generate(
        [Message(role="user", content="Return value 2.")],
        response_model=Answer,
        contains={DataClass.SCHEMA},
        purpose="smoke",
        prompt_version="v1",
        max_tokens=64,
    )
    assert res.parsed.value == 2
