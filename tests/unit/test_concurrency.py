from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import polars as pl

from meldr import LLMGateway, Meldr, RawBatch
from meldr.llm.base import Message
from meldr.llm.providers.fake import FakeProvider
from meldr.pipeline.oracle import heuristic_oracle
from tests.helpers import make_config


def _batch(source_id: str) -> RawBatch:
    return RawBatch.from_polars(
        pl.DataFrame(
            {
                "org_id": ["1"],
                "name": ["Acme"],
                "country": ["US"],
            }
        ),
        source_id,
    )


async def test_concurrent_meld_usage_is_scoped_to_each_call(tmp_path: Any) -> None:
    provider = FakeProvider(mode="oracle", oracle=heuristic_oracle)
    gateway = LLMGateway(provider, retry_base_s=0.0)
    mr = Meldr(make_config(tmp_path / "state"), llm=gateway)

    results = await asyncio.gather(
        mr.ameld(_batch("source_a"), "Organization", write=False, index=False),
        mr.ameld(_batch("source_b"), "Organization", write=False, index=False),
    )

    assert all(result.report.llm_calls > 0 for result in results)
    assert sum(result.report.llm_calls for result in results) == len(provider.calls)


async def test_sync_sink_does_not_block_the_caller_loop(tmp_path: Any, monkeypatch: Any) -> None:
    mr = Meldr(make_config(tmp_path / "state"))
    batch = _batch("source")
    spec = await mr.apropose(batch, "Organization")
    loop_progress = threading.Event()
    observed_progress: list[bool] = []

    def slow_write(*args: Any, **kwargs: Any) -> None:
        time.sleep(0.05)
        observed_progress.append(loop_progress.is_set())

    monkeypatch.setattr(mr, "_write", slow_write)

    async def progress() -> None:
        await asyncio.sleep(0.01)
        loop_progress.set()

    await asyncio.gather(
        mr.ameld(batch, "Organization", spec=spec, index=False),
        progress(),
    )

    assert observed_progress == [True]


def test_openai_provider_builds_a_client_for_each_event_loop(monkeypatch: Any) -> None:
    created: list[object] = []

    class Completions:
        async def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"value": 1}'))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                model="test-model",
            )

    class Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    def make_client(**kwargs: Any) -> Client:
        client = Client()
        created.append(client)
        return client

    import meldr.llm.providers.openai as openai_module

    monkeypatch.setattr(openai_module, "make_openai_client", make_client)
    provider = openai_module.OpenAIProvider("test-model", api_key="not-a-secret")
    messages = [Message(role="user", content="return one")]

    asyncio.run(
        provider.complete(
            messages,
            system=None,
            json_schema=None,
            temperature=0.0,
            max_tokens=10,
        )
    )
    asyncio.run(
        provider.complete(
            messages,
            system=None,
            json_schema=None,
            temperature=0.0,
            max_tokens=10,
        )
    )

    assert len(created) == 2
    assert created[0] is not created[1]
