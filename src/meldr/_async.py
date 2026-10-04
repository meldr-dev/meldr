"""Sync wrappers that work inside an active event loop (for example, in Jupyter)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
from typing import Any, TypeVar

R = TypeVar("R")


def run_sync(coro: Coroutine[Any, Any, R]) -> R:
    """Run `coro` to completion from sync code, also when a loop is already active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def iter_sync(source: AsyncIterator[R]) -> Iterator[R]:
    """Iterate an async iterator from sync code one item at a time (no buffer).

    Drives a private event loop inline, or on one worker thread when a loop is already active.
    """
    loop = asyncio.new_event_loop()
    pool: concurrent.futures.ThreadPoolExecutor | None = None
    step: Callable[[Awaitable[Any]], Any] = loop.run_until_complete
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        runner = pool

        def step(coro: Awaitable[Any]) -> Any:
            return runner.submit(loop.run_until_complete, coro).result()

    try:
        while True:
            try:
                yield step(source.__anext__())
            except StopAsyncIteration:
                return
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            step(aclose())
        step(loop.shutdown_asyncgens())
        if pool is not None:
            pool.shutdown()
        loop.close()
