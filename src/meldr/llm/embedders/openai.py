"""OpenAI and OpenAI-compatible embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from meldr.llm.providers._common import LoopClientPool, translate_sdk_error
from meldr.llm.providers.openai import make_openai_client


class OpenAIEmbedder:
    """`embeddings.create` on OpenAI or on a compatible server (set `base_url`)."""

    def __init__(
        self,
        model: str,
        *,
        dim: int,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.name = "openai_compatible" if base_url else "openai"
        self._clients = LoopClientPool(
            lambda: make_openai_client(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
                feature="OpenAI embeddings",
            ),
            client=client,
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            resp = await self._clients.get().embeddings.create(model=self.model, input=list(texts))
        except Exception as e:
            raise translate_sdk_error(e, self.name) from e
        return [list(d.embedding) for d in sorted(resp.data, key=lambda d: d.index)]

    async def aclose(self) -> None:
        """Close SDK clients made by this embedder."""
        await self._clients.aclose()
