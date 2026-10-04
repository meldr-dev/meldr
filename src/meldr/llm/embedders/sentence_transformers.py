"""Local embeddings with sentence-transformers (optional extra `local-embed`)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from meldr.core.errors import MissingExtraError


class SentenceTransformerEmbedder:
    """Runs a local sentence-transformers model in a worker thread."""

    name = "sentence_transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2", *, st_model: Any = None) -> None:
        if st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise MissingExtraError("local-embed", "Local embeddings") from e
            st_model = SentenceTransformer(model)
        self._model = st_model
        self.model = model
        dim = st_model.get_sentence_embedding_dimension()
        self.dim = int(dim) if dim else 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(
            self._model.encode, list(texts), normalize_embeddings=True
        )
        return [list(map(float, v)) for v in vectors]
