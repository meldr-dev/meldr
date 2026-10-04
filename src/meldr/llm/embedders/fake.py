"""Deterministic hash-based embedder: texts that share tokens get similar vectors."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-hashed-tokens vectors, L2 normalized. No network, fully deterministic."""

    name = "fake"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.casefold()):
            h = hashlib.sha256(tok.encode()).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            vec[idx] += 1.0 if h[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]
