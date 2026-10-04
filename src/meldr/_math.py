from __future__ import annotations

import math
from collections.abc import Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Zero vectors count as unit length."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
