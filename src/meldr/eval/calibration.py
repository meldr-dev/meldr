"""Choose the auto-approve threshold from labeled outcomes, not from a guess.

Matcher scores and LLM self-reported confidence rank well, but they are not probabilities
(Dong and others, VLDB'07). The 2026 tabular-calibration study found a verbalized-confidence
ECE of 0.35-0.64. `calibrate` takes (confidence, correct) pairs from a benchmark with ground
truth. It finds the lowest threshold at which the Wilson score lower bound of the precision
still meets the target.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class Calibration:
    target_precision: float
    points: int
    threshold: float | None = None  # None: this data cannot guarantee the target
    precision: float | None = None
    lower_bound: float | None = None
    auto_rate: float = 0.0  # share of proposals that `threshold` auto-approves
    reliability: list[dict[str, float]] = field(default_factory=list)


def wilson_lower(correct: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a proportion (95% by default)."""
    if total == 0:
        return 0.0
    p = correct / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, (centre - margin) / denom)


def calibrate(
    points: Sequence[tuple[float, bool]],
    *,
    target_precision: float = 0.97,
    z: float = 1.96,
    bins: int = 10,
) -> Calibration:
    """Lowest confidence threshold at which the Wilson lower bound on precision meets the target."""
    ranked = sorted(points, key=lambda p: -p[0])
    result = Calibration(target_precision=target_precision, points=len(ranked))
    correct = 0
    for accepted, (score, ok) in enumerate(ranked, start=1):
        correct += ok
        if accepted < len(ranked) and ranked[accepted][0] == score:
            continue  # a threshold admits every point with the same score
        lower = wilson_lower(correct, accepted, z)
        if lower >= target_precision:
            result.threshold = round(score, 4)
            result.precision = round(correct / accepted, 4)
            result.lower_bound = round(lower, 4)
            result.auto_rate = round(accepted / len(ranked), 4)
    width = 1.0 / bins
    for i in range(bins):
        lo, hi = i * width, (i + 1) * width
        inside = [ok for s, ok in ranked if lo <= s < hi or (i == bins - 1 and s == 1.0)]
        if inside:
            result.reliability.append(
                {
                    "low": round(lo, 2),
                    "high": round(hi, 2),
                    "count": len(inside),
                    "precision": round(sum(inside) / len(inside), 4),
                }
            )
    return result
