"""Configured token prices and usage totals. Meldr does not hard-code prices."""

from __future__ import annotations

from pydantic import BaseModel, Field

from meldr.llm.base import Usage


class Pricing(BaseModel):
    """USD for each million tokens, by model: model -> (input, output)."""

    per_million: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)

    def estimate(self, model: str, usage: Usage) -> float | None:
        """Estimated USD cost, or None when prices are unset for this model."""
        prices = self.per_million.get(model)
        if prices is None or prices[0] is None or prices[1] is None:
            return None
        return (usage.input_tokens * prices[0] + usage.output_tokens * prices[1]) / 1_000_000


class UsageTotals(BaseModel):
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float | None = None
    by_model: dict[str, Usage] = Field(default_factory=dict)

    def add(self, model: str, usage: Usage, cost: float | None) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.by_model[model] = self.by_model.get(model, Usage()) + usage
        if cost is not None:
            self.est_cost_usd = (self.est_cost_usd or 0.0) + cost

    def merged(self, other: UsageTotals) -> UsageTotals:
        out = self.model_copy(deep=True)
        out.calls += other.calls
        out.cache_hits += other.cache_hits
        out.input_tokens += other.input_tokens
        out.output_tokens += other.output_tokens
        for m, u in other.by_model.items():
            out.by_model[m] = out.by_model.get(m, Usage()) + u
        if other.est_cost_usd is not None:
            out.est_cost_usd = (out.est_cost_usd or 0.0) + other.est_cost_usd
        return out
