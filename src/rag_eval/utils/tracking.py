"""Usage tracking — monitor token consumption and costs across LLM calls."""

from __future__ import annotations
from typing import Any

from collections import defaultdict

from rag_eval.utils.provider import LLMResponse


class UsageTracker:
    """Tracks cumulative token usage and cost across all LLM calls.

    Maintains both aggregate totals and per-model breakdowns, enabling
    cost analysis and budget monitoring during evaluation runs.
    """

    def __init__(self) -> None:
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0
        self._call_count: int = 0
        self._per_model: dict[str, dict[str, float]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}
        )

    def record(self, response: LLMResponse) -> None:
        """Record usage from an LLM response.

        Args:
            response: The structured LLM response containing token counts and cost.
        """
        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens
        self._total_cost += response.cost_estimate
        self._call_count += 1

        key = f"{response.provider}/{response.model}"
        self._per_model[key]["input_tokens"] += response.input_tokens
        self._per_model[key]["output_tokens"] += response.output_tokens
        self._per_model[key]["cost"] += response.cost_estimate
        self._per_model[key]["calls"] += 1

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all calls."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all calls."""
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all calls."""
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_cost(self) -> float:
        """Total estimated cost in USD across all calls."""
        return self._total_cost

    @property
    def call_count(self) -> int:
        """Total number of LLM calls made."""
        return self._call_count

    def summary(self) -> dict[str, Any]:
        """Return a summary of all usage.

        Returns:
            Dictionary with aggregate totals and per-model breakdown.
        """
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "total_calls": self._call_count,
            "per_model": dict(self._per_model),
        }

    def reset(self) -> None:
        """Reset all tracked usage to zero."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0
        self._per_model.clear()

    def __repr__(self) -> str:
        return (
            f"UsageTracker(calls={self._call_count}, "
            f"tokens={self.total_tokens}, "
            f"cost=${self._total_cost:.4f})"
        )
