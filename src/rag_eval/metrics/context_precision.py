"""Context Precision metric — measures the signal-to-noise ratio of retrieved contexts."""

from __future__ import annotations

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric


class ContextPrecision(BaseMetric):
    """Evaluates whether the retrieved contexts are relevant (low noise)."""

    @property
    def name(self) -> str:
        return "context_precision"

    async def score(self, sample: TestSample) -> EvalResult:
        # TODO: Implement context precision scoring
        raise NotImplementedError
