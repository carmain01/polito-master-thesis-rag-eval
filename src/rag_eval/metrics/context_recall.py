"""Context Recall metric — measures whether all relevant information was retrieved."""

from __future__ import annotations

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric


class ContextRecall(BaseMetric):
    """Evaluates whether the retrieved contexts cover all necessary information."""

    @property
    def name(self) -> str:
        return "context_recall"

    async def score(self, sample: TestSample) -> EvalResult:
        # TODO: Implement context recall scoring
        raise NotImplementedError
