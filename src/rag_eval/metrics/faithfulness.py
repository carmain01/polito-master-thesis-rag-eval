"""Faithfulness metric — measures whether the answer is supported by the retrieved contexts."""

from __future__ import annotations

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric


class Faithfulness(BaseMetric):
    """Evaluates if claims in the generated answer are grounded in the provided contexts."""

    @property
    def name(self) -> str:
        return "faithfulness"

    async def score(self, sample: TestSample) -> EvalResult:
        # TODO: Implement LLM-as-judge faithfulness scoring
        raise NotImplementedError
