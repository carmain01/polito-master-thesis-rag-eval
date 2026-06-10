"""Answer Relevance metric — measures how well the answer addresses the question."""

from __future__ import annotations

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric


class AnswerRelevance(BaseMetric):
    """Evaluates whether the generated answer is relevant to the user's question."""

    @property
    def name(self) -> str:
        return "answer_relevance"

    async def score(self, sample: TestSample) -> EvalResult:
        # TODO: Implement LLM-as-judge answer relevance scoring
        raise NotImplementedError
