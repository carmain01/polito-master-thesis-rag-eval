"""Semantic Similarity metric — embedding-based comparison between answer and ground truth."""

from __future__ import annotations

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric


class SemanticSimilarity(BaseMetric):
    """Computes embedding cosine similarity between generated answer and ground truth."""

    @property
    def name(self) -> str:
        return "semantic_similarity"

    async def score(self, sample: TestSample) -> EvalResult:
        # TODO: Implement embedding-based semantic similarity
        raise NotImplementedError
