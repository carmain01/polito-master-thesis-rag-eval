"""Semantic Similarity metric — measures cosine similarity between embeddings."""

from __future__ import annotations

import asyncio

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric
from rag_eval.utils.embeddings import EmbeddingClient, cosine_similarity


class SemanticSimilarity(BaseMetric):
    """Evaluates embedding-based semantic similarity between answer and ground truth."""

    def __init__(self, embed_client: EmbeddingClient) -> None:
        self.embed = embed_client

    @property
    def name(self) -> str:
        return "semantic_similarity"

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer or not sample.ground_truth:
            return EvalResult(
                metric_name=self.name, score=0.0, reason="Missing answer or ground truth."
            )

        try:
            # Offload CPU-intensive embedding to a thread to avoid blocking the event loop
            ans_emb = await asyncio.to_thread(self.embed.embed_single, sample.answer)
            ref_emb = await asyncio.to_thread(self.embed.embed_single, sample.ground_truth)

            # Compute cosine similarity
            sim = cosine_similarity(ans_emb, ref_emb)

            # Ensure it is bounded
            sim = max(0.0, min(1.0, sim))

            return EvalResult(
                metric_name=self.name,
                score=sim,
                reason="Computed cosine similarity between embeddings.",
            )
        except Exception as e:
            return EvalResult(metric_name=self.name, score=0.0, reason=f"Failed to embed: {e}")
