"""BERTScore metric — measures semantic overlap using a language model."""

from __future__ import annotations

import asyncio
import logging
import warnings

from bert_score import score as bert_score

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class BERTScore(BaseMetric):
    """Evaluates semantic similarity using BERTScore."""

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli") -> None:
        self.model_type = model_type
        # Pre-load or warm up the model? It is automatically loaded by bert_score on first use.

    @property
    def name(self) -> str:
        return "bert_score"

    def _compute_bert_score(self, answer: str, ground_truth: str) -> tuple[float, float, float]:
        """Run BERTScore synchronously (called via asyncio.to_thread)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P, R, F1 = bert_score(
                [answer],
                [ground_truth],
                model_type=self.model_type,
                lang="en",
                verbose=False,
            )
        return float(P.item()), float(R.item()), float(F1.item())

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer or not sample.ground_truth:
            return EvalResult(
                metric_name=self.name, score=0.0, reason="Missing answer or ground truth."
            )

        # Offload CPU-intensive BERTScore computation to a thread to avoid blocking the event loop
        P, R, F1 = await asyncio.to_thread(
            self._compute_bert_score, sample.answer, sample.ground_truth
        )

        f1_score = max(0.0, min(1.0, F1))

        return EvalResult(
            metric_name=self.name,
            score=f1_score,
            reason=f"Computed BERTScore using {self.model_type}",
            metadata={"precision": P, "recall": R, "f1": f1_score},
        )
