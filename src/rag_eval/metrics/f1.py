"""Token-level F1 metric."""

from __future__ import annotations

import string
from collections import Counter

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric


def normalize_text(text: str) -> str:
    """Lowercases, removes punctuation, articles, and standardizes whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens)


class TokenF1(BaseMetric):
    """Evaluates token-level F1 score between answer and ground truth."""

    @property
    def name(self) -> str:
        return "token_f1"

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer or not sample.ground_truth:
            return EvalResult(
                metric_name=self.name, score=0.0, reason="Missing answer or ground truth."
            )

        ref_tokens = normalize_text(sample.ground_truth).split()
        hyp_tokens = normalize_text(sample.answer).split()

        if not ref_tokens or not hyp_tokens:
            return EvalResult(metric_name=self.name, score=0.0, reason="Empty after normalization.")

        ref_counter = Counter(ref_tokens)
        hyp_counter = Counter(hyp_tokens)
        common_tokens = sum((ref_counter & hyp_counter).values())
        if common_tokens == 0:
            return EvalResult(metric_name=self.name, score=0.0, reason="No common tokens.")

        precision = common_tokens / len(hyp_tokens)
        recall = common_tokens / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)

        return EvalResult(
            metric_name=self.name,
            score=f1,
            reason="Computed token-level F1",
            metadata={"precision": precision, "recall": recall},
        )
