"""Exact Match metric."""

from __future__ import annotations

import string

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric


def normalize_answer(text: str) -> str:
    """Lowercases, removes punctuation, articles, and standardizes whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens)


class ExactMatch(BaseMetric):
    """Evaluates normalized exact match."""

    @property
    def name(self) -> str:
        return "exact_match"

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer or not sample.ground_truth:
            return EvalResult(
                metric_name=self.name, score=0.0, reason="Missing answer or ground truth."
            )

        ref_norm = normalize_answer(sample.ground_truth)
        hyp_norm = normalize_answer(sample.answer)

        is_match = ref_norm == hyp_norm

        return EvalResult(
            metric_name=self.name,
            score=1.0 if is_match else 0.0,
            reason="Matched exactly after normalization." if is_match else "Did not match exactly.",
        )
