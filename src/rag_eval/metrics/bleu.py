"""BLEU Score metric — measures n-gram overlap between answer and ground truth."""

from __future__ import annotations

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric


class BLEU(BaseMetric):
    """Evaluates BLEU score for n-gram overlap."""

    def __init__(self, variant: int = 4) -> None:
        if variant not in [1, 2, 3, 4]:
            raise ValueError("BLEU variant must be 1, 2, 3, or 4.")
        self.variant = variant
        self.weights = {
            1: (1.0, 0, 0, 0),
            2: (0.5, 0.5, 0, 0),
            3: (0.33, 0.33, 0.33, 0),
            4: (0.25, 0.25, 0.25, 0.25),
        }[self.variant]

    @property
    def name(self) -> str:
        return f"bleu-{self.variant}"

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer or not sample.ground_truth:
            return EvalResult(
                metric_name=self.name, score=0.0, reason="Missing answer or ground truth."
            )

        try:
            # Tokenize simply
            ref_tokens = nltk.word_tokenize(sample.ground_truth.lower())
            hyp_tokens = nltk.word_tokenize(sample.answer.lower())
        except LookupError:
            nltk.download("punkt")
            nltk.download("punkt_tab")
            ref_tokens = nltk.word_tokenize(sample.ground_truth.lower())
            hyp_tokens = nltk.word_tokenize(sample.answer.lower())

        if not ref_tokens or not hyp_tokens:
            return EvalResult(metric_name=self.name, score=0.0, reason="Empty after tokenization.")

        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            [ref_tokens], hyp_tokens, weights=self.weights, smoothing_function=smoothing
        )

        return EvalResult(
            metric_name=self.name,
            score=float(bleu_score),
            reason=f"Computed BLEU-{self.variant} using nltk",
        )
