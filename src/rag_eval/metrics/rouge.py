"""ROUGE Score metric — measures recall-focused n-gram overlap."""

from __future__ import annotations

from rouge_score import rouge_scorer

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric


class ROUGE(BaseMetric):
    """Evaluates ROUGE score."""

    def __init__(self, variant: str = "rougeL") -> None:
        if variant not in ["rouge1", "rouge2", "rougeL"]:
            raise ValueError("ROUGE variant must be rouge1, rouge2, or rougeL.")
        self.variant = variant
        self.scorer = rouge_scorer.RougeScorer([self.variant], use_stemmer=True)

    @property
    def name(self) -> str:
        return self.variant

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer or not sample.ground_truth:
            return EvalResult(metric_name=self.name, score=0.0, reason="Missing answer or ground truth.")

        scores = self.scorer.score(sample.ground_truth, sample.answer)
        score_obj = scores[self.variant]

        return EvalResult(
            metric_name=self.name,
            score=score_obj.fmeasure,
            reason=f"Computed {self.variant} F1 using rouge-score",
            metadata={
                "precision": score_obj.precision,
                "recall": score_obj.recall,
                "f1": score_obj.fmeasure
            }
        )
