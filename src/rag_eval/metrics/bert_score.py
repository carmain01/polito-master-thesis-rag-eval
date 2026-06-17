"""BERTScore metric — measures semantic overlap using a language model."""

from __future__ import annotations

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

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer or not sample.ground_truth:
            return EvalResult(metric_name=self.name, score=0.0, reason="Missing answer or ground truth.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # bert_score returns (P, R, F1) tensors
            P, R, F1 = bert_score(
                [sample.answer],
                [sample.ground_truth],
                model_type=self.model_type,
                lang="en",
                verbose=False
            )

        f1_score = F1.item()

        return EvalResult(
            metric_name=self.name,
            score=f1_score,
            reason=f"Computed BERTScore using {self.model_type}",
            metadata={
                "precision": P.item(),
                "recall": R.item(),
                "f1": f1_score
            }
        )
