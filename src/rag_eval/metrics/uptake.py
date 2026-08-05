"""Uptake metric — measures how much an answer elaborates on the question and context.

Uses a BERT-based Next Utterance Classification (NUC) model to compute a
proxy Jensen–Shannon Divergence (pJSD) between source and target texts.
A higher score indicates stronger "uptake" — i.e. the answer meaningfully
engages with the source rather than ignoring or copy-pasting it.

Three sub-scores are produced:
  - **conversational_uptake**: question → answer elaboration.
  - **context_uptake**: retrieved context → answer elaboration.
  - **context_overlap**: lexical overlap baseline (%-IN-T) between context and answer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric

logger = logging.getLogger(__name__)

# Minimum number of tokens in the question for conversational uptake to be
# meaningful (short queries lack enough substance to measure elaboration).
_MIN_QUESTION_TOKENS = 5


class Uptake(BaseMetric):
    """Measures how much a generated answer elaborates on the question and context.

    The metric relies on a BERT model fine-tuned on the Next Utterance
    Classification (NUC) task.  It produces the probability that the answer is
    a *true continuation* of the source text, which serves as a proxy for
    conversational uptake (Demszky et al., 2021).

    Parameters
    ----------
    model_path : str
        Hugging Face model identifier or local path to the NUC-BERT model.
    device : str, optional
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).  Defaults to
        ``"cpu"`` for broad compatibility.
    max_length : int, optional
        Maximum combined token length for the (source, target) pair.
    """

    def __init__(
        self,
        model_path: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device: str = "cpu",
        max_length: int = 240,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            device
        )
        self.model.eval()

    @property
    def name(self) -> str:
        return "uptake"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_uptake_prob(self, source_text: str, target_text: str) -> float:
        """Return the probability that *target_text* is a true continuation of *source_text*.

        This corresponds to the softmax probability of the positive class
        (label 1) from the NUC classifier.
        """
        inputs = self.tokenizer(
            source_text,
            target_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            prob_true = torch.softmax(logits, dim=-1)[0][1].item()

        return float(prob_true)

    @staticmethod
    def _compute_word_overlap(source_text: str, target_text: str) -> float:
        """Compute the %-IN-T baseline: fraction of non-stop source tokens present in target."""
        try:
            from nltk.corpus import stopwords

            stop_words = set(stopwords.words("english"))
        except (ImportError, LookupError):
            # Graceful fallback when NLTK or its data is unavailable.
            stop_words: set[str] = set()
            logger.debug("NLTK stopwords unavailable; word-overlap computed without filtering.")

        source_tokens = [
            w.lower()
            for w in source_text.split()
            if w.isalnum() and w.lower() not in stop_words
        ]
        target_tokens = {
            w.lower()
            for w in target_text.split()
            if w.isalnum() and w.lower() not in stop_words
        }

        if not source_tokens:
            return 0.0

        overlap = sum(1 for t in source_tokens if t in target_tokens)
        return overlap / len(source_tokens)

    # ------------------------------------------------------------------
    # Public API (BaseMetric interface)
    # ------------------------------------------------------------------

    async def score(self, sample: TestSample) -> EvalResult:
        """Score a single sample for uptake.

        The primary ``score`` value is the **conversational uptake** (question →
        answer).  Context-level sub-scores are available in ``metadata``.
        """
        if not sample.answer:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No answer provided.",
            )

        metadata: dict[str, Any] = {}

        # --- Conversational uptake (question → answer) ---
        question_tokens = len(sample.question.split())
        if question_tokens < _MIN_QUESTION_TOKENS:
            conversational_uptake = None
            reason = (
                f"Question too short ({question_tokens} tokens < {_MIN_QUESTION_TOKENS}); "
                "conversational uptake not computed."
            )
        else:
            conversational_uptake = await asyncio.to_thread(
                self._compute_uptake_prob, sample.question, sample.answer
            )
            reason = "Computed conversational uptake via NUC model."
        metadata["conversational_uptake"] = conversational_uptake

        # --- Context uptake (context → answer) ---
        combined_context = " ".join(sample.contexts) if sample.contexts else ""
        if combined_context:
            context_uptake = await asyncio.to_thread(
                self._compute_uptake_prob, combined_context, sample.answer
            )
            context_overlap = await asyncio.to_thread(
                self._compute_word_overlap, combined_context, sample.answer
            )
            metadata["context_uptake"] = context_uptake
            metadata["context_overlap"] = context_overlap
        else:
            metadata["context_uptake"] = None
            metadata["context_overlap"] = None

        # The primary score is the conversational uptake when available,
        # otherwise fall back to the context uptake.
        primary_score: float
        if conversational_uptake is not None:
            primary_score = conversational_uptake
        elif metadata.get("context_uptake") is not None:
            primary_score = metadata["context_uptake"]
            reason = "Computed context uptake (question too short for conversational uptake)."
        else:
            primary_score = 0.0
            reason = "Insufficient data to compute any uptake score."

        primary_score = max(0.0, min(1.0, primary_score))

        return EvalResult(
            metric_name=self.name,
            score=primary_score,
            reason=reason,
            metadata=metadata,
        )
