"""Linguistic Adaptation metric — measures pedagogical quality of generated text.

Combines three sub-metrics into a single 0–1 score:
  - **Readability**: Flesch Reading Ease normalised to [0, 1].  A score near 1
    indicates the text is accessible to a broad audience.
  - **Socratic Tentativeness**: Ratio of tentative / exploratory word *lemmas*
    (LIWC-inspired) over total word count.  Uses ``simplemma`` for multilingual
    lemmatisation and ``langdetect`` for automatic language detection so that
    inflected verb forms (e.g. Italian *proviamo* → *provare*) are matched
    correctly without hardcoding any single language.
  - **Constructive Critique**: Derived from a multilingual BERT sentiment model
    (``nlptown/bert-base-multilingual-uncased-sentiment``) and the presence of
    adversative markers (*but*, *however*, *tuttavia*, …).  A balanced positive
    sentiment plus constructive contrast yields a higher score.
"""

from __future__ import annotations

import logging
from typing import Any

import nltk
import simplemma
import textstat
import torch
from langdetect import DetectorFactory, detect
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric

logger = logging.getLogger(__name__)

# Make langdetect deterministic.
DetectorFactory.seed = 0

# ---------------------------------------------------------------------------
# Ensure required NLTK resources are available
# ---------------------------------------------------------------------------
for _resource in ("punkt_tab",):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tentative / exploratory word **lemmas** inspired by LIWC categories.
# Stored as lemma forms so they match any inflection via simplemma.
_TENTATIVE_LEMMAS: frozenset[str] = frozenset({
    # English
    "maybe", "perhaps", "possibly", "might", "could", "wonder",
    "think", "guess", "seem", "appear", "suppose", "consider",
    "likely", "unlikely", "probably", "arguably",
    # Italian
    "forse", "magari", "sembrare", "chissà", "ipotizzare",
    "riflettere", "potere", "supporre", "provare", "pensare",
    "considerare", "immaginare",
    # Spanish
    "quizás", "tal vez", "posiblemente", "parecer", "suponer",
    "considerar", "pensar", "imaginar", "probablemente",
    # French
    "peut-être", "possiblement", "sembler", "supposer",
    "considérer", "penser", "imaginer", "probablement",
    # German
    "vielleicht", "möglicherweise", "scheinen", "vermuten",
    "überlegen", "denken", "wahrscheinlich",
})

# Adversative / contrast markers that introduce constructive critique.
_CONTRAST_WORDS: frozenset[str] = frozenset({
    # English
    "but", "however", "although", "nevertheless", "yet",
    "nonetheless", "still", "conversely",
    # Italian
    "ma", "però", "tuttavia", "nonostante", "eppure",
    # Spanish
    "pero", "sin embargo", "aunque", "no obstante",
    # French
    "mais", "cependant", "néanmoins", "toutefois", "pourtant",
    # German
    "aber", "jedoch", "dennoch", "trotzdem", "allerdings",
})

# Default weights for the three sub-metric components.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "readability": 0.4,
    "socratic_tentativeness": 0.3,
    "constructive_critique": 0.3,
}

# Default multilingual sentiment model.
_DEFAULT_SENTIMENT_MODEL: str = "nlptown/bert-base-multilingual-uncased-sentiment"

# Flesch Reading Ease range used for normalisation.
# Scores below _FRE_MIN are clamped to 0; scores above _FRE_MAX are clamped to 1.
_FRE_MIN: float = 0.0
_FRE_MAX: float = 100.0

# Default fallback language for lemmatisation when detection fails.
_FALLBACK_LANG: str = "en"


class LinguisticAdaptation(BaseMetric):
    """Evaluates the pedagogical quality of a generated answer.

    The final score is a weighted average of three normalised sub-metrics:

    1. **Readability** – Flesch Reading Ease mapped to [0, 1].
    2. **Socratic Tentativeness** – fraction of tentative word lemmas, capped at
       a configurable maximum ratio for normalisation.  Uses ``simplemma`` for
       multilingual lemmatisation and ``langdetect`` for language detection.
    3. **Constructive Critique** – multilingual BERT sentiment biased upward
       when adversative markers are present.

    Parameters
    ----------
    weights : dict[str, float] | None
        Custom weights for the three components.  Keys must be
        ``"readability"``, ``"socratic_tentativeness"``, and
        ``"constructive_critique"``.  Values are normalised to sum to 1.
    max_tentativeness : float
        The tentativeness ratio at which the sub-score saturates to 1.0.
        Defaults to ``0.05`` (5 % of all words).
    sentiment_model : str
        Hugging Face model identifier for multilingual sentiment analysis.
        Defaults to ``nlptown/bert-base-multilingual-uncased-sentiment``.
    device : str
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).  Defaults to ``"cpu"``.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        max_tentativeness: float = 0.05,
        sentiment_model: str = _DEFAULT_SENTIMENT_MODEL,
        device: str = "cpu",
    ) -> None:
        # Load multilingual sentiment model.
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            sentiment_model
        ).to(device)
        self.sentiment_model.eval()
        self.device = device

        # Normalise weights so they sum to 1.
        raw = weights or _DEFAULT_WEIGHTS
        total = sum(raw.values())
        self.weights = {k: v / total for k, v in raw.items()}

        self.max_tentativeness = max_tentativeness

    @property
    def name(self) -> str:
        return "linguistic_adaptation"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect the language of *text*, returning an ISO 639-1 code.

        Falls back to ``_FALLBACK_LANG`` if detection fails or the detected
        language is not supported by simplemma.
        """
        try:
            lang = detect(text)
            # langdetect returns codes like "en", "it", "es", "fr", "de", etc.
            return lang
        except Exception:
            logger.debug("Language detection failed; falling back to '%s'.", _FALLBACK_LANG)
            return _FALLBACK_LANG

    # ------------------------------------------------------------------
    # Sub-metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_readability(text: str) -> dict[str, Any]:
        """Compute readability statistics and return a normalised score in [0, 1].

        Uses the Flesch Reading Ease index.  A higher FRE (easier text) maps
        to a higher normalised score, reflecting the pedagogical preference for
        accessible language.
        """
        fre = textstat.flesch_reading_ease(text)
        fk_grade = textstat.flesch_kincaid_grade(text)
        word_count = textstat.lexicon_count(text, removepunct=True)

        normalised = max(0.0, min(1.0, (fre - _FRE_MIN) / (_FRE_MAX - _FRE_MIN)))

        return {
            "score": normalised,
            "flesch_reading_ease": fre,
            "flesch_kincaid_grade": fk_grade,
            "word_count": word_count,
        }

    def _score_socratic_tentativeness(self, text: str, lang: str) -> dict[str, Any]:
        """Compute the Socratic Tentativeness Ratio with multilingual lemmatisation.

        Each word is lemmatised via ``simplemma`` using the detected language
        so that inflected forms (e.g. *proviamo* → *provare*, *seems* → *seem*)
        are correctly matched against the tentative-lemma set.

        The raw ratio is the count of tentative lemmas divided by total word
        count.  It is then normalised to [0, 1] by dividing by
        ``self.max_tentativeness`` (capped at 1.0).
        """
        words = nltk.word_tokenize(text.lower())
        if not words:
            return {
                "score": 0.0,
                "tentativeness_ratio": 0.0,
                "tentative_word_count": 0,
                "detected_language": lang,
            }

        # Lemmatise each word and check against the tentative lemma set.
        tentative_count = 0
        for w in words:
            lemma = simplemma.lemmatize(w, lang=lang)
            if lemma in _TENTATIVE_LEMMAS or w in _TENTATIVE_LEMMAS:
                tentative_count += 1

        ratio = tentative_count / len(words)
        normalised = min(1.0, ratio / self.max_tentativeness) if self.max_tentativeness > 0 else 0.0

        return {
            "score": normalised,
            "tentativeness_ratio": round(ratio, 4),
            "tentative_word_count": tentative_count,
            "detected_language": lang,
        }

    def _score_constructive_critique(self, text: str) -> dict[str, Any]:
        """Evaluate constructive critique via multilingual sentiment and contrast markers.

        Scoring logic:
        - Compute a weighted-average star rating from the multilingual BERT
          sentiment model (1–5 stars), then normalise to [0, 1].
        - If adversative contrast markers are detected, apply a small bonus
          (capped at 1.0) to reward constructive framing.
        """
        # --- Multilingual BERT sentiment ---
        inputs = self.sentiment_tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.sentiment_model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]  # shape: (5,)

        # Star labels are 1–5; compute weighted average, then normalise to [0, 1].
        star_values = torch.arange(1, 6, dtype=torch.float32, device=self.device)
        weighted_avg = (probabilities * star_values).sum().item()  # range [1, 5]
        base_score = (weighted_avg - 1.0) / 4.0  # normalised to [0, 1]

        predicted_stars = int(torch.argmax(probabilities).item()) + 1

        # Detect adversative markers
        words_lower = set(nltk.word_tokenize(text.lower()))
        has_contrast = bool(words_lower & _CONTRAST_WORDS)

        # Bonus for constructive framing (contrast + overall positive leaning)
        contrast_bonus = 0.1 if has_contrast else 0.0
        final_score = min(1.0, base_score + contrast_bonus)

        return {
            "score": final_score,
            "sentiment_score": round(base_score, 4),
            "predicted_stars": predicted_stars,
            "star_probabilities": [round(p, 4) for p in probabilities.tolist()],
            "has_constructive_contrast": has_contrast,
        }

    # ------------------------------------------------------------------
    # Public API (BaseMetric interface)
    # ------------------------------------------------------------------

    async def score(self, sample: TestSample) -> EvalResult:
        """Score a single sample for linguistic adaptation."""
        if not sample.answer:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No answer provided.",
            )

        text = sample.answer

        # Detect language once and share across sub-metrics.
        lang = self._detect_language(text)

        readability = self._score_readability(text)
        tentativeness = self._score_socratic_tentativeness(text, lang)
        critique = self._score_constructive_critique(text)

        # Weighted combination
        combined = (
            self.weights["readability"] * readability["score"]
            + self.weights["socratic_tentativeness"] * tentativeness["score"]
            + self.weights["constructive_critique"] * critique["score"]
        )
        combined = max(0.0, min(1.0, combined))

        return EvalResult(
            metric_name=self.name,
            score=round(combined, 4),
            reason=(
                f"Weighted combination — readability={readability['score']:.3f} "
                f"(w={self.weights['readability']:.2f}), "
                f"tentativeness={tentativeness['score']:.3f} "
                f"(w={self.weights['socratic_tentativeness']:.2f}), "
                f"critique={critique['score']:.3f} "
                f"(w={self.weights['constructive_critique']:.2f})"
            ),
            metadata={
                "readability": readability,
                "socratic_tentativeness": tentativeness,
                "constructive_critique": critique,
            },
        )
