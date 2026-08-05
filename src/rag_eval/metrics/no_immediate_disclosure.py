"""No-Immediate-Disclosure & Helpfulness metric — LLM-judged pedagogical quality.

Evaluates two complementary aspects of a tutoring response:

  - **No-Immediate-Disclosure** (Maurya et al.): Evaluates how well the tutor
    avoids revealing the final answer or critical solution steps on a continuous
    scale (0.0 = full disclosure, 0.33 = partial disclosure, 0.67 = hints,
    1.0 = pure scaffolding).
  - **Helpfulness** (Tack & Piech): How effectively the tutor's response
    supports the student's learning on a 1–5 scale.

The two sub-scores are combined into a single 0–1 primary score via a
configurable weighted average. Both sub-scores and their reasoning are
available in the ``EvalResult.metadata`` dictionary.
"""

from __future__ import annotations

import logging
import os
from string import Template

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric
from rag_eval.utils.llm import LLMClient

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

# Default weights for combining the two sub-metrics.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "no_immediate_disclosure": 0.5,
    "helpfulness": 0.5,
}


class NoImmediateDisclosure(BaseMetric):
    """Evaluates whether a tutor follows the No-Immediate-Disclosure principle
    and how helpful its response is for student learning.

    Both sub-scores are computed via LLM-as-a-judge and combined into a single
    primary score in [0, 1].

    Parameters
    ----------
    llm_client : LLMClient
        The LLM client used to evaluate the tutor response.
    weights : dict[str, float] | None
        Custom weights for the two sub-metrics. Keys must be
        ``"no_immediate_disclosure"`` and ``"helpfulness"``.
        Values are normalised to sum to 1.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.llm = llm_client

        with open(
            os.path.join(PROMPTS_DIR, "no_immediate_disclosure.prompt"),
            encoding="utf-8",
        ) as f:
            self.nid_prompt_template = Template(f.read())

        with open(
            os.path.join(PROMPTS_DIR, "helpfulness.prompt"),
            encoding="utf-8",
        ) as f:
            self.helpfulness_prompt_template = Template(f.read())

        # Normalise weights so they sum to 1.
        raw = weights or _DEFAULT_WEIGHTS
        total = sum(raw.values())
        self.weights = {k: v / total for k, v in raw.items()}

    @property
    def name(self) -> str:
        return "no_immediate_disclosure"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _evaluate_nid(
        self, context: str, student_query: str, tutor_response: str
    ) -> dict:
        """Evaluate No-Immediate-Disclosure via LLM judge on a continuous 0.0–1.0 scale."""
        prompt = self.nid_prompt_template.safe_substitute(
            context=context,
            student_query=student_query,
            tutor_response=tutor_response,
        )
        result_json = await self.llm.complete_json(prompt=prompt)

        reasoning = str(result_json.get("reasoning", ""))

        try:
            score = float(result_json.get("score", 0.0))
        except (ValueError, TypeError):
            score = 0.0
        score = max(0.0, min(1.0, score))

        # Infer revealed_immediately if missing, or use LLM boolean
        revealed = bool(result_json.get("revealed_immediately", score <= 0.33))

        return {
            "reasoning": reasoning,
            "revealed_immediately": revealed,
            "score": round(score, 4),
        }

    async def _evaluate_helpfulness(
        self, context: str, student_query: str, tutor_response: str
    ) -> dict:
        """Evaluate Helpfulness via LLM judge (1–5 scale)."""
        prompt = self.helpfulness_prompt_template.safe_substitute(
            context=context,
            student_query=student_query,
            tutor_response=tutor_response,
        )
        result_json = await self.llm.complete_json(prompt=prompt)

        reasoning = str(result_json.get("reasoning", ""))

        try:
            raw_score = int(result_json.get("score", 1))
        except (ValueError, TypeError):
            raw_score = 1
        raw_score = max(1, min(5, raw_score))

        # Normalise from 1–5 to 0–1.
        normalised = (raw_score - 1) / 4.0

        return {
            "reasoning": reasoning,
            "raw_score": raw_score,
            "score": round(normalised, 4),
        }

    # ------------------------------------------------------------------
    # Public API (BaseMetric interface)
    # ------------------------------------------------------------------

    async def score(self, sample: TestSample) -> EvalResult:
        """Score a single sample for no-immediate-disclosure and helpfulness.

        The ``TestSample`` fields are mapped as follows:

        - ``sample.contexts`` → exercise context (joined).
        - ``sample.question`` → student query / error.
        - ``sample.answer`` → AI tutor response to evaluate.
        """
        if not sample.answer:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No answer provided.",
            )

        context = "\n".join(sample.contexts) if sample.contexts else ""
        student_query = sample.question
        tutor_response = sample.answer

        try:
            nid_result = await self._evaluate_nid(context, student_query, tutor_response)
            helpfulness_result = await self._evaluate_helpfulness(
                context, student_query, tutor_response
            )

            # Weighted combination
            combined = (
                self.weights["no_immediate_disclosure"] * nid_result["score"]
                + self.weights["helpfulness"] * helpfulness_result["score"]
            )
            combined = max(0.0, min(1.0, round(combined, 4)))

            reason = (
                f"NID={'NOT revealed' if not nid_result['revealed_immediately'] else 'REVEALED'} "
                f"(score={nid_result['score']:.2f}, w={self.weights['no_immediate_disclosure']:.2f}), "
                f"Helpfulness={helpfulness_result['raw_score']}/5 "
                f"(norm={helpfulness_result['score']:.2f}, w={self.weights['helpfulness']:.2f})"
            )

            return EvalResult(
                metric_name=self.name,
                score=combined,
                reason=reason,
                metadata={
                    "no_immediate_disclosure": nid_result,
                    "helpfulness": helpfulness_result,
                },
            )
        except Exception as e:
            err_msg = str(e) or repr(e)
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=f"Failed to evaluate: {err_msg}",
                metadata={"error": err_msg},
            )
