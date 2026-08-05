"""Sentiment metric — classifies the pedagogical intent of a tutoring system's answer.

Assigns one of three states:
  A) Concept Teaching — the system is teaching a new academic concept.
  B) Error Remediation — the system is correcting a student's mistake.
  C) Assessment / Socratic Questioning — the system is evaluating the student.
"""

from __future__ import annotations

import os
from string import Template

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric
from rag_eval.utils.llm import LLMClient

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

# The primary score is the LLM's classification confidence (0.0–1.0).
# The actual pedagogical state (A/B/C) is stored in metadata["state"]
# and metadata["state_label"].  This avoids the misleading numeric
# encoding (A=1.0, B=0.5, C=0.0) that produced meaningless summary
# statistics when averaged across samples.


class State(BaseMetric):
    """Classifies the pedagogical intent behind a tutoring system's answer.

    The metric uses an LLM to determine whether the system's answer is:
      - **State A**: teaching a new concept,
      - **State B**: remediating an error, or
      - **State C**: assessing / questioning the student.

    The classification is stored in :pyattr:`EvalResult.metadata` under the keys
    ``state`` (``"A"``/``"B"``/``"C"``) and ``state_label`` (human-readable).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        with open(os.path.join(PROMPTS_DIR, "state.prompt"), encoding="utf-8") as f:
            self.prompt_template = Template(f.read())

    @property
    def name(self) -> str:
        return "state"

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer:
            return EvalResult(metric_name=self.name, score=0.0, reason="No answer provided.")

        prompt = self.prompt_template.safe_substitute(
            question=sample.question, answer=sample.answer
        )

        try:
            result_json = await self.llm.complete_json(prompt=prompt)

            raw_state = str(result_json.get("state", "")).strip().upper()
            state = ""
            for char in raw_state:
                if char in ("A", "B", "C"):
                    state = char
                    break

            state_label = str(result_json.get("state_label", ""))
            try:
                confidence = float(result_json.get("confidence", 0.0))
            except (ValueError, TypeError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            reason = str(result_json.get("reason", ""))

            # Use classification confidence as the primary score.
            score = confidence

            return EvalResult(
                metric_name=self.name,
                score=score,
                reason=f"[State {state}] {reason}",
                metadata={
                    "state": state,
                    "state_label": state_label,
                    "confidence": confidence,
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
