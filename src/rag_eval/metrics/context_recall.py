"""Context recall metric — measures if contexts cover the ground truth."""

from __future__ import annotations

import os
from string import Template

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric
from rag_eval.utils.llm import LLMClient

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


class ContextRecall(BaseMetric):
    """Evaluates if the retrieved contexts contain all information from the ground truth."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        with open(os.path.join(PROMPTS_DIR, "context_recall.prompt"), encoding="utf-8") as f:
            self.prompt_template = Template(f.read())

    @property
    def name(self) -> str:
        return "context_recall"

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.ground_truth:
            return EvalResult(metric_name=self.name, score=0.0, reason="No ground truth provided.")

        if not sample.contexts:
            return EvalResult(metric_name=self.name, score=0.0, reason="No contexts provided.")

        contexts_str = "\n".join(f"[{i}] {c}" for i, c in enumerate(sample.contexts))

        prompt = self.prompt_template.safe_substitute(
            ground_truth=sample.ground_truth, contexts=contexts_str
        )

        try:
            result_json = await self.llm.complete_json(prompt=prompt)
            if not isinstance(result_json, dict):
                raise ValueError(f"LLM did not return a JSON dictionary: {result_json}")

            statements = result_json.get("statements")
            if statements is None:
                raise ValueError(f"LLM response missing 'statements' key: {result_json}")
            
            if not statements:
                score = 0.0
            else:
                covered = sum(1 for s in statements if s.get("is_covered", False))
                score = float(covered) / len(statements)
                
            reason = str(result_json.get("reason", ""))
            return EvalResult(
                metric_name=self.name,
                score=score,
                reason=reason,
                metadata={"statements": statements},
            )
        except Exception as e:
            err_msg = str(e) or repr(e)
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=f"Failed to evaluate: {err_msg}",
                metadata={"error": err_msg},
            )
