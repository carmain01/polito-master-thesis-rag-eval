"""Faithfulness metric — measures whether the answer is supported by the retrieved contexts."""

from __future__ import annotations

import os
from string import Template

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric
from rag_eval.utils.llm import LLMClient

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

class Faithfulness(BaseMetric):
    """Evaluates if claims in the generated answer are grounded in the provided contexts."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        with open(os.path.join(PROMPTS_DIR, "faithfulness.prompt"), encoding="utf-8") as f:
            self.prompt_template = Template(f.read())

    @property
    def name(self) -> str:
        return "faithfulness"

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer:
            return EvalResult(metric_name=self.name, score=0.0, reason="No answer provided.")
        
        contexts_str = "\n".join(f"[{i}] {c}" for i, c in enumerate(sample.contexts))
        if not contexts_str:
            contexts_str = "No contexts provided."

        prompt = self.prompt_template.safe_substitute(
            contexts=contexts_str,
            answer=sample.answer
        )

        try:
            result_json = await self.llm.complete_json(prompt=prompt)
            score = float(result_json.get("score", 0.0))
            reason = str(result_json.get("reason", ""))
            return EvalResult(
                metric_name=self.name,
                score=score,
                reason=reason,
                metadata={"claims": result_json.get("claims", [])}
            )
        except Exception as e:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=f"Failed to evaluate: {str(e)}",
                metadata={"error": str(e)}
            )
