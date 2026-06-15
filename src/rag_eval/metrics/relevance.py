"""Answer relevance metric — measures how well the answer addresses the question."""

from __future__ import annotations

import os
from string import Template

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric
from rag_eval.utils.llm import LLMClient

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

class AnswerRelevance(BaseMetric):
    """Evaluates how well the answer directly addresses the user's question."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        with open(os.path.join(PROMPTS_DIR, "relevance.prompt"), encoding="utf-8") as f:
            self.prompt_template = Template(f.read())

    @property
    def name(self) -> str:
        return "relevance"

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.answer:
            return EvalResult(metric_name=self.name, score=0.0, reason="No answer provided.")
        
        prompt = self.prompt_template.safe_substitute(
            question=sample.question,
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
            )
        except Exception as e:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=f"Failed to evaluate: {str(e)}",
                metadata={"error": str(e)}
            )
