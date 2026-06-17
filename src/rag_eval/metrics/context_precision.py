"""Context precision metric — measures if retrieved contexts are relevant to the question."""

from __future__ import annotations

import os
from string import Template

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric
from rag_eval.utils.llm import LLMClient

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

class ContextPrecision(BaseMetric):
    """Evaluates whether the retrieved context chunks are relevant to the question."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        with open(os.path.join(PROMPTS_DIR, "context_precision.prompt"), encoding="utf-8") as f:
            self.prompt_template = Template(f.read())

    @property
    def name(self) -> str:
        return "context_precision"

    async def score(self, sample: TestSample) -> EvalResult:
        if not sample.contexts:
            return EvalResult(metric_name=self.name, score=0.0, reason="No contexts provided.")

        contexts_str = "\n".join(f"[{i}] {c}" for i, c in enumerate(sample.contexts))

        prompt = self.prompt_template.safe_substitute(
            question=sample.question,
            contexts=contexts_str
        )

        try:
            result_json = await self.llm.complete_json(prompt=prompt)
            score = float(result_json.get("score", 0.0))
            reason = str(result_json.get("reason", ""))
            return EvalResult(
                metric_name=self.name,
                score=score,
                reason=reason,
                metadata={"evaluations": result_json.get("evaluations", [])}
            )
        except Exception as e:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=f"Failed to evaluate: {str(e)}",
                metadata={"error": str(e)}
            )
