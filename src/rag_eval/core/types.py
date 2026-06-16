"""Data models for the RAG evaluation framework."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TestSample(BaseModel):
    """A single evaluation sample."""

    __test__ = False #per evitare warning nei test

    question: str = Field(..., description="The user query / question.")
    answer: str = Field(default="", description="The generated answer from the RAG system.")
    ground_truth: str = Field(default="", description="The reference / expected answer.")
    contexts: list[str] = Field(default_factory=list, description="Retrieved context chunks.")
    metadata: dict = Field(default_factory=dict, description="Arbitrary metadata.")


class EvalResult(BaseModel):
    """Result of a single metric evaluation on one sample."""

    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(default="", description="LLM-generated explanation for the score.")
    metadata: dict = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Aggregated evaluation report across all samples and metrics."""

    results: list[EvalResult] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict, description="Metric name → avg score.")

    def add_result(self, result: EvalResult) -> None:
        self.results.append(result)

    def compute_summary(self) -> dict[str, float]:
        """Compute average scores per metric."""
        from collections import defaultdict

        scores: dict[str, list[float]] = defaultdict(list)
        for r in self.results:
            scores[r.metric_name].append(r.score)
        self.summary = {name: sum(s) / len(s) for name, s in scores.items()}
        return self.summary
