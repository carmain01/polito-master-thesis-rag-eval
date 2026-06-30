"""Data models for the RAG evaluation framework."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TestSample(BaseModel):
    """A single evaluation sample."""

    __test__ = False  # per evitare warning nei test

    question: str = Field(..., description="The user query / question.")
    answer: str = Field(default="", description="The generated answer from the RAG system.")
    ground_truth: str = Field(default="", description="The reference / expected answer.")
    contexts: list[str] = Field(default_factory=list, description="Retrieved context chunks.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata.")


class EvalResult(BaseModel):
    """Result of a single metric evaluation on one sample."""

    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(default="", description="LLM-generated explanation for the score.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Aggregated evaluation report across all samples and metrics."""

    config: dict[str, Any] = Field(
        default_factory=dict, description="Configuration used for the evaluation."
    )
    results: list[EvalResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Metric name → avg score, plus cost aggregates."
    )

    def add_result(self, result: EvalResult) -> None:
        self.results.append(result)

    def compute_summary(self) -> dict[str, Any]:
        """Compute average scores per metric."""
        from collections import defaultdict

        scores: dict[str, list[float]] = defaultdict(list)
        for r in self.results:
            scores[r.metric_name].append(r.score)

        # Preserve existing non-float stats (like tokens/cost) when recomputing
        for name, s in scores.items():
            self.summary[name] = sum(s) / len(s)

        return self.summary

    @classmethod
    def load_from_json(cls, path: str) -> EvalReport:
        """Load an EvalReport from a JSON file (round-trip support)."""
        import json

        with open(path) as f:
            data = json.load(f)

        # Handle the case where the JSON contains metadata wrapper
        if "report" in data:
            data = data["report"]

        return cls.model_validate(data)
