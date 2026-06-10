"""Evaluator — the main evaluation pipeline that orchestrates metrics over datasets."""

from __future__ import annotations

import asyncio

from rag_eval.core.config import EvalConfig
from rag_eval.core.types import EvalReport, TestSample
from rag_eval.metrics.base import BaseMetric


class Evaluator:
    """Orchestrates the evaluation of RAG samples across multiple metrics."""

    def __init__(
        self,
        metrics: list[BaseMetric],
        config: EvalConfig | None = None,
    ) -> None:
        self.metrics = metrics
        self.config = config or EvalConfig()

    async def evaluate_async(self, samples: list[TestSample]) -> EvalReport:
        """Run all metrics on all samples asynchronously."""
        report = EvalReport()

        for metric in self.metrics:
            results = await metric.score_batch(samples)
            for result in results:
                report.add_result(result)

        report.compute_summary()
        return report

    def evaluate(self, samples: list[TestSample]) -> EvalReport:
        """Synchronous wrapper for evaluate_async."""
        return asyncio.run(self.evaluate_async(samples))
