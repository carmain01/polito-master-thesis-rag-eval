"""Evaluator — the main evaluation pipeline that orchestrates metrics over datasets."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, List

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from rag_eval.core.config import EvalConfig
from rag_eval.core.types import EvalReport, EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrates the evaluation of RAG samples across multiple metrics."""

    def __init__(
        self,
        metrics: list[BaseMetric],
        config: EvalConfig | None = None,
        cache_dir: str = ".rag_eval_cache",
    ) -> None:
        self.metrics = metrics
        self.config = config or EvalConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, report: EvalReport, batch_idx: int) -> None:
        """Save an intermediate checkpoint of the report."""
        checkpoint_path = self.cache_dir / f"checkpoint_batch_{batch_idx}.json"
        try:
            # We must serialize the Pydantic models. Assuming they have dict() or model_dump().
            # In Pydantic v2 it's model_dump(), v1 it's dict(). Let's use model_dump if available, else dict.
            data = {"results": [r.model_dump() if hasattr(r, 'model_dump') else r.dict() for r in report.results]}
            with open(checkpoint_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    async def _evaluate_sample_metric(
        self, metric: BaseMetric, sample: TestSample, sem: asyncio.Semaphore
    ) -> EvalResult | None:
        """Evaluate a single metric on a single sample, with error handling and concurrency limits."""
        async with sem:
            try:
                result = await metric.score(sample)
                return result
            except Exception as e:
                logger.warning(f"Metric '{metric.name}' failed on sample: {e}")
                return None

    async def evaluate_async(
        self, 
        samples: list[TestSample],
        batch_size: int = 50,
        max_concurrency: int = 10,
    ) -> EvalReport:
        """Run all metrics on all samples asynchronously with batching and progress tracking."""
        report = EvalReport()
        sem = asyncio.Semaphore(max_concurrency)
        total_tasks = len(samples) * len(self.metrics)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("[cyan]Evaluating...", total=total_tasks)

            # Process in batches
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i : i + batch_size]
                
                coroutines = []
                for sample in batch_samples:
                    for metric in self.metrics:
                        coroutines.append(self._evaluate_sample_metric(metric, sample, sem))

                # Run the batch concurrently
                results = []
                for coro in asyncio.as_completed(coroutines):
                    res = await coro
                    if res is not None:
                        report.add_result(res)
                    progress.advance(task_id)

                self._save_checkpoint(report, i // batch_size)

        report.compute_summary()
        self._aggregate_costs(report)
        return report

    def _aggregate_costs(self, report: EvalReport) -> None:
        """Aggregate total tokens and estimated costs from all results."""
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        for res in report.results:
            if "total_tokens" in res.metadata:
                pass # Just ensuring it exists, but we sum specific tokens if available
            input_toks = res.metadata.get("prompt_tokens", 0)
            output_toks = res.metadata.get("completion_tokens", 0)
            cost = res.metadata.get("estimated_cost", 0.0)

            total_input_tokens += input_toks
            total_output_tokens += output_toks
            total_cost += cost

        report.summary["total_input_tokens"] = total_input_tokens
        report.summary["total_output_tokens"] = total_output_tokens
        report.summary["total_estimated_cost"] = total_cost

    def evaluate(
        self, 
        samples: list[TestSample],
        batch_size: int = 50,
        max_concurrency: int = 10,
    ) -> EvalReport:
        """Synchronous wrapper for evaluate_async."""
        return asyncio.run(self.evaluate_async(samples, batch_size, max_concurrency))
