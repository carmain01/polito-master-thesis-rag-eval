"""Evaluator — the main evaluation pipeline that orchestrates metrics over datasets."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from rag_eval.core.config import EvalConfig
from rag_eval.core.types import EvalReport, EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrates the evaluation of RAG samples across multiple metrics."""

    def __init__(
        self,
        metrics: Sequence[BaseMetric],
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
            # Pydantic v2 provides model_dump() for robust serialization.
            # mode="json" ensures all types (like datetimes/UUIDs) are safely converted.
            data = report.model_dump(mode="json")
            with open(checkpoint_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    async def _evaluate_sample_metric(
        self, metric: BaseMetric, sample: TestSample, sample_index: int, sem: asyncio.Semaphore
    ) -> EvalResult | None:
        """Evaluate a single metric on a single sample, with error handling and concurrency limits."""
        async with sem:
            try:
                result = await metric.score(sample)
                result.sample_index = sample_index
                return result
            except Exception as e:
                logger.warning(f"Metric '{metric.name}' failed on sample: {e}")
                return None

    async def evaluate_async(
        self,
        samples: Sequence[TestSample],
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
            enumerated_samples = list(enumerate(samples))
            for i in range(0, len(enumerated_samples), batch_size):
                batch_samples = enumerated_samples[i : i + batch_size]

                coroutines = []
                for sample_idx, sample in batch_samples:
                    for metric in self.metrics:
                        coroutines.append(self._evaluate_sample_metric(metric, sample, sample_idx, sem))

                # Run the batch concurrently
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

        # Get unique LLM clients from metrics
        llm_clients = {
            metric.llm
            for metric in self.metrics
            if hasattr(metric, "llm") and metric.llm is not None
        }

        if llm_clients:
            # Prefer global usage tracker from LLM clients (avoids missing JSON mode tokens)
            for llm in llm_clients:
                if hasattr(llm, "usage"):
                    total_input_tokens += llm.usage.total_input_tokens
                    total_output_tokens += llm.usage.total_output_tokens
                    total_cost += llm.usage.total_cost
        else:
            # Fallback to per-result metadata
            for res in report.results:
                total_input_tokens += res.metadata.get("prompt_tokens", 0)
                total_output_tokens += res.metadata.get("completion_tokens", 0)
                total_cost += res.metadata.get("estimated_cost", 0.0)

        report.summary["total_input_tokens"] = total_input_tokens
        report.summary["total_output_tokens"] = total_output_tokens
        report.summary["total_estimated_cost"] = total_cost

    def evaluate(
        self,
        samples: Sequence[TestSample],
        batch_size: int = 50,
        max_concurrency: int = 10,
    ) -> EvalReport:
        """Synchronous wrapper for evaluate_async."""
        return asyncio.run(self.evaluate_async(samples, batch_size, max_concurrency))
