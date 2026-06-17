import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from rag_eval.core.types import TestSample, EvalResult
from rag_eval.metrics.base import BaseMetric
from rag_eval.pipeline.evaluator import Evaluator

class DummyMetric(BaseMetric):
    def __init__(self, name="dummy", score=1.0, should_fail=False):
        self._name = name
        self._score = score
        self.should_fail = should_fail
        
    @property
    def name(self) -> str:
        return self._name
        
    async def score(self, sample: TestSample) -> EvalResult:
        if self.should_fail:
            raise ValueError("Simulated failure")
        return EvalResult(
            metric_name=self.name, 
            score=self._score, 
            metadata={"prompt_tokens": 10, "estimated_cost": 0.05}
        )

@pytest.fixture
def samples():
    return [
        TestSample(question=f"Q{i}", answer=f"A{i}", ground_truth=f"GT{i}", contexts=[])
        for i in range(5)
    ]

@pytest.mark.asyncio
async def test_evaluator_success(samples, tmp_path):
    metric = DummyMetric()
    evaluator = Evaluator(metrics=[metric], cache_dir=str(tmp_path))
    
    report = await evaluator.evaluate_async(samples, batch_size=2, max_concurrency=2)
    
    assert len(report.results) == 5
    assert report.summary["dummy"] == 1.0
    assert report.summary["total_input_tokens"] == 50  # 5 * 10
    assert pytest.approx(report.summary["total_estimated_cost"]) == 0.25 # 5 * 0.05
    
    # Check that checkpoints were written
    checkpoints = list(tmp_path.glob("checkpoint_batch_*.json"))
    assert len(checkpoints) == 3  # 5 samples, batch size 2 -> 3 batches

@pytest.mark.asyncio
async def test_evaluator_error_handling(samples, tmp_path):
    metric_good = DummyMetric(name="good", score=1.0)
    metric_bad = DummyMetric(name="bad", should_fail=True)
    
    evaluator = Evaluator(metrics=[metric_good, metric_bad], cache_dir=str(tmp_path))
    
    # The bad metric will raise an error on every sample. The evaluator should catch it, log warning, and continue.
    report = await evaluator.evaluate_async(samples)
    
    assert len(report.results) == 5  # Only the good metric results should be present
    for res in report.results:
        assert res.metric_name == "good"
    assert "good" in report.summary
    assert "bad" not in report.summary
