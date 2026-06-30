import pytest

from rag_eval.core.registry import MetricRegistry
from rag_eval.metrics.base import BaseMetric
from rag_eval.metrics.exact_match import ExactMatch


class CustomMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "custom_metric"

    async def compute(self, **kwargs) -> dict:
        return {"score": 1.0}

    async def score(self, **kwargs) -> float:
        return 1.0


def test_registry_register_and_create():
    MetricRegistry.register(CustomMetric)
    metric = MetricRegistry.create_metric("custom_metric")
    assert isinstance(metric, CustomMetric)


def test_registry_discover_metrics():
    # Calling create_metric with a known standard metric will auto-discover
    metric = MetricRegistry.create_metric("exact_match")
    assert isinstance(metric, ExactMatch)


def test_registry_case_insensitivity():
    metric = MetricRegistry.create_metric("ExactMatch")
    assert isinstance(metric, ExactMatch)


def test_registry_not_found():
    with pytest.raises(ValueError, match="Metric 'unknown' not found in registry"):
        MetricRegistry.create_metric("unknown")
