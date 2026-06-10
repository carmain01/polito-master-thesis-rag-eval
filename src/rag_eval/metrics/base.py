"""Base metric interface — all metrics inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_eval.core.types import EvalResult, TestSample


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this metric."""
        ...

    @abstractmethod
    async def score(self, sample: TestSample) -> EvalResult:
        """Evaluate a single sample and return a result."""
        ...

    async def score_batch(self, samples: list[TestSample]) -> list[EvalResult]:
        """Evaluate a batch of samples. Override for optimized batch processing."""
        results = []
        for sample in samples:
            result = await self.score(sample)
            results.append(result)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
