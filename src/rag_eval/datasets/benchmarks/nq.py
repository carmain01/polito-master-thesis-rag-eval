"""Natural Questions benchmark adapter."""

from __future__ import annotations

from rag_eval.core.types import TestSample
from rag_eval.datasets.benchmarks.base import BaseBenchmarkAdapter


class NaturalQuestionsAdapter(BaseBenchmarkAdapter):
    """Adapter for the Natural Questions (NQ-Open) benchmark."""

    def load(self, split: str = "validation", max_samples: int | None = None) -> list[TestSample]:
        dataset = self._load_hf_dataset("nq_open", split=split)
        
        samples: list[TestSample] = []
        for i, item in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
            
            # nq_open has a list of possible answers
            answers = item.get("answer", [])
            ground_truth = answers[0] if answers else ""
            
            sample = TestSample(
                question=item["question"],
                ground_truth=ground_truth,
                contexts=[],  # nq_open does not provide contexts by default
                metadata={"all_answers": answers}
            )
            samples.append(sample)
            
        return samples
