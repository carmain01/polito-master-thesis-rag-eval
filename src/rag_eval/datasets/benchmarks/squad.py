"""SQuAD 2.0 benchmark adapter."""

from __future__ import annotations

from rag_eval.core.types import TestSample
from rag_eval.datasets.benchmarks.base import BaseBenchmarkAdapter


class Squad2Adapter(BaseBenchmarkAdapter):
    """Adapter for the SQuAD 2.0 benchmark.
    
    Unanswerable questions (where answers text is empty) are filtered out
    by default based on the provided configuration.
    """

    def load(self, split: str = "validation", max_samples: int | None = None) -> list[TestSample]:
        dataset = self._load_hf_dataset("squad_v2", split=split)
        
        samples: list[TestSample] = []
        count = 0
        for item in dataset:
            if max_samples is not None and count >= max_samples:
                break
            
            answers = item.get("answers", {}).get("text", [])
            
            # Filter out unanswerable questions
            if not answers:
                continue
                
            ground_truth = answers[0]
            context = item.get("context", "")
            
            sample = TestSample(
                question=item["question"],
                ground_truth=ground_truth,
                contexts=[context] if context else [],
                metadata={"title": item.get("title"), "id": item.get("id")}
            )
            samples.append(sample)
            count += 1
            
        return samples
