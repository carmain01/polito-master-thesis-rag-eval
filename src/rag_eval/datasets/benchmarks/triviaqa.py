"""TriviaQA benchmark adapter."""

from __future__ import annotations

from rag_eval.core.types import TestSample
from rag_eval.datasets.benchmarks.base import BaseBenchmarkAdapter


class TriviaQAAdapter(BaseBenchmarkAdapter):
    """Adapter for the TriviaQA benchmark."""

    def load(self, split: str = "validation", max_samples: int | None = None) -> list[TestSample]:
        dataset = self._load_hf_dataset("trivia_qa", name="rc", split=split)
        
        samples: list[TestSample] = []
        for i, item in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
            
            answer_data = item.get("answer", {})
            ground_truth = answer_data.get("value", "")
            
            contexts = []
            if "search_results" in item:
                for search_res in item["search_results"].get("search_context", []):
                    if search_res:
                        contexts.append(search_res)
            
            sample = TestSample(
                question=item["question"],
                ground_truth=ground_truth,
                contexts=contexts,
                metadata={
                    "answer_aliases": answer_data.get("aliases", []),
                    "question_id": item.get("question_id")
                }
            )
            samples.append(sample)
            
        return samples
