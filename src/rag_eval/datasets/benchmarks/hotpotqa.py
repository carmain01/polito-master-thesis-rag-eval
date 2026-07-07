"""HotpotQA benchmark adapter."""

from __future__ import annotations

from rag_eval.core.types import TestSample
from rag_eval.datasets.benchmarks.base import BaseBenchmarkAdapter


class HotpotQAAdapter(BaseBenchmarkAdapter):
    """Adapter for the HotpotQA benchmark (multi-hop reasoning)."""

    def load(self, split: str = "validation", max_samples: int | None = None) -> list[TestSample]:
        dataset = self._load_hf_dataset("hotpot_qa", name="distractor", split=split)
        
        samples: list[TestSample] = []
        for i, item in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
                
            contexts = []
            if "context" in item:
                titles = item["context"]["title"]
                sentences = item["context"]["sentences"]
                for t, s_list in zip(titles, sentences):
                    ctx = f"Title: {t}\n" + "".join(s_list)
                    contexts.append(ctx)
            
            sample = TestSample(
                question=item["question"],
                ground_truth=item["answer"],
                contexts=contexts,
                metadata={
                    "level": item.get("level"), 
                    "type": item.get("type"),
                    "id": item.get("id")
                }
            )
            samples.append(sample)
            
        return samples
