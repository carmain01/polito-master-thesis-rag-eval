"""RECALL benchmark adapter."""

from __future__ import annotations

from datasets import load_dataset
from rag_eval.core.types import TestSample
from rag_eval.datasets.benchmarks.base import BaseBenchmarkAdapter


class RecallAdapter(BaseBenchmarkAdapter):
    """Adapter for the RECALL benchmark.
    
    Supports flexible loading from either a HuggingFace repository or a local file.
    """

    def __init__(self, dataset_path: str = "recall-benchmark", name: str | None = None) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.name = name

    def load(self, split: str = "validation", max_samples: int | None = None) -> list[TestSample]:
        if self.dataset_path.endswith(".json") or self.dataset_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        else:
            try:
                dataset = self._load_hf_dataset(self.dataset_path, name=self.name, split=split)
            except Exception:
                # Fallback if standard split doesn't exist
                dataset = self._load_hf_dataset(self.dataset_path, name=self.name, split="train")
            
        samples: list[TestSample] = []
        for i, item in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
            
            question = item.get("question", item.get("query", ""))
            ground_truth = item.get("ground_truth", item.get("answer", ""))
            
            contexts = []
            if "context" in item:
                ctx = item["context"]
                contexts = [ctx] if isinstance(ctx, str) else ctx
            elif "docs" in item:
                contexts = item["docs"]
            
            sample = TestSample(
                question=question,
                ground_truth=ground_truth,
                contexts=contexts,
                metadata={k: v for k, v in item.items() if k not in ["question", "query", "ground_truth", "answer", "context", "docs"]}
            )
            samples.append(sample)
            
        return samples
