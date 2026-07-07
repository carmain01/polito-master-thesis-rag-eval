"""RGB (Retrieval-Generated Benchmark) adapter."""

from __future__ import annotations

from datasets import load_dataset
from rag_eval.core.types import TestSample
from rag_eval.datasets.benchmarks.base import BaseBenchmarkAdapter


class RGBAdapter(BaseBenchmarkAdapter):
    """Adapter for the RGB (Retrieval-Generated Benchmark).
    
    Defaults to assuming standard field names like 'query' for question,
    'answer' for ground truth, and 'docs' or 'contexts' for retrieved context.
    """

    def __init__(self, dataset_path: str = "THUDM/RGB", name: str | None = None) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.name = name

    def load(self, split: str = "validation", max_samples: int | None = None) -> list[TestSample]:
        # Sometimes RGB datasets are loaded from json directly if not on HF.
        if self.dataset_path.endswith(".json") or self.dataset_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        else:
            dataset = self._load_hf_dataset(self.dataset_path, name=self.name, split=split)
            
        samples: list[TestSample] = []
        for i, item in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
            
            # Map query/question to question
            question = item.get("query", item.get("question", ""))
            # Map answer/ground_truth to ground_truth
            ground_truth = item.get("answer", item.get("ground_truth", ""))
            
            # Map docs/contexts to contexts
            contexts = []
            if "docs" in item:
                contexts = item["docs"]
                if isinstance(contexts, list) and len(contexts) > 0 and isinstance(contexts[0], dict):
                    contexts = [c.get("text", "") for c in contexts]
            elif "context" in item:
                ctx = item["context"]
                contexts = [ctx] if isinstance(ctx, str) else ctx
            
            sample = TestSample(
                question=question,
                ground_truth=ground_truth,
                contexts=contexts,
                metadata={k: v for k, v in item.items() if k not in ["query", "question", "answer", "ground_truth", "docs", "context"]}
            )
            samples.append(sample)
            
        return samples
