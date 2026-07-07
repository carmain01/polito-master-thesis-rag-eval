"""Benchmark adapters for popular QA and RAG datasets."""

from rag_eval.datasets.benchmarks.base import BaseBenchmarkAdapter
from rag_eval.datasets.benchmarks.hotpotqa import HotpotQAAdapter
from rag_eval.datasets.benchmarks.nq import NaturalQuestionsAdapter
from rag_eval.datasets.benchmarks.recall import RecallAdapter
from rag_eval.datasets.benchmarks.rgb import RGBAdapter
from rag_eval.datasets.benchmarks.squad import Squad2Adapter
from rag_eval.datasets.benchmarks.triviaqa import TriviaQAAdapter

__all__ = [
    "BaseBenchmarkAdapter",
    "HotpotQAAdapter",
    "NaturalQuestionsAdapter",
    "TriviaQAAdapter",
    "Squad2Adapter",
    "RGBAdapter",
    "RecallAdapter",
    "get_benchmark_adapter",
]

_ADAPTER_REGISTRY = {
    "hotpotqa": HotpotQAAdapter,
    "nq": NaturalQuestionsAdapter,
    "triviaqa": TriviaQAAdapter,
    "squad2": Squad2Adapter,
    "rgb": RGBAdapter,
    "recall": RecallAdapter,
}

def get_benchmark_adapter(name: str, **kwargs) -> BaseBenchmarkAdapter:
    """Factory function to get a benchmark adapter by name.
    
    Args:
        name: Name of the benchmark (e.g., 'hotpotqa', 'nq', 'squad2').
        **kwargs: Optional arguments passed to the adapter constructor.
        
    Returns:
        An instance of the requested BaseBenchmarkAdapter.
    """
    name = name.lower().strip()
    if name not in _ADAPTER_REGISTRY:
        raise ValueError(f"Unknown benchmark adapter: {name}. Available: {list(_ADAPTER_REGISTRY.keys())}")
    
    return _ADAPTER_REGISTRY[name](**kwargs)
