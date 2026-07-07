"""Base class for benchmark dataset adapters."""

from abc import ABC, abstractmethod
from typing import Any

from datasets import load_dataset
from rag_eval.core.types import TestSample


class BaseBenchmarkAdapter(ABC):
    """Abstract base class for benchmark adapters.
    
    Each adapter is responsible for loading a specific benchmark dataset
    and converting its samples into the framework's internal TestSample format.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def load(self, split: str = "validation", max_samples: int | None = None) -> list[TestSample]:
        """Load and adapt the benchmark dataset into TestSample objects.

        Args:
            split: The dataset split to load (e.g., 'train', 'validation', 'test').
            max_samples: If provided, limits the number of samples returned.

        Returns:
            A list of TestSample objects.
        """
        pass

    def _load_hf_dataset(self, path: str, name: str | None = None, split: str = "validation") -> Any:
        """Helper to load a dataset from HuggingFace.
        
        Args:
            path: The dataset repository or local path.
            name: The dataset configuration name (optional).
            split: The split to load.
            
        Returns:
            A HuggingFace Dataset object.
        """
        if name:
            return load_dataset(path, name, split=split)
        return load_dataset(path, split=split)
