"""Synthetic data generation — create test datasets using LLMs."""

from __future__ import annotations

from rag_eval.core.types import TestSample


class SyntheticGenerator:
    """Generate synthetic evaluation datasets from source documents."""

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents

    async def generate(self, num_samples: int = 10) -> list[TestSample]:
        """Generate synthetic question-answer pairs from documents.

        Uses an LLM to create realistic QA pairs grounded in the source documents.
        """
        # TODO: Implement LLM-based synthetic data generation
        raise NotImplementedError
