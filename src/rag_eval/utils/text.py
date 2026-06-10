"""Text processing utilities — chunking, cleaning, tokenization helpers."""

from __future__ import annotations


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string."""
    import tiktoken

    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences (simple heuristic)."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_into_claims(text: str) -> list[str]:
    """Split a text into atomic claims for faithfulness evaluation.

    TODO: Replace with LLM-based claim decomposition.
    """
    return split_into_sentences(text)
