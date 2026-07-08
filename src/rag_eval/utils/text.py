"""Text processing utilities — tokenization, sentence splitting, claim decomposition."""

from __future__ import annotations

import re
from typing import Any


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string.

    Uses tiktoken for OpenAI models (exact). For other providers, falls back
    to a word-based heuristic (~1.3 tokens per word).

    Args:
        text: The text to tokenize.
        model: Model identifier for provider-specific tokenization.

    Returns:
        Estimated number of tokens.
    """
    # OpenAI models: use tiktoken for exact counts
    openai_models = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o3",
        "o3-mini",
        "o4-mini",
    }
    if model in openai_models or model.startswith("gpt-"):
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except (KeyError, ImportError):
            pass

    # Fallback: word-based heuristic (reasonable for most models)
    words = text.split()
    return int(len(words) * 1.3)


def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences using NLTK.

    Uses NLTK's Punkt sentence tokenizer for accurate splitting
    that handles abbreviations (Dr., U.S., etc.) correctly.

    Args:
        text: The text to split.

    Returns:
        List of sentences.
    """
    if not text or not text.strip():
        return []

    try:
        import nltk

        # Ensure the punkt tokenizer data is available
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        sentences = nltk.sent_tokenize(text.strip())
        return [s.strip() for s in sentences if s.strip()]
    except ImportError:
        # Fallback to regex if NLTK is not available
        return _regex_split_sentences(text)


def _regex_split_sentences(text: str) -> list[str]:
    """Fallback regex-based sentence splitter."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_into_claims(text: str) -> list[str]:
    """Split a text into atomic claims for faithfulness evaluation.

    Currently uses sentence splitting as a proxy for claim extraction.
    In Phase 2, this will be enhanced with LLM-based claim decomposition
    that can break complex sentences into individual factual statements.

    Args:
        text: The text to decompose into claims.

    Returns:
        List of atomic claims (one factual statement each).
    """
    return split_into_sentences(text)


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace, remove articles.

    Args:
        text: The text to normalize.

    Returns:
        Normalized text string.
    """
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_json_from_text(text: str) -> dict[str, Any] | list[Any]:
    """Extract a JSON object or array from text that may contain surrounding content.

    Useful for parsing LLM responses that include JSON within explanation text.

    Args:
        text: Text potentially containing a JSON object or array.

    Returns:
        Parsed JSON as a dictionary or list.

    Raises:
        ValueError: If no valid JSON is found.
    """
    import json

    # Try parsing the entire text first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object or array within the text
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = text.find(open_char)
        if start == -1:
            continue

        # Find the matching closing bracket
        depth = 0
        for i, char in enumerate(text[start:], start=start):
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"No valid JSON found in text: {text[:100]}...")
