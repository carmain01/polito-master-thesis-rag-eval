"""Utilities — LLM clients, embeddings, text processing, caching, and tracking."""

from rag_eval.utils.cache import ResponseCache
from rag_eval.utils.embeddings import EmbeddingClient, cosine_similarity
from rag_eval.utils.llm import LLMClient
from rag_eval.utils.provider import BaseLLMProvider, LLMResponse
from rag_eval.utils.tracking import UsageTracker

__all__ = [
    "BaseLLMProvider",
    "EmbeddingClient",
    "LLMClient",
    "LLMResponse",
    "ResponseCache",
    "UsageTracker",
    "cosine_similarity",
]
