"""LLM client utilities — unified interface for LLM calls."""

from __future__ import annotations

from rag_eval.core.config import LLMConfig


class LLMClient:
    """Thin wrapper around LLM provider APIs for metric evaluations."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the configured LLM and return the response text."""
        # TODO: Implement multi-provider LLM client (OpenAI, Anthropic, etc.)
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        # TODO: Implement embedding generation
        raise NotImplementedError
