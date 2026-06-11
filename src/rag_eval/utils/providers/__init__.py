"""Provider factory — instantiate the correct LLM provider from config."""

from __future__ import annotations

from rag_eval.core.config import LLMConfig
from rag_eval.utils.provider import BaseLLMProvider


def create_provider(config: LLMConfig) -> BaseLLMProvider:
    """Create an LLM provider instance from configuration.

    Args:
        config: LLM configuration specifying the provider and model.

    Returns:
        An instance of the appropriate provider.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    provider_name = config.provider.lower()

    if provider_name == "openai":
        from rag_eval.utils.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(config)

    if provider_name == "anthropic":
        from rag_eval.utils.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(config)

    if provider_name == "google":
        from rag_eval.utils.providers.google_provider import GoogleProvider

        return GoogleProvider(config)

    if provider_name == "ollama":
        from rag_eval.utils.providers.ollama_provider import OllamaProvider

        return OllamaProvider(config)

    if provider_name == "vllm":
        from rag_eval.utils.providers.vllm_provider import VLLMProvider

        return VLLMProvider(config)

    supported = ["openai", "anthropic", "google", "ollama", "vllm"]
    raise ValueError(
        f"Unknown provider '{provider_name}'. Supported: {', '.join(supported)}"
    )


__all__ = ["create_provider"]
