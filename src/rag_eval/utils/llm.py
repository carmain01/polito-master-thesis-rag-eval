"""LLM client — unified high-level interface over all providers."""

from __future__ import annotations

import json
import logging

from rag_eval.core.config import CacheConfig, LLMConfig
from rag_eval.utils.cache import ResponseCache
from rag_eval.utils.provider import LLMResponse
from rag_eval.utils.providers import create_provider
from rag_eval.utils.tracking import UsageTracker

logger = logging.getLogger(__name__)


class LLMClient:
    """High-level LLM client with caching, usage tracking, and provider abstraction.

    This is the main entry point for making LLM calls in the framework.
    It delegates to the appropriate provider based on config, caches responses
    to disk, and tracks token usage and costs.
    
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache_config: CacheConfig | None = None,
        enable_cache: bool = True,
    ) -> None:
        self.config = config or LLMConfig()
        self._provider = create_provider(self.config)
        self._tracker = UsageTracker()

        # Initialize cache
        self._cache: ResponseCache | None = None
        cache_conf = cache_config or CacheConfig()
        if enable_cache and cache_conf.enabled:
            self._cache = ResponseCache(cache_conf)

        logger.info(
            "LLMClient initialized: provider=%s, model=%s, cache=%s",
            self.config.provider,
            self.config.model,
            "enabled" if self._cache else "disabled",
        )

    async def complete(
        self,
        prompt: str,
        system: str = "",
        use_cache: bool = True,
        **kwargs: object,
    ) -> LLMResponse:
        """Send a prompt to the LLM and return a structured response.

        Args:
            prompt: The user prompt / message.
            system: Optional system prompt for context/instructions.
            use_cache: Whether to use caching for this call.
                Set to False for non-deterministic or one-off calls.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.).

        Returns:
            Structured LLMResponse with text, token counts, cost, and latency.
        """
        # Check cache first
        if use_cache and self._cache:
            cached = self._cache.get(prompt, system, self.config.model)
            if cached is not None:
                logger.debug("Cache hit for prompt (len=%d)", len(prompt))
                return cached

        # Call the provider
        response = await self._provider.complete(prompt, system, **kwargs)

        # Track usage
        self._tracker.record(response)

        # Store in cache
        if use_cache and self._cache:
            self._cache.put(prompt, system, self.config.model, response)

        logger.debug(
            "LLM call: model=%s, tokens=%d+%d, cost=$%.4f, latency=%.0fms",
            response.model,
            response.input_tokens,
            response.output_tokens,
            response.cost_estimate,
            response.latency_ms,
        )

        return response

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        use_cache: bool = True,
        **kwargs: object,
    ) -> dict:
        """Send a prompt and get a structured JSON response.

        Args:
            prompt: The user prompt (should request JSON output).
            system: Optional system prompt.
            use_cache: Whether to use caching for this call.
            **kwargs: Provider-specific overrides.

        Returns:
            Parsed JSON response as a dictionary.
        """
        # For JSON, we use the underlying complete to get the full response for caching
        if use_cache and self._cache:
            cached = self._cache.get(prompt, system, self.config.model)
            if cached is not None:
                logger.debug("Cache hit for JSON prompt (len=%d)", len(prompt))
                return json.loads(cached.text)

        # Call provider's JSON mode
        result = await self._provider.complete_json(prompt, system, **kwargs)

        # Track and cache the raw response if available
        if hasattr(self._provider, "_last_response"):
            raw_response = self._provider._last_response
            if raw_response is not None:
                self._tracker.record(raw_response)
                if use_cache and self._cache:
                    self._cache.put(prompt, system, self.config.model, raw_response)

        return result

    @property
    def usage(self) -> UsageTracker:
        """Access the usage tracker for cost/token monitoring."""
        return self._tracker

    @property
    def cache(self) -> ResponseCache | None:
        """Access the response cache (if enabled)."""
        return self._cache

    def reset_usage(self) -> None:
        """Reset usage tracking counters to zero."""
        self._tracker.reset()

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self.config.provider!r}, "
            f"model={self.config.model!r}, "
            f"usage={self._tracker})"
        )
