"""Base LLM provider interface and response model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from rag_eval.core.config import LLMConfig


class LLMResponse(BaseModel):
    """Structured response from any LLM provider."""

    text: str = Field(..., description="The generated text content.")
    input_tokens: int = Field(default=0, ge=0, description="Number of input tokens.")
    output_tokens: int = Field(default=0, ge=0, description="Number of output tokens.")
    model: str = Field(default="", description="Model identifier used.")
    provider: str = Field(default="", description="Provider name.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Request latency in milliseconds.")
    cost_estimate: float = Field(default=0.0, ge=0.0, description="Estimated cost in USD.")


# Pricing per 1M tokens: (input_price, output_price) in USD.
# Updated as of mid-2025 — adjust as prices change.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "gpt-5-nano": (0.05, 0.20),
    # Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    # Google
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    # Local (free)
    "llama3.2": (0.0, 0.0),
    "llama3.1": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "phi3": (0.0, 0.0),
    "qwen2": (0.0, 0.0),
}


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Each provider implements this interface to enable uniform access
    across OpenAI, Anthropic, Google, Ollama, and vLLM.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # Stores the last LLMResponse from complete_json(), used by LLMClient for usage tracking
        self._last_response: LLMResponse | None = None

    def _make_retry(self) -> "Any":
        """Create a tenacity retry decorator based on config.max_retries."""
        from tenacity import retry, stop_after_attempt, wait_exponential_jitter

        return retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
            reraise=True,
        )

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider's unique identifier (e.g., 'openai')."""
        ...

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a prompt to the LLM and return a structured response.

        Args:
            prompt: The user prompt / message.
            system: Optional system prompt.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.).
        """
        ...

    @abstractmethod
    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a prompt and parse the response as JSON.

        Uses JSON mode or function calling where supported.
        Falls back to parsing the text response as JSON.

        Args:
            prompt: The user prompt, should request JSON output.
            system: Optional system prompt.
            **kwargs: Provider-specific overrides.

        Returns:
            Parsed JSON as a dictionary.
        """
        ...

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost of a request in USD.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pricing = MODEL_PRICING.get(self.config.model, (0.0, 0.0))
        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]
        return input_cost + output_cost

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model!r})"
