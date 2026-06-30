"""Anthropic LLM provider — Claude Sonnet, Opus, Haiku."""

from __future__ import annotations

import json
import time
from typing import Any, cast

from anthropic.types import Message

from rag_eval.core.config import LLMConfig
from rag_eval.utils.provider import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic API.

    Supports Claude Sonnet 4, Opus 4, Haiku 3.5, and other Claude models.
    Uses the official ``anthropic`` SDK with native async support.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        import anthropic

        api_key = config.get_api_key()
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY in .env or pass api_key in config."
            )

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=config.timeout,
            max_retries=0,  # We handle retries via tenacity
        )

        # Apply configurable retry from config.max_retries
        _retry = self._make_retry()
        self.complete = _retry(self.complete)  # type: ignore[method-assign]
        self.complete_json = _retry(self.complete_json)  # type: ignore[method-assign]

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def complete(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a message to the Anthropic API."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": cast(int, kwargs.get("max_tokens", self.config.max_tokens)),
            "temperature": cast(float, kwargs.get("temperature", self.config.temperature)),
        }
        if system:
            create_kwargs["system"] = system

        start = time.perf_counter()
        response = cast(Message, await self._client.messages.create(**create_kwargs))
        latency_ms = (time.perf_counter() - start) * 1000

        content = response.content
        text = getattr(content[0], "text", "") if content and isinstance(content, list) else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_estimate=self.estimate_cost(input_tokens, output_tokens),
        )

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a message and parse the response as JSON.

        Anthropic doesn't have a native JSON mode, so we instruct the model
        to return JSON and parse the response text.
        """
        from typing import cast
        
        json_system = system or "You are a helpful assistant."
        json_system += (
            "\nYou MUST respond with valid JSON only. "
            "Do not include any text outside the JSON object."
        )

        # Prefill the assistant response with "{" to encourage JSON output
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ]

        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": cast(int, kwargs.get("max_tokens", self.config.max_tokens)),
            "temperature": cast(float, kwargs.get("temperature", self.config.temperature)),
            "system": json_system,
        }

        start = time.perf_counter()
        response = cast(Message, await self._client.messages.create(**create_kwargs))
        latency_ms = (time.perf_counter() - start) * 1000

        content = response.content
        raw_text = getattr(content[0], "text", "") if content and isinstance(content, list) else ""
        # Prepend the "{" we used as prefill
        text = "{" + raw_text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        self._last_response = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_estimate=self.estimate_cost(input_tokens, output_tokens),
        )

        return cast(dict[str, Any], json.loads(text))
