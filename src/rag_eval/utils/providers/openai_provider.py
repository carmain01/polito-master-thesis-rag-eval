"""OpenAI LLM provider — GPT-4o, GPT-4o-mini, o3, etc."""

from __future__ import annotations

import json
import time
from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam

from rag_eval.core.config import LLMConfig
from rag_eval.utils.provider import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI API.

    Supports GPT-4o, GPT-4o-mini, o3, and other OpenAI chat models.
    Uses the official ``openai`` SDK with native async support.
    """

    _client: openai.AsyncOpenAI

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

        api_key = config.get_api_key()
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in .env or pass api_key in config."
            )

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.timeout,
            "max_retries": 0,  # We handle retries ourselves via tenacity
        }
        api_base = config.get_api_base()
        if api_base:
            client_kwargs["base_url"] = api_base

        self._client = openai.AsyncOpenAI(**client_kwargs)

        # Apply configurable retry from config.max_retries
        _retry = self._make_retry()
        self.complete = _retry(self.complete)  # type: ignore[method-assign]
        self.complete_json = _retry(self.complete_json)  # type: ignore[method-assign]


    @property
    def provider_name(self) -> str:
        return "openai"

    async def complete(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to OpenAI."""
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=float(kwargs.get("temperature", self.config.temperature)),
            max_tokens=int(kwargs.get("max_tokens", self.config.max_tokens)),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        text = response.choices[0].message.content or ""

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
        """Send a chat completion request with JSON response format."""
        messages: list[ChatCompletionMessageParam] = []
        json_system = system or "You are a helpful assistant."
        json_system += "\nAlways respond with valid JSON."
        messages.append({"role": "system", "content": json_system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=float(kwargs.get("temperature", self.config.temperature)),
            max_tokens=int(kwargs.get("max_tokens", self.config.max_tokens)),
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.choices[0].message.content or "{}"
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Track the raw response for cost/usage even though we return parsed JSON
        self._last_response = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_estimate=self.estimate_cost(input_tokens, output_tokens),
        )

        from typing import cast
        return cast(dict[str, Any], json.loads(text))
