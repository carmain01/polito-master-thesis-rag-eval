"""Azure OpenAI LLM provider — GPT-4o, GPT-5-nano, o1, o3, etc. on Azure."""

from __future__ import annotations

import json
import os
import time
from typing import Any, cast

import openai
from openai.types.chat import ChatCompletionMessageParam

from rag_eval.core.config import LLMConfig
from rag_eval.utils.provider import BaseLLMProvider, LLMResponse


class AzureProvider(BaseLLMProvider):
    """LLM provider backed by Azure OpenAI Service or Azure AI Foundry.

    Uses ``openai.AsyncAzureOpenAI`` with native async support, supporting
    both standard chat models and reasoning models (gpt-5, o1, o3).
    """

    _client: openai.AsyncAzureOpenAI

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

        api_key = config.get_api_key()
        if not api_key:
            raise ValueError(
                "Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY in .env or pass api_key in config."
            )

        azure_endpoint = config.get_api_base()
        if not azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint is required. Set AZURE_OPENAI_ENDPOINT in .env or pass api_base in config."
            )

        api_version = getattr(config, "api_version", "2024-06-01") or "2024-06-01"

        self._client = openai.AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=config.timeout,
            max_retries=0,  # Retries handled via tenacity in BaseLLMProvider
        )

        _retry = self._make_retry()
        self.complete = _retry(self.complete)  # type: ignore[method-assign]
        self.complete_json = _retry(self.complete_json)  # type: ignore[method-assign]

    @property
    def provider_name(self) -> str:
        return "azure"

    def _is_reasoning_or_newer_model(self) -> bool:
        """Check if model uses max_completion_tokens and does not support custom temperature."""
        m = (self.config.model or "").lower()
        return any(p in m for p in ["gpt-5", "o1", "o3", "o4", "reasoning"])

    async def complete(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Azure OpenAI."""
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()

        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }

        max_tokens = int(kwargs.get("max_tokens", self.config.max_tokens))
        if self._is_reasoning_or_newer_model():
            create_kwargs["max_completion_tokens"] = max(max_tokens, 32768)
            effort = kwargs.get("reasoning_effort") or getattr(self.config, "reasoning_effort", "high") or "high"
            if effort:
                create_kwargs["reasoning_effort"] = effort
        else:
            create_kwargs["max_tokens"] = max_tokens
            create_kwargs["temperature"] = float(kwargs.get("temperature", self.config.temperature))

        response = await self._client.chat.completions.create(**create_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        if choice.finish_reason == "length":
            import logging
            logging.getLogger(__name__).warning("Azure OpenAI completion was truncated by max_completion_tokens limit.")

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        text = choice.message.content or ""

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

        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        max_tokens = int(kwargs.get("max_tokens", self.config.max_tokens))
        if self._is_reasoning_or_newer_model():
            create_kwargs["max_completion_tokens"] = max(max_tokens, 32768)
            effort = kwargs.get("reasoning_effort") or getattr(self.config, "reasoning_effort", "high") or "high"
            if effort:
                create_kwargs["reasoning_effort"] = effort
        else:
            create_kwargs["max_tokens"] = max_tokens
            create_kwargs["temperature"] = float(kwargs.get("temperature", self.config.temperature))

        response = await self._client.chat.completions.create(**create_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"Azure OpenAI response was truncated due to token limit (model={self.config.model}, finish_reason='length')"
            )

        raw_content = choice.message.content
        if not raw_content or not raw_content.strip():
            raise RuntimeError(
                f"Azure OpenAI returned empty content (finish_reason={choice.finish_reason!r}, model={self.config.model})"
            )

        text = raw_content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        self._last_response = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_estimate=self.estimate_cost(input_tokens, output_tokens),
        )

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Azure OpenAI returned invalid JSON ({exc}): {text[:200]}") from exc

        if not isinstance(parsed, dict) or not parsed:
            raise RuntimeError(f"Azure OpenAI returned empty or non-dict JSON: {parsed}")

        return cast(dict[str, Any], parsed)
