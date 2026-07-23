"""vLLM LLM provider — local models via vLLM's OpenAI-compatible API."""

from __future__ import annotations

import json
import time
from typing import Any, cast

import httpx

from rag_eval.core.config import LLMConfig
from rag_eval.utils.provider import BaseLLMProvider, LLMResponse


class VLLMProvider(BaseLLMProvider):
    """LLM provider backed by a vLLM server.

    vLLM provides an OpenAI-compatible API for running open-source models
    locally with high throughput. **No API key required.**

    Prerequisites:
        1. Install vLLM: ``pip install vllm``
        2. Start the server: ``vllm serve <model-name>``
        3. By default, the server runs on ``http://localhost:8000``
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._base_url = config.get_api_base() or "http://localhost:8000"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )
        _retry = self._make_retry()
        self.complete = _retry(self.complete)  # type: ignore[method-assign]
        self.complete_json = _retry(self.complete_json)  # type: ignore[method-assign]

    @property
    def provider_name(self) -> str:
        return "vllm"


    async def complete(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to the vLLM server.

        vLLM exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        start = time.perf_counter()
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"vLLM returned empty choices: {data}")
        text = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_estimate=0.0,  # Local models are free
        )


    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completion request with JSON output.

        Uses the ``response_format`` parameter (OpenAI-compatible).
        """
        messages: list[dict[str, str]] = []
        json_system = system or "You are a helpful assistant."
        json_system += "\nAlways respond with valid JSON only."
        messages.append({"role": "system", "content": json_system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "response_format": {"type": "json_object"},
        }

        start = time.perf_counter()
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"vLLM returned empty choices: {data}")
        text = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        self._last_response = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_estimate=0.0,
        )

        return cast(dict[str, Any], json.loads(text))

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
