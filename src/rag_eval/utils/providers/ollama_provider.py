"""Ollama LLM provider — local models via Ollama REST API."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from rag_eval.core.config import LLMConfig
from rag_eval.utils.provider import BaseLLMProvider, LLMResponse


class OllamaProvider(BaseLLMProvider):
    """LLM provider backed by a local Ollama server.

    Ollama runs open-source models (Llama, Mistral, Phi, Qwen, etc.)
    locally on your machine. **No API key required — completely free.**

    Prerequisites:
        1. Install Ollama: https://ollama.com
        2. Pull a model: ``ollama pull llama3.2``
        3. Ollama runs automatically on ``http://localhost:11434``
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._base_url = config.get_api_base() or "http://localhost:11434"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        system: str = "",
        **kwargs: object,
    ) -> LLMResponse:
        """Send a chat request to the Ollama API."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }

        start = time.perf_counter()
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = response.json()
        text = data.get("message", {}).get("content", "")

        # Ollama provides token counts in eval_count and prompt_eval_count
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_estimate=0.0,  # Local models are free
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
        reraise=True,
    )
    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        **kwargs: "Any",
    ) -> dict[str, Any]:
        """Send a chat request with JSON format output.

        Ollama supports a ``format`` parameter to enforce JSON output.
        """
        messages: list[dict[str, str]] = []
        json_system = system or "You are a helpful assistant."
        json_system += (
            "\nYou MUST respond with valid JSON only. "
            "Do not include any text outside the JSON object."
        )
        messages.append({"role": "system", "content": json_system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }

        start = time.perf_counter()
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = response.json()
        text = data.get("message", {}).get("content", "{}")
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        self._last_response = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_estimate=0.0,
        )

        from typing import cast
        return cast(dict[str, Any], json.loads(text))

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
