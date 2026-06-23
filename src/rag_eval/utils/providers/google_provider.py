"""Google LLM provider — Gemini 2.5 Pro, Gemini 2.5 Flash, etc."""

from __future__ import annotations

import json
import time

from rag_eval.core.config import LLMConfig
from rag_eval.utils.provider import BaseLLMProvider, LLMResponse


class GoogleProvider(BaseLLMProvider):
    """LLM provider backed by the Google Gemini API.

    Supports Gemini 2.5 Pro, Gemini 2.5 Flash, and other Gemini models.
    Uses the official ``google-genai`` SDK.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        from google import genai

        api_key = config.get_api_key()
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Set GOOGLE_API_KEY in .env or pass api_key in config."
            )

        self._client = genai.Client(api_key=api_key)

        # Apply configurable retry from config.max_retries
        _retry = self._make_retry()
        self.complete = _retry(self.complete)
        self.complete_json = _retry(self.complete_json)

    @property
    def provider_name(self) -> str:
        return "google"

    async def complete(
        self,
        prompt: str,
        system: str = "",
        **kwargs: object,
    ) -> LLMResponse:
        """Send a request to the Gemini API."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        if system:
            config.system_instruction = system

        start = time.perf_counter()
        response = await self._client.aio.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=config,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.text or ""
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

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
        **kwargs: object,
    ) -> dict:
        """Send a request with JSON response format."""
        from google.genai import types

        json_system = system or "You are a helpful assistant."
        json_system += "\nAlways respond with valid JSON only."

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            response_mime_type="application/json",
            system_instruction=json_system,
        )

        start = time.perf_counter()
        response = await self._client.aio.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=config,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.text or "{}"
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        self._last_response = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            cost_estimate=self.estimate_cost(input_tokens, output_tokens),
        )

        return json.loads(text)
