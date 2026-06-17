"""Tests for LLM provider implementations (mocked — no real API calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_eval.core.config import LLMConfig
from rag_eval.utils.provider import MODEL_PRICING, LLMResponse
from rag_eval.utils.providers import create_provider

# ---------------------------------------------------------------------------
# Provider factory tests
# ---------------------------------------------------------------------------

class TestCreateProvider:
    """Test the provider factory function."""

    def test_unknown_provider_raises(self):
        config = LLMConfig(provider="unknown_provider")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    def test_create_openai(self):
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        provider = create_provider(config)
        assert provider.provider_name == "openai"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_create_anthropic(self):
        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        provider = create_provider(config)
        assert provider.provider_name == "anthropic"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "aig-test"})
    def test_create_google(self):
        config = LLMConfig(provider="google", model="gemini-2.5-flash")
        provider = create_provider(config)
        assert provider.provider_name == "google"

    def test_create_ollama(self):
        config = LLMConfig(provider="ollama", model="llama3.2")
        provider = create_provider(config)
        assert provider.provider_name == "ollama"

    def test_create_vllm(self):
        config = LLMConfig(provider="vllm", model="mistral")
        provider = create_provider(config)
        assert provider.provider_name == "vllm"


# ---------------------------------------------------------------------------
# Cost estimation tests
# ---------------------------------------------------------------------------

class TestCostEstimation:
    """Test the base provider cost estimation."""

    def test_openai_cost(self):
        config = LLMConfig(provider="ollama", model="gpt-4o")
        provider = create_provider(config)
        # Override the model for cost calculation
        provider.config = LLMConfig(model="gpt-4o")
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 1e-8

    def test_local_model_free(self):
        config = LLMConfig(provider="ollama", model="llama3.2")
        provider = create_provider(config)
        cost = provider.estimate_cost(input_tokens=10000, output_tokens=5000)
        assert cost == 0.0

    def test_unknown_model_free(self):
        config = LLMConfig(provider="ollama", model="some-custom-model")
        provider = create_provider(config)
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# OpenAI provider tests (mocked)
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    """Test OpenAI provider with mocked API calls."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI response."""
        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 100

        choice = MagicMock()
        choice.message.content = "This is a test response."

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        return response

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @pytest.mark.asyncio
    async def test_complete(self, mock_openai_response):
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        provider = create_provider(config)

        provider._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        result = await provider.complete("Hello, world!")
        assert isinstance(result, LLMResponse)
        assert result.text == "This is a test response."
        assert result.input_tokens == 50
        assert result.output_tokens == 100
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
        assert result.latency_ms > 0

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @pytest.mark.asyncio
    async def test_complete_json(self, mock_openai_response):
        mock_openai_response.choices[0].message.content = '{"score": 0.85}'
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        provider = create_provider(config)

        provider._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        result = await provider.complete_json("Rate this answer.")
        assert isinstance(result, dict)
        assert result["score"] == 0.85

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="")
        with pytest.raises(ValueError, match="API key is required"):
            create_provider(config)


# ---------------------------------------------------------------------------
# Ollama provider tests (mocked)
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    """Test Ollama provider with mocked HTTP calls."""

    @pytest.mark.asyncio
    async def test_complete(self):
        config = LLMConfig(provider="ollama", model="llama3.2")
        provider = create_provider(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hello from Ollama!"},
            "prompt_eval_count": 30,
            "eval_count": 50,
        }
        mock_response.raise_for_status = MagicMock()

        provider._client.post = AsyncMock(return_value=mock_response)

        result = await provider.complete("Say hello")
        assert isinstance(result, LLMResponse)
        assert result.text == "Hello from Ollama!"
        assert result.input_tokens == 30
        assert result.output_tokens == 50
        assert result.provider == "ollama"
        assert result.cost_estimate == 0.0

    @pytest.mark.asyncio
    async def test_complete_json(self):
        config = LLMConfig(provider="ollama", model="llama3.2")
        provider = create_provider(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": '{"result": "ok"}'},
            "prompt_eval_count": 20,
            "eval_count": 10,
        }
        mock_response.raise_for_status = MagicMock()

        provider._client.post = AsyncMock(return_value=mock_response)

        result = await provider.complete_json("Return JSON")
        assert result == {"result": "ok"}


# ---------------------------------------------------------------------------
# vLLM provider tests (mocked)
# ---------------------------------------------------------------------------

class TestVLLMProvider:
    """Test vLLM provider with mocked HTTP calls."""

    @pytest.mark.asyncio
    async def test_complete(self):
        config = LLMConfig(provider="vllm", model="mistral")
        provider = create_provider(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from vLLM!"}}],
            "usage": {"prompt_tokens": 25, "completion_tokens": 40},
        }
        mock_response.raise_for_status = MagicMock()

        provider._client.post = AsyncMock(return_value=mock_response)

        result = await provider.complete("Say hello")
        assert result.text == "Hello from vLLM!"
        assert result.input_tokens == 25
        assert result.output_tokens == 40
        assert result.provider == "vllm"
        assert result.cost_estimate == 0.0


# ---------------------------------------------------------------------------
# LLMResponse model tests
# ---------------------------------------------------------------------------

class TestLLMResponse:
    """Test the LLMResponse data model."""

    def test_create_response(self):
        response = LLMResponse(
            text="Test",
            input_tokens=10,
            output_tokens=20,
            model="gpt-4o",
            provider="openai",
            latency_ms=100.0,
            cost_estimate=0.001,
        )
        assert response.text == "Test"
        assert response.input_tokens == 10
        assert response.output_tokens == 20

    def test_default_values(self):
        response = LLMResponse(text="Hello")
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.cost_estimate == 0.0

    def test_serialization(self):
        response = LLMResponse(text="Test", model="llama3.2", provider="ollama")
        data = response.model_dump()
        restored = LLMResponse(**data)
        assert restored.text == "Test"
        assert restored.model == "llama3.2"


class TestModelPricing:
    """Test that model pricing table is correctly structured."""

    def test_pricing_has_known_models(self):
        assert "gpt-4o" in MODEL_PRICING
        assert "llama3.2" in MODEL_PRICING
        assert "gemini-2.5-pro" in MODEL_PRICING

    def test_local_models_are_free(self):
        for model in ["llama3.2", "llama3.1", "mistral", "phi3", "qwen2"]:
            assert MODEL_PRICING[model] == (0.0, 0.0), f"{model} should be free"
