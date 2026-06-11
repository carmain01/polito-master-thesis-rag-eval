"""Tests for the LLMClient facade."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_eval.core.config import CacheConfig, LLMConfig
from rag_eval.utils.llm import LLMClient
from rag_eval.utils.provider import LLMResponse


@pytest.fixture
def mock_response():
    """Create a standard mock LLM response."""
    return LLMResponse(
        text="Test response",
        input_tokens=50,
        output_tokens=100,
        model="llama3.2",
        provider="ollama",
        latency_ms=200.0,
        cost_estimate=0.0,
    )


@pytest.fixture
def client_no_cache():
    """Create a client with caching disabled."""
    config = LLMConfig(provider="ollama", model="llama3.2")
    return LLMClient(config, enable_cache=False)


@pytest.fixture
def client_with_cache(tmp_path):
    """Create a client with caching enabled in a temp directory."""
    config = LLMConfig(provider="ollama", model="llama3.2")
    cache_config = CacheConfig(directory=str(tmp_path / "cache"))
    return LLMClient(config, cache_config=cache_config, enable_cache=True)


class TestLLMClientBasic:
    """Test basic LLMClient operations."""

    @pytest.mark.asyncio
    async def test_complete(self, client_no_cache, mock_response):
        client_no_cache._provider.complete = AsyncMock(return_value=mock_response)

        result = await client_no_cache.complete("Hello")
        assert result.text == "Test response"
        assert result.input_tokens == 50
        assert result.output_tokens == 100

    @pytest.mark.asyncio
    async def test_usage_tracking(self, client_no_cache, mock_response):
        client_no_cache._provider.complete = AsyncMock(return_value=mock_response)

        await client_no_cache.complete("Test 1")
        await client_no_cache.complete("Test 2")

        assert client_no_cache.usage.call_count == 2
        assert client_no_cache.usage.total_input_tokens == 100
        assert client_no_cache.usage.total_output_tokens == 200

    @pytest.mark.asyncio
    async def test_reset_usage(self, client_no_cache, mock_response):
        client_no_cache._provider.complete = AsyncMock(return_value=mock_response)

        await client_no_cache.complete("Test")
        assert client_no_cache.usage.call_count == 1

        client_no_cache.reset_usage()
        assert client_no_cache.usage.call_count == 0
        assert client_no_cache.usage.total_tokens == 0

    def test_repr(self, client_no_cache):
        r = repr(client_no_cache)
        assert "ollama" in r
        assert "llama3.2" in r


class TestLLMClientCaching:
    """Test LLMClient caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, client_with_cache, mock_response):
        client_with_cache._provider.complete = AsyncMock(return_value=mock_response)

        # First call — cache miss
        result1 = await client_with_cache.complete("Same prompt")
        assert result1.text == "Test response"

        # Second call — should be cache hit (provider NOT called again)
        result2 = await client_with_cache.complete("Same prompt")
        assert result2.text == "Test response"

        # Provider should only be called once
        assert client_with_cache._provider.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_disabled_per_call(self, client_with_cache, mock_response):
        client_with_cache._provider.complete = AsyncMock(return_value=mock_response)

        await client_with_cache.complete("Prompt", use_cache=False)
        await client_with_cache.complete("Prompt", use_cache=False)

        # Provider should be called both times when cache is disabled per-call
        assert client_with_cache._provider.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_different_prompts_no_cache_hit(self, client_with_cache, mock_response):
        client_with_cache._provider.complete = AsyncMock(return_value=mock_response)

        await client_with_cache.complete("Prompt A")
        await client_with_cache.complete("Prompt B")

        # Different prompts = different cache keys = 2 provider calls
        assert client_with_cache._provider.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_stats(self, client_with_cache, mock_response):
        client_with_cache._provider.complete = AsyncMock(return_value=mock_response)

        await client_with_cache.complete("Test prompt")
        await client_with_cache.complete("Test prompt")

        stats = client_with_cache.cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1


class TestLLMClientJSON:
    """Test LLMClient JSON completion."""

    @pytest.mark.asyncio
    async def test_complete_json(self, client_no_cache):
        json_response = LLMResponse(
            text='{"score": 0.9}',
            input_tokens=30,
            output_tokens=10,
            model="llama3.2",
            provider="ollama",
        )
        client_no_cache._provider.complete_json = AsyncMock(
            return_value={"score": 0.9}
        )
        client_no_cache._provider._last_response = json_response

        result = await client_no_cache.complete_json("Rate this.")
        assert result == {"score": 0.9}
