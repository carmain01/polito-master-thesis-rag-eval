"""Tests for the ResponseCache."""

from __future__ import annotations

import time

import pytest

from rag_eval.core.config import CacheConfig
from rag_eval.utils.cache import ResponseCache
from rag_eval.utils.provider import LLMResponse


@pytest.fixture
def cache(tmp_path):
    """Create a cache in a temporary directory."""
    config = CacheConfig(directory=str(tmp_path / "test_cache"))
    return ResponseCache(config)


@pytest.fixture
def sample_response():
    """A sample LLM response for testing."""
    return LLMResponse(
        text="The capital of France is Paris.",
        input_tokens=20,
        output_tokens=10,
        model="llama3.2",
        provider="ollama",
        latency_ms=150.0,
        cost_estimate=0.0,
    )


class TestCacheBasic:
    """Test basic cache operations."""

    def test_get_miss(self, cache):
        result = cache.get("prompt", "system", "model")
        assert result is None

    def test_put_and_get(self, cache, sample_response):
        cache.put("test prompt", "system", "llama3.2", sample_response)
        result = cache.get("test prompt", "system", "llama3.2")

        assert result is not None
        assert result.text == "The capital of France is Paris."
        assert result.input_tokens == 20
        assert result.output_tokens == 10

    def test_different_prompts_different_keys(self, cache, sample_response):
        cache.put("prompt_a", "", "llama3.2", sample_response)

        assert cache.get("prompt_a", "", "llama3.2") is not None
        assert cache.get("prompt_b", "", "llama3.2") is None

    def test_different_models_different_keys(self, cache, sample_response):
        cache.put("prompt", "", "llama3.2", sample_response)

        assert cache.get("prompt", "", "llama3.2") is not None
        assert cache.get("prompt", "", "gpt-4o") is None

    def test_different_system_different_keys(self, cache, sample_response):
        cache.put("prompt", "system_a", "llama3.2", sample_response)

        assert cache.get("prompt", "system_a", "llama3.2") is not None
        assert cache.get("prompt", "system_b", "llama3.2") is None


class TestCacheClear:
    """Test cache clearing."""

    def test_clear(self, cache, sample_response):
        cache.put("prompt1", "", "model", sample_response)
        cache.put("prompt2", "", "model", sample_response)

        assert cache.stats()["entries"] == 2

        cache.clear()
        assert cache.stats()["entries"] == 0
        assert cache.get("prompt1", "", "model") is None


class TestCacheTTL:
    """Test cache TTL (time-to-live) expiration."""

    def test_expired_entry(self, tmp_path, sample_response):
        config = CacheConfig(directory=str(tmp_path / "ttl_cache"), ttl_seconds=1)
        cache = ResponseCache(config)

        cache.put("prompt", "", "model", sample_response)
        assert cache.get("prompt", "", "model") is not None

        # Wait for TTL to expire
        time.sleep(1.1)
        assert cache.get("prompt", "", "model") is None

    def test_no_ttl_never_expires(self, tmp_path, sample_response):
        config = CacheConfig(directory=str(tmp_path / "no_ttl"), ttl_seconds=None)
        cache = ResponseCache(config)

        cache.put("prompt", "", "model", sample_response)
        # Should still be valid (no TTL)
        assert cache.get("prompt", "", "model") is not None


class TestCacheStats:
    """Test cache statistics tracking."""

    def test_initial_stats(self, cache):
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["entries"] == 0

    def test_hit_miss_tracking(self, cache, sample_response):
        cache.get("missing", "", "model")  # miss
        cache.put("exists", "", "model", sample_response)
        cache.get("exists", "", "model")  # hit
        cache.get("also_missing", "", "model")  # miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["entries"] == 1
        assert abs(stats["hit_rate"] - 1 / 3) < 0.01

    def test_repr(self, cache, sample_response):
        cache.put("p", "", "m", sample_response)
        r = repr(cache)
        assert "entries=1" in r
