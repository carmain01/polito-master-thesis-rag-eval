"""Tests for the UsageTracker."""

from __future__ import annotations

import pytest

from rag_eval.utils.provider import LLMResponse
from rag_eval.utils.tracking import UsageTracker


@pytest.fixture
def tracker():
    return UsageTracker()


@pytest.fixture
def openai_response():
    return LLMResponse(
        text="Hello",
        input_tokens=100,
        output_tokens=50,
        model="gpt-4o",
        provider="openai",
        latency_ms=500.0,
        cost_estimate=0.00075,
    )


@pytest.fixture
def ollama_response():
    return LLMResponse(
        text="Hi",
        input_tokens=30,
        output_tokens=20,
        model="llama3.2",
        provider="ollama",
        latency_ms=200.0,
        cost_estimate=0.0,
    )


class TestUsageTracker:
    """Test usage tracking."""

    def test_initial_state(self, tracker):
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0
        assert tracker.call_count == 0

    def test_record_single(self, tracker, openai_response):
        tracker.record(openai_response)

        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert tracker.total_tokens == 150
        assert tracker.call_count == 1
        assert abs(tracker.total_cost - 0.00075) < 1e-8

    def test_record_multiple(self, tracker, openai_response, ollama_response):
        tracker.record(openai_response)
        tracker.record(ollama_response)

        assert tracker.total_input_tokens == 130
        assert tracker.total_output_tokens == 70
        assert tracker.total_tokens == 200
        assert tracker.call_count == 2
        assert abs(tracker.total_cost - 0.00075) < 1e-8  # Ollama is free

    def test_per_model_breakdown(self, tracker, openai_response, ollama_response):
        tracker.record(openai_response)
        tracker.record(ollama_response)
        tracker.record(openai_response)

        summary = tracker.summary()
        per_model = summary["per_model"]

        assert "openai/gpt-4o" in per_model
        assert per_model["openai/gpt-4o"]["calls"] == 2
        assert per_model["openai/gpt-4o"]["input_tokens"] == 200

        assert "ollama/llama3.2" in per_model
        assert per_model["ollama/llama3.2"]["calls"] == 1

    def test_reset(self, tracker, openai_response):
        tracker.record(openai_response)
        assert tracker.call_count == 1

        tracker.reset()
        assert tracker.call_count == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0
        assert tracker.summary()["per_model"] == {}

    def test_summary_format(self, tracker, openai_response):
        tracker.record(openai_response)
        summary = tracker.summary()

        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "total_tokens" in summary
        assert "total_cost_usd" in summary
        assert "total_calls" in summary
        assert "per_model" in summary

    def test_repr(self, tracker, openai_response):
        tracker.record(openai_response)
        r = repr(tracker)
        assert "calls=1" in r
        assert "tokens=150" in r
