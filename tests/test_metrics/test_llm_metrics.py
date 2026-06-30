from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_eval.core.types import TestSample
from rag_eval.metrics.context_precision import ContextPrecision
from rag_eval.metrics.context_recall import ContextRecall
from rag_eval.metrics.faithfulness import Faithfulness
from rag_eval.metrics.relevance import AnswerRelevance
from rag_eval.utils.llm import LLMClient


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.complete_json = AsyncMock()
    return llm


@pytest.fixture
def sample():
    return TestSample(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        ground_truth="The capital of France is Paris.",
        contexts=["France is a country in Western Europe. Its capital is Paris."],
    )


@pytest.mark.asyncio
async def test_faithfulness(mock_llm, sample):
    mock_llm.complete_json.return_value = {
        "claims": [{"claim": "Paris is the capital of France", "supported": True}],
        "reason": "The context says the capital is Paris.",
        "score": 1.0,
    }

    metric = Faithfulness(mock_llm)
    result = await metric.score(sample)

    assert result.metric_name == "faithfulness"
    assert result.score == 1.0
    assert "Paris is the capital" in result.metadata["claims"][0]["claim"]


@pytest.mark.asyncio
async def test_relevance(mock_llm, sample):
    mock_llm.complete_json.return_value = {"score": 1.0, "reason": "Directly answers the question."}

    metric = AnswerRelevance(mock_llm)
    result = await metric.score(sample)

    assert result.metric_name == "relevance"
    assert result.score == 1.0
    assert result.reason == "Directly answers the question."


@pytest.mark.asyncio
async def test_context_precision(mock_llm, sample):
    mock_llm.complete_json.return_value = {
        "evaluations": [{"chunk_index": 0, "is_relevant": True}],
        "reason": "Chunk 0 discusses the capital of France.",
        "score": 1.0,
    }

    metric = ContextPrecision(mock_llm)
    result = await metric.score(sample)

    assert result.metric_name == "context_precision"
    assert result.score == 1.0
    assert len(result.metadata["evaluations"]) == 1


@pytest.mark.asyncio
async def test_context_recall(mock_llm, sample):
    mock_llm.complete_json.return_value = {
        "statements": [{"statement": "The capital of France is Paris.", "is_covered": True}],
        "reason": "Context covers the ground truth.",
        "score": 1.0,
    }

    metric = ContextRecall(mock_llm)
    result = await metric.score(sample)

    assert result.metric_name == "context_recall"
    assert result.score == 1.0
    assert len(result.metadata["statements"]) == 1
