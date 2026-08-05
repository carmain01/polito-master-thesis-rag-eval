"""Tests for the NoImmediateDisclosure metric."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_eval.core.types import TestSample
from rag_eval.metrics.no_immediate_disclosure import NoImmediateDisclosure
from rag_eval.utils.llm import LLMClient


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.complete_json = AsyncMock()
    return llm


@pytest.fixture
def socratic_sample():
    """A sample where the tutor uses scaffolding (good pedagogy)."""
    return TestSample(
        question="Non so da dove iniziare, mi spieghi come fare?",
        answer=(
            "Ricordi qual è la formula generale per trovare lo spazio occupato "
            "da un rettangolo? Pensa a come si legano la base e l'altezza."
        ),
        contexts=[
            "Problema: Calcola l'area di un rettangolo con base = 5 cm e altezza = 8 cm."
        ],
    )


@pytest.fixture
def revealing_sample():
    """A sample where the tutor reveals the answer immediately (bad pedagogy)."""
    return TestSample(
        question="Non so da dove iniziare, mi spieghi come fare?",
        answer=(
            "L'area si calcola moltiplicando la base per l'altezza, "
            "quindi 5 * 8 = 40 cm^2. La risposta finale è 40."
        ),
        contexts=[
            "Problema: Calcola l'area di un rettangolo con base = 5 cm e altezza = 8 cm."
        ],
    )


# ── Good tutor (Socratic approach) ──────────────────────────────────


@pytest.mark.asyncio
async def test_socratic_tutor_scores_high(mock_llm, socratic_sample):
    """A scaffolding response should score 1.0 NID and high helpfulness."""
    mock_llm.complete_json.side_effect = [
        # First call: no-immediate-disclosure
        {
            "reasoning": "The tutor asks a guiding question without revealing the result.",
            "revealed_immediately": False,
            "score": 1.0,
        },
        # Second call: helpfulness
        {
            "reasoning": "Excellent scaffolding that stimulates active learning.",
            "score": 5,
        },
    ]

    metric = NoImmediateDisclosure(mock_llm)
    result = await metric.score(socratic_sample)

    assert result.metric_name == "no_immediate_disclosure"
    # NID=1.0 * 0.5 + Helpfulness=(5-1)/4 * 0.5 = 0.5 + 0.5 = 1.0
    assert result.score == 1.0
    assert result.metadata["no_immediate_disclosure"]["revealed_immediately"] is False
    assert result.metadata["no_immediate_disclosure"]["score"] == 1.0
    assert result.metadata["helpfulness"]["raw_score"] == 5
    assert result.metadata["helpfulness"]["score"] == 1.0


# ── Bad tutor (reveals the answer) ──────────────────────────────────


@pytest.mark.asyncio
async def test_revealing_tutor_scores_low(mock_llm, revealing_sample):
    """A tutor that reveals the answer immediately should score 0.0 NID."""
    mock_llm.complete_json.side_effect = [
        # First call: no-immediate-disclosure
        {
            "reasoning": "The tutor directly provides the final calculation 5*8=40.",
            "revealed_immediately": True,
            "score": 0.0,
        },
        # Second call: helpfulness
        {
            "reasoning": "The response gives the answer but doesn't stimulate learning.",
            "score": 2,
        },
    ]

    metric = NoImmediateDisclosure(mock_llm)
    result = await metric.score(revealing_sample)

    assert result.metric_name == "no_immediate_disclosure"
    # NID=0.0 * 0.5 + Helpfulness=(2-1)/4 * 0.5 = 0.0 + 0.125 = 0.125
    assert result.score == 0.125
    assert result.metadata["no_immediate_disclosure"]["revealed_immediately"] is True
    assert result.metadata["no_immediate_disclosure"]["score"] == 0.0
    assert result.metadata["helpfulness"]["raw_score"] == 2


# ── Empty answer ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_answer_returns_zero(mock_llm):
    sample = TestSample(question="Help me", answer="", contexts=["Some context"])

    metric = NoImmediateDisclosure(mock_llm)
    result = await metric.score(sample)

    assert result.score == 0.0
    assert "No answer" in result.reason
    mock_llm.complete_json.assert_not_called()


# ── Custom weights ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_custom_weights(mock_llm, socratic_sample):
    """Custom weights should change the combined score."""
    mock_llm.complete_json.side_effect = [
        {
            "reasoning": "No disclosure.",
            "revealed_immediately": False,
            "score": 1.0,
        },
        {
            "reasoning": "Adequate help.",
            "score": 3,
        },
    ]

    # Weight NID 80%, helpfulness 20%
    metric = NoImmediateDisclosure(
        mock_llm,
        weights={"no_immediate_disclosure": 0.8, "helpfulness": 0.2},
    )
    result = await metric.score(socratic_sample)

    # NID=1.0 * 0.8 + Helpfulness=(3-1)/4 * 0.2 = 0.8 + 0.1 = 0.9
    assert result.score == 0.9


# ── LLM error handling ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_error_returns_zero(mock_llm, socratic_sample):
    """If the LLM call fails, the metric should return score 0.0 gracefully."""
    mock_llm.complete_json.side_effect = RuntimeError("LLM service unavailable")

    metric = NoImmediateDisclosure(mock_llm)
    result = await metric.score(socratic_sample)

    assert result.score == 0.0
    assert "Failed to evaluate" in result.reason
    assert "error" in result.metadata


# ── No contexts ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_contexts_still_works(mock_llm):
    """The metric should work even when no contexts are provided."""
    sample = TestSample(
        question="What is 2+2?",
        answer="Think about what happens when you combine two groups of two.",
        contexts=[],
    )
    mock_llm.complete_json.side_effect = [
        {
            "reasoning": "The tutor guides without revealing.",
            "revealed_immediately": False,
            "score": 1.0,
        },
        {
            "reasoning": "Good scaffolding.",
            "score": 4,
        },
    ]

    metric = NoImmediateDisclosure(mock_llm)
    result = await metric.score(sample)

    assert result.metric_name == "no_immediate_disclosure"
    # NID=1.0 * 0.5 + Helpfulness=(4-1)/4 * 0.5 = 0.5 + 0.375 = 0.875
    assert result.score == 0.875


# ── Malformed LLM output ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_malformed_scores_are_clamped(mock_llm, socratic_sample):
    """Invalid/out-of-range scores from the LLM should be clamped safely."""
    mock_llm.complete_json.side_effect = [
        {
            "reasoning": "Unclear.",
            "revealed_immediately": False,
            "score": 2.5,  # Out of range – should be clamped to 1.0
        },
        {
            "reasoning": "Bad value.",
            "score": "not_a_number",  # Invalid – should default to 1
        },
    ]

    metric = NoImmediateDisclosure(mock_llm)
    result = await metric.score(socratic_sample)

    assert 0.0 <= result.score <= 1.0
    # NID clamped to 1.0 * 0.5 + Helpfulness=(1-1)/4 * 0.5 = 0.5 + 0.0 = 0.5
    assert result.score == 0.5
