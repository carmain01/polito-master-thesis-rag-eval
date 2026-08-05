"""Tests for the Uptake metric."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from rag_eval.core.types import TestSample
from rag_eval.metrics.uptake import Uptake


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_uptake():
    """Create an Uptake instance with mocked model and tokenizer."""
    with (
        patch("rag_eval.metrics.uptake.AutoTokenizer") as mock_tok_cls,
        patch("rag_eval.metrics.uptake.AutoModelForSequenceClassification") as mock_model_cls,
    ):
        # Set up the tokenizer mock to return tensor-like inputs.
        # We use a MagicMock wrapping the dict so that .to() is supported
        # and the object can still be unpacked with **kwargs by the model.
        mock_tokenizer = MagicMock()
        fake_inputs = MagicMock()
        tensor_data = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        fake_inputs.__getitem__ = lambda self, key: tensor_data[key]
        fake_inputs.keys.return_value = tensor_data.keys()
        fake_inputs.to.return_value = fake_inputs
        mock_tokenizer.return_value = fake_inputs
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        # Set up the model mock — high uptake by default (prob ≈ 0.73)
        mock_model = MagicMock()
        mock_model.return_value.logits = torch.tensor([[0.0, 1.0]])
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        metric = Uptake(model_path="fake/model", device="cpu")
        yield metric


@pytest.fixture
def sample_full():
    return TestSample(
        question="What is the process of photosynthesis in plants?",
        answer="Photosynthesis is the process by which plants convert sunlight into energy.",
        ground_truth="",
        contexts=["Plants use chlorophyll to absorb sunlight and produce glucose."],
    )


@pytest.fixture
def sample_short_question():
    return TestSample(
        question="What?",
        answer="Photosynthesis is a complex process.",
        ground_truth="",
        contexts=["Context about photosynthesis."],
    )


@pytest.fixture
def sample_no_answer():
    return TestSample(
        question="What is photosynthesis?",
        answer="",
        ground_truth="",
        contexts=[],
    )


@pytest.fixture
def sample_no_context():
    return TestSample(
        question="What is the detailed process of photosynthesis in plants?",
        answer="Photosynthesis converts light energy into chemical energy.",
        ground_truth="",
        contexts=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uptake_name(mock_uptake):
    assert mock_uptake.name == "uptake"


@pytest.mark.asyncio
async def test_uptake_full_sample(mock_uptake, sample_full):
    result = await mock_uptake.score(sample_full)

    assert result.metric_name == "uptake"
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["conversational_uptake"] is not None
    assert result.metadata["context_uptake"] is not None
    assert result.metadata["context_overlap"] is not None


@pytest.mark.asyncio
async def test_uptake_no_answer(mock_uptake, sample_no_answer):
    result = await mock_uptake.score(sample_no_answer)

    assert result.score == 0.0
    assert "No answer" in result.reason


@pytest.mark.asyncio
async def test_uptake_short_question(mock_uptake, sample_short_question):
    result = await mock_uptake.score(sample_short_question)

    # Conversational uptake should be skipped for short questions
    assert result.metadata["conversational_uptake"] is None
    # Context uptake should still be computed
    assert result.metadata["context_uptake"] is not None


@pytest.mark.asyncio
async def test_uptake_no_context(mock_uptake, sample_no_context):
    result = await mock_uptake.score(sample_no_context)

    # Conversational uptake should be computed
    assert result.metadata["conversational_uptake"] is not None
    # Context uptake should be None when no contexts are provided
    assert result.metadata["context_uptake"] is None
    assert result.metadata["context_overlap"] is None


@pytest.mark.asyncio
async def test_uptake_score_bounded(mock_uptake, sample_full):
    result = await mock_uptake.score(sample_full)
    assert 0.0 <= result.score <= 1.0


def test_word_overlap_identical():
    """Overlap of a text with itself should be 1.0 (minus stop words)."""
    text = "machine learning algorithms process large datasets effectively"
    overlap = Uptake._compute_word_overlap(text, text)
    assert overlap == 1.0


def test_word_overlap_disjoint():
    """Completely different texts should have zero overlap."""
    src = "alpha beta gamma"
    tgt = "delta epsilon zeta"
    overlap = Uptake._compute_word_overlap(src, tgt)
    assert overlap == 0.0


def test_word_overlap_empty_source():
    overlap = Uptake._compute_word_overlap("", "some target text")
    assert overlap == 0.0
