from unittest.mock import MagicMock

import pytest

from rag_eval.core.types import TestSample
from rag_eval.metrics.bert_score import BERTScore
from rag_eval.metrics.bleu import BLEU
from rag_eval.metrics.exact_match import ExactMatch
from rag_eval.metrics.f1 import TokenF1
from rag_eval.metrics.rouge import ROUGE
from rag_eval.metrics.semantic_similarity import SemanticSimilarity
from rag_eval.utils.embeddings import EmbeddingClient


@pytest.fixture
def sample():
    return TestSample(
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        ground_truth="The capital of France is Paris.",
        contexts=[],
    )


@pytest.fixture
def sample_mismatch():
    return TestSample(
        question="What is the capital of France?",
        answer="London is the capital of UK.",
        ground_truth="The capital of France is Paris.",
        contexts=[],
    )


@pytest.mark.asyncio
async def test_bleu(sample, sample_mismatch):
    metric = BLEU(variant=4)
    res_match = await metric.score(sample)
    assert res_match.metric_name == "bleu-4"
    assert res_match.score == 1.0

    res_mis = await metric.score(sample_mismatch)
    assert res_mis.score < 1.0


@pytest.mark.asyncio
async def test_rouge(sample, sample_mismatch):
    metric = ROUGE(variant="rougeL")
    res_match = await metric.score(sample)
    assert res_match.metric_name == "rougeL"
    assert res_match.score == 1.0

    res_mis = await metric.score(sample_mismatch)
    assert res_mis.score < 1.0


@pytest.mark.asyncio
async def test_f1(sample, sample_mismatch):
    metric = TokenF1()
    res_match = await metric.score(sample)
    assert res_match.metric_name == "token_f1"
    assert res_match.score == 1.0

    res_mis = await metric.score(sample_mismatch)
    assert res_mis.score < 1.0


@pytest.mark.asyncio
async def test_exact_match(sample, sample_mismatch):
    metric = ExactMatch()
    res_match = await metric.score(sample)
    assert res_match.metric_name == "exact_match"
    assert res_match.score == 1.0

    res_mis = await metric.score(sample_mismatch)
    assert res_mis.score == 0.0


@pytest.mark.asyncio
async def test_semantic_similarity(sample, sample_mismatch):
    mock_embed = MagicMock(spec=EmbeddingClient)
    mock_embed.embed_single = MagicMock(
        side_effect=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )

    metric = SemanticSimilarity(mock_embed)
    res_match = await metric.score(sample)
    assert res_match.metric_name == "semantic_similarity"
    assert res_match.score == 1.0

    res_mis = await metric.score(sample_mismatch)
    assert res_mis.score == 0.0


@pytest.mark.asyncio
async def test_bert_score(sample, sample_mismatch):
    # For speed and isolation in tests, we can use a very tiny model or just verify initialization
    # To run a real BERTScore test we use albert-base-v2 which is smaller
    metric = BERTScore(model_type="albert-base-v2")
    res_match = await metric.score(sample)
    assert res_match.metric_name == "bert_score"
    # Even exact match might not be exactly 1.0 for some models in bert_score due to scaling
    assert res_match.score > 0.8
