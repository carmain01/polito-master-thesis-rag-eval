"""Tests for the LinguisticAdaptation metric."""

import pytest

from rag_eval.core.types import TestSample
from rag_eval.metrics.linguistic_adaptation import LinguisticAdaptation


@pytest.fixture
def metric():
    return LinguisticAdaptation()


@pytest.fixture
def sample_pedagogical():
    """A response with tentative language and constructive contrast — should score well."""
    return TestSample(
        question="What is photosynthesis?",
        answer=(
            "Photosynthesis is the process by which plants convert sunlight into energy. "
            "You might think of it as the plant's way of eating. "
            "Perhaps the most interesting part is the role of chlorophyll, "
            "but it is also important to consider how water and CO2 contribute. "
            "Could you guess which step produces oxygen?"
        ),
        ground_truth="",
        contexts=[],
    )


@pytest.fixture
def sample_empty():
    return TestSample(
        question="What is photosynthesis?",
        answer="",
        ground_truth="",
        contexts=[],
    )


@pytest.fixture
def sample_authoritative():
    """A response that is direct and authoritative — lower tentativeness expected."""
    return TestSample(
        question="What is the capital of Italy?",
        answer="The capital of Italy is Rome. Rome is located in central Italy.",
        ground_truth="",
        contexts=[],
    )


@pytest.fixture
def sample_italian_socratic():
    """An Italian Socratic response with conjugated tentative verbs."""
    return TestSample(
        question="Perché il Selection Sort è sempre O(n^2)?",
        answer=(
            "Proviamo a riflettere insieme su come funzionano i cicli annidati. "
            "Anche se l'array fosse già ordinato, pensando ai confronti necessari, "
            "potrebbe l'algoritmo sapere che un elemento è il minimo senza "
            "scorrere tutto il resto del sotto-array?"
        ),
        ground_truth="",
        contexts=[],
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_name(metric):
    assert metric.name == "linguistic_adaptation"


@pytest.mark.asyncio
async def test_empty_answer(metric, sample_empty):
    result = await metric.score(sample_empty)
    assert result.score == 0.0
    assert "No answer" in result.reason


@pytest.mark.asyncio
async def test_score_in_range(metric, sample_pedagogical):
    result = await metric.score(sample_pedagogical)
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_metadata_keys(metric, sample_pedagogical):
    result = await metric.score(sample_pedagogical)
    assert "readability" in result.metadata
    assert "socratic_tentativeness" in result.metadata
    assert "constructive_critique" in result.metadata


@pytest.mark.asyncio
async def test_pedagogical_scores_higher_tentativeness(metric, sample_pedagogical, sample_authoritative):
    """The pedagogical sample has tentative words; the authoritative one does not."""
    res_ped = await metric.score(sample_pedagogical)
    res_auth = await metric.score(sample_authoritative)

    tent_ped = res_ped.metadata["socratic_tentativeness"]["tentativeness_ratio"]
    tent_auth = res_auth.metadata["socratic_tentativeness"]["tentativeness_ratio"]
    assert tent_ped > tent_auth


@pytest.mark.asyncio
async def test_constructive_contrast_detection(metric, sample_pedagogical, sample_authoritative):
    """The pedagogical sample contains 'but', the authoritative one does not."""
    res_ped = await metric.score(sample_pedagogical)
    res_auth = await metric.score(sample_authoritative)

    assert res_ped.metadata["constructive_critique"]["has_constructive_contrast"] is True
    assert res_auth.metadata["constructive_critique"]["has_constructive_contrast"] is False

    # Verify new metadata keys from multilingual BERT sentiment model
    critique = res_ped.metadata["constructive_critique"]
    assert "sentiment_score" in critique
    assert "predicted_stars" in critique
    assert "star_probabilities" in critique
    assert 1 <= critique["predicted_stars"] <= 5
    assert len(critique["star_probabilities"]) == 5


@pytest.mark.asyncio
async def test_custom_weights():
    """Custom weights should influence the final score."""
    metric_read = LinguisticAdaptation(
        weights={"readability": 1.0, "socratic_tentativeness": 0.0, "constructive_critique": 0.0}
    )
    metric_tent = LinguisticAdaptation(
        weights={"readability": 0.0, "socratic_tentativeness": 1.0, "constructive_critique": 0.0}
    )

    sample = TestSample(
        question="Explain gravity.",
        answer=(
            "Gravity is a force. It seems like everything falls down. "
            "Maybe you could think about what happens on the moon."
        ),
        ground_truth="",
        contexts=[],
    )

    res_read = await metric_read.score(sample)
    res_tent = await metric_tent.score(sample)

    # The scores should differ because the weights emphasise different components
    assert res_read.score != res_tent.score


@pytest.mark.asyncio
async def test_readability_metadata(metric, sample_pedagogical):
    result = await metric.score(sample_pedagogical)
    read = result.metadata["readability"]
    assert "flesch_reading_ease" in read
    assert "flesch_kincaid_grade" in read
    assert "word_count" in read
    assert read["word_count"] > 0


@pytest.mark.asyncio
async def test_detected_language_in_metadata(metric, sample_pedagogical, sample_italian_socratic):
    """Language detection should report the detected language in tentativeness metadata."""
    res_en = await metric.score(sample_pedagogical)
    res_it = await metric.score(sample_italian_socratic)

    assert "detected_language" in res_en.metadata["socratic_tentativeness"]
    assert "detected_language" in res_it.metadata["socratic_tentativeness"]

    assert res_en.metadata["socratic_tentativeness"]["detected_language"] == "en"
    assert res_it.metadata["socratic_tentativeness"]["detected_language"] == "it"


@pytest.mark.asyncio
async def test_italian_lemmatization_detects_tentative_words(metric, sample_italian_socratic):
    """Italian conjugated forms like 'proviamo', 'riflettere', 'potrebbe' should be
    detected as tentative via lemmatisation (provare, riflettere, potere)."""
    result = await metric.score(sample_italian_socratic)
    tent = result.metadata["socratic_tentativeness"]

    # At least some tentative words should be found thanks to lemmatisation
    assert tent["tentative_word_count"] > 0, (
        f"Expected >0 tentative words in Italian Socratic text, got {tent['tentative_word_count']}"
    )
    assert tent["tentativeness_ratio"] > 0.0
    assert tent["score"] > 0.0
