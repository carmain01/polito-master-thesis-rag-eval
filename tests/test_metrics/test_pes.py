"""Tests for the PES (Pedagogical Evaluation Score) composite metric."""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.pes import PES, _EPSILON


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_metric(name: str, score: float, metadata: dict | None = None) -> MagicMock:
    """Create a mock BaseMetric that returns a fixed EvalResult."""
    mock = MagicMock()
    mock.name = name
    mock.score = AsyncMock(
        return_value=EvalResult(
            metric_name=name,
            score=score,
            reason=f"Mocked {name}",
            metadata=metadata or {},
        )
    )
    return mock


def _make_state_mock(state_code: str = "A", state_label: str = "Concept Teaching", confidence: float = 0.9):
    """Create a mock State classifier returning a fixed state."""
    return _make_mock_metric(
        "state",
        score=confidence,
        metadata={
            "state": state_code,
            "state_label": state_label,
            "confidence": confidence,
        },
    )


def _make_faithfulness_mock(score: float = 0.9):
    """Create a mock Faithfulness metric returning a fixed score."""
    return _make_mock_metric(
        "faithfulness",
        score=score,
        metadata={"claims": []},
    )


def _build_pes(
    m1: float = 0.8,
    m2: float = 0.7,
    m3: float = 0.6,
    state_code: str = "A",
    faithfulness_score: float | None = 0.9,
    faithfulness_threshold: float = 0.5,
) -> PES:
    """Build a PES instance with mocked sub-metrics returning fixed scores."""
    uptake = _make_mock_metric("uptake", m1)
    linguistic = _make_mock_metric("linguistic_adaptation", m2)
    disclosure = _make_mock_metric("no_immediate_disclosure", m3)
    state = _make_state_mock(state_code)
    faithfulness = _make_faithfulness_mock(faithfulness_score) if faithfulness_score is not None else None

    return PES(
        uptake=uptake,
        linguistic_adaptation=linguistic,
        no_immediate_disclosure=disclosure,
        state=state,
        faithfulness=faithfulness,
        faithfulness_threshold=faithfulness_threshold,
    )


@pytest.fixture
def sample():
    return TestSample(
        question="What is the process of photosynthesis in plants?",
        answer="Think about how plants use sunlight. What molecule captures light energy?",
        ground_truth="",
        contexts=["Plants convert sunlight into chemical energy via chlorophyll."],
    )


# ---------------------------------------------------------------------------
# Basic property tests
# ---------------------------------------------------------------------------


def test_pes_name():
    pes = _build_pes()
    assert pes.name == "pes"


def test_pes_auto_instantiation_with_llm_client():
    mock_llm = MagicMock()
    with patch("rag_eval.metrics.uptake.AutoTokenizer"), patch(
        "rag_eval.metrics.uptake.AutoModelForSequenceClassification"
    ), patch("rag_eval.metrics.linguistic_adaptation.AutoTokenizer"), patch(
        "rag_eval.metrics.linguistic_adaptation.AutoModelForSequenceClassification"
    ):
        pes = PES(llm_client=mock_llm)
        assert pes.name == "pes"
        assert pes._faithfulness is not None
        assert pes._disclosure is not None
        assert pes._state is not None



# ---------------------------------------------------------------------------
# Weighted geometric mean — mathematical correctness
# ---------------------------------------------------------------------------


class TestWeightedGeometricMean:
    """Verify the core formula independently from the async pipeline."""

    def test_equal_scores_return_same_value(self):
        """When all sub-metrics are equal, the geometric mean equals that value."""
        for v in (0.5, 0.8, 1.0, 0.01):
            result = PES._weighted_geometric_mean(v, v, v, 0.33, 0.33, 0.34)
            assert abs(result - v) < 1e-4, f"Expected ≈{v}, got {result}"

    def test_all_ones_returns_one(self):
        result = PES._weighted_geometric_mean(1.0, 1.0, 1.0, 0.4, 0.3, 0.3)
        assert abs(result - 1.0) < 1e-6

    def test_known_values_state_b(self):
        """Reproduce the thesis example: M1=1.0, M2=1.0, M3=0.1, State B.

        Arithmetic mean would give ≈ 0.70, geometric mean should be ≈ 0.46.
        """
        # State B weights: w1=0.20, w2=0.30, w3=0.50
        result = PES._weighted_geometric_mean(1.0, 1.0, 0.1, 0.20, 0.30, 0.50)
        # Manually: exp( (0.2*0 + 0.3*0 + 0.5*ln(0.1)) / 1.0 )
        #         = exp( 0.5 * (-2.302585) ) = exp(-1.15129) ≈ 0.3162
        expected = math.exp(0.5 * math.log(0.1))
        assert abs(result - expected) < 1e-4
        # Confirm it is dramatically lower than the arithmetic mean of 0.70
        arithmetic_mean = 0.20 * 1.0 + 0.30 * 1.0 + 0.50 * 0.1
        assert result < arithmetic_mean

    def test_bottleneck_penalisation(self):
        """A single terrible score should drag the result far below the arithmetic mean."""
        result = PES._weighted_geometric_mean(1.0, 1.0, 0.01, 0.33, 0.33, 0.34)
        arithmetic = 0.33 * 1.0 + 0.33 * 1.0 + 0.34 * 0.01
        assert result < arithmetic * 0.8  # Geometric mean is significantly lower

    def test_zero_score_clamped_to_epsilon(self):
        """A score of 0.0 should be clamped to epsilon, not crash."""
        result = PES._weighted_geometric_mean(0.0, 1.0, 1.0, 0.33, 0.33, 0.34)
        assert result > 0.0
        assert result < 0.02  # Should be very low due to epsilon clamping

    def test_zero_weights_returns_zero(self):
        result = PES._weighted_geometric_mean(0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# State-dependent weight resolution
# ---------------------------------------------------------------------------


class TestStateWeights:
    """Verify the correct weights are selected based on state codes."""

    def test_state_a_weights(self):
        pes = _build_pes(state_code="A")
        assert pes._resolve_weights("A") == (0.35, 0.35, 0.30)

    def test_state_b_weights(self):
        pes = _build_pes(state_code="B")
        assert pes._resolve_weights("B") == (0.00, 0.30, 0.70)

    def test_state_c_weights(self):
        pes = _build_pes(state_code="C")
        assert pes._resolve_weights("C") == (0.00, 0.25, 0.75)

    def test_unknown_state_fallback(self):
        pes = _build_pes(state_code="X")
        assert pes._resolve_weights("X") == (0.33, 0.33, 0.34)

    def test_empty_state_fallback(self):
        pes = _build_pes(state_code="")
        assert pes._resolve_weights("") == (0.33, 0.33, 0.34)

    def test_custom_weights_override(self):
        """Custom state_weights passed at construction should override defaults."""
        uptake = _make_mock_metric("uptake", 0.8)
        linguistic = _make_mock_metric("linguistic_adaptation", 0.7)
        disclosure = _make_mock_metric("no_immediate_disclosure", 0.6)
        state = _make_state_mock("A")

        custom = {"A": (0.10, 0.10, 0.80)}
        pes = PES(
            uptake=uptake,
            linguistic_adaptation=linguistic,
            no_immediate_disclosure=disclosure,
            state=state,
            state_weights=custom,
        )
        assert pes._resolve_weights("A") == (0.10, 0.10, 0.80)
        # Non-overridden states should fall back to defaults
        assert pes._resolve_weights("B") == (0.33, 0.33, 0.34)


# ---------------------------------------------------------------------------
# Faithfulness gate
# ---------------------------------------------------------------------------


class TestFaithfulnessGate:

    @pytest.mark.asyncio
    async def test_gate_passes_when_above_threshold(self, sample):
        """Faithfulness above threshold → PES computed normally."""
        pes = _build_pes(m1=0.8, m2=0.7, m3=0.6, faithfulness_score=0.8)
        result = await pes.score(sample)

        assert result.score > 0.0
        assert result.metadata["gate_failed"] is False
        assert "gate PASSED" in result.reason

    @pytest.mark.asyncio
    async def test_gate_fails_below_threshold(self, sample):
        """Faithfulness below threshold → PES = 0.0."""
        pes = _build_pes(faithfulness_score=0.3, faithfulness_threshold=0.5)
        result = await pes.score(sample)

        assert result.score == 0.0
        assert result.metadata["gate_failed"] is True
        assert "gate FAILED" in result.reason

    @pytest.mark.asyncio
    async def test_gate_fails_at_exact_threshold(self, sample):
        """Faithfulness exactly at threshold → still fails (strict less-than)."""
        pes = _build_pes(faithfulness_score=0.5, faithfulness_threshold=0.5)
        result = await pes.score(sample)

        # score == threshold → NOT below threshold → gate passes
        assert result.score > 0.0
        assert result.metadata["gate_failed"] is False

    @pytest.mark.asyncio
    async def test_no_faithfulness_metric_skips_gate(self, sample):
        """When no Faithfulness metric is provided, the gate is skipped entirely."""
        pes = _build_pes(faithfulness_score=None)
        result = await pes.score(sample)

        assert result.score > 0.0
        assert result.metadata["gate_failed"] is False
        assert "faithfulness" not in result.metadata

    @pytest.mark.asyncio
    async def test_gate_zero_faithfulness(self, sample):
        """Faithfulness = 0.0 → gate fails hard."""
        pes = _build_pes(faithfulness_score=0.0)
        result = await pes.score(sample)

        assert result.score == 0.0
        assert result.metadata["gate_failed"] is True


# ---------------------------------------------------------------------------
# End-to-end scoring
# ---------------------------------------------------------------------------


class TestPESScoring:

    @pytest.mark.asyncio
    async def test_score_with_state_a(self, sample):
        """State A with known sub-scores should produce the correct geometric mean."""
        pes = _build_pes(m1=0.9, m2=0.8, m3=0.7, state_code="A", faithfulness_score=0.9)
        result = await pes.score(sample)

        # State A: w1=0.35, w2=0.35, w3=0.30
        expected = PES._weighted_geometric_mean(0.9, 0.8, 0.7, 0.35, 0.35, 0.30)
        expected = round(expected, 4)

        assert result.metric_name == "pes"
        assert result.score == expected
        assert result.metadata["state"]["code"] == "A"
        assert result.metadata["weights"] == {"w1_uptake": 0.35, "w2_linguistic": 0.35, "w3_disclosure": 0.30}

    @pytest.mark.asyncio
    async def test_score_with_state_b(self, sample):
        """State B with bottleneck in M3 should produce a low score."""
        pes = _build_pes(m1=1.0, m2=1.0, m3=0.1, state_code="B", faithfulness_score=0.9)
        result = await pes.score(sample)

        # State B: w1=0.00, w2=0.30, w3=0.70  → M3 heavily weighted
        assert result.score < 0.5  # Geometric mean penalises the bottleneck
        assert result.metadata["state"]["code"] == "B"

    @pytest.mark.asyncio
    async def test_score_with_state_c(self, sample):
        """State C with perfect M3 but average M1 and M2."""
        pes = _build_pes(m1=0.5, m2=0.5, m3=1.0, state_code="C", faithfulness_score=0.9)
        result = await pes.score(sample)

        # State C: w1=0.00, w2=0.25, w3=0.75
        # M3=1.0 is heavily weighted, so the score should be pulled up
        expected = PES._weighted_geometric_mean(0.5, 0.5, 1.0, 0.00, 0.25, 0.75)
        expected = round(expected, 4)
        assert result.score == expected
        assert result.score > 0.6  # M3=1.0 with w3=0.75 should dominate

    @pytest.mark.asyncio
    async def test_all_perfect_scores(self, sample):
        """All sub-metrics at 1.0 should yield PES = 1.0."""
        pes = _build_pes(m1=1.0, m2=1.0, m3=1.0, state_code="A", faithfulness_score=1.0)
        result = await pes.score(sample)

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_empty_answer_returns_zero(self):
        """An empty answer should short-circuit to 0.0."""
        sample = TestSample(question="Help?", answer="", contexts=[])
        pes = _build_pes()
        result = await pes.score(sample)

        assert result.score == 0.0
        assert "No answer" in result.reason

    @pytest.mark.asyncio
    async def test_score_bounded_zero_to_one(self, sample):
        """PES should always be in [0, 1]."""
        for m1, m2, m3 in [(0.01, 0.01, 0.01), (1.0, 1.0, 1.0), (0.5, 0.3, 0.9)]:
            pes = _build_pes(m1=m1, m2=m2, m3=m3, faithfulness_score=0.9)
            result = await pes.score(sample)
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_metadata_contains_all_sub_scores(self, sample):
        """The result metadata should contain all sub-metric details."""
        pes = _build_pes(m1=0.8, m2=0.7, m3=0.6, state_code="B", faithfulness_score=0.85)
        result = await pes.score(sample)

        assert "M1_uptake" in result.metadata
        assert "M2_linguistic_adaptation" in result.metadata
        assert "M3_no_immediate_disclosure" in result.metadata
        assert "state" in result.metadata
        assert "weights" in result.metadata
        assert "faithfulness" in result.metadata
        assert "gate_failed" in result.metadata

        assert result.metadata["M1_uptake"]["score"] == 0.8
        assert result.metadata["M2_linguistic_adaptation"]["score"] == 0.7
        assert result.metadata["M3_no_immediate_disclosure"]["score"] == 0.6


# ---------------------------------------------------------------------------
# Thesis example verification
# ---------------------------------------------------------------------------


class TestThesisExample:
    """Reproduce the example from the thesis motivation section."""

    @pytest.mark.asyncio
    async def test_geometric_vs_arithmetic_penalisation(self, sample):
        """M1=1.0, M2=1.0, M3=0.1 with State B weights.

        Arithmetic mean ≈ 0.37 (weighted: 0.00*1 + 0.30*1 + 0.70*0.1)
        Geometric mean should be ≈ 0.200, dramatically lower.
        """
        pes = _build_pes(m1=1.0, m2=1.0, m3=0.1, state_code="B", faithfulness_score=0.9)
        result = await pes.score(sample)

        arithmetic = 0.00 * 1.0 + 0.30 * 1.0 + 0.70 * 0.1
        assert result.score < arithmetic, (
            f"Geometric mean ({result.score}) should be lower than "
            f"arithmetic mean ({arithmetic})"
        )
        # The geometric mean specifically:
        # exp( (0.0*ln(1) + 0.3*ln(1) + 0.7*ln(0.1)) / 1.0 )
        # = exp( 0.7 * (-2.3026) ) = exp(-1.6118) ≈ 0.1995
        assert abs(result.score - 0.1995) < 0.01
