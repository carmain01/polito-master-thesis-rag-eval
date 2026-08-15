"""PES — Pedagogical Evaluation Score (composite metric).

Combines three pedagogical sub-metrics into a single 0–1 score using a
**Weighted Geometric Mean** conditioned on the detected pedagogical State:

.. math::

    \\text{PES} = \\left(
        M_1^{w_1(S)} \\cdot M_2^{w_2(S)} \\cdot M_3^{w_3(S)}
    \\right)^{\\frac{1}{w_1(S)+w_2(S)+w_3(S)}}

Sub-metrics
-----------
- **M1 — Uptake**: conversational & context elaboration (NUC-BERT).
- **M2 — Linguistic Adaptation**: readability + socratic tentativeness +
  constructive critique.
- **M3 — No-Immediate-Disclosure**: LLM-judged scaffolding quality +
  helpfulness.

State-dependent weights
-----------------------
The weights ``w_i(S)`` shift based on the pedagogical intent detected by the
:class:`State` classifier:

+-------+-----------------------+------+------+------+
| State | Label                 |  w1  |  w2  |  w3  |
+=======+=======================+======+======+======+
|   A   | Concept Teaching      | 0.35 | 0.35 | 0.30 |
+-------+-----------------------+------+------+------+
|   B   | Error Remediation     | 0.00 | 0.30 | 0.70 |
+-------+-----------------------+------+------+------+
|   C   | Socratic Assessment   | 0.00 | 0.25 | 0.75 |
+-------+-----------------------+------+------+------+
| other | Default / Unknown     | 0.33 | 0.33 | 0.34 |
+-------+-----------------------+------+------+------+

Faithfulness Gate (Level 0)
---------------------------
If a :class:`Faithfulness` metric is provided, it is run first.  When the
faithfulness score drops below a configurable threshold (default **0.5**),
the PES is forced to **0.0** — a hallucinating answer must never receive a
positive pedagogical score.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from rag_eval.core.types import EvalResult, TestSample
from rag_eval.metrics.base import BaseMetric
from rag_eval.metrics.faithfulness import Faithfulness
from rag_eval.metrics.linguistic_adaptation import LinguisticAdaptation
from rag_eval.metrics.no_immediate_disclosure import NoImmediateDisclosure
from rag_eval.metrics.state import State
from rag_eval.metrics.uptake import Uptake

from rag_eval.utils.llm import LLMClient

logger = logging.getLogger(__name__)

# Epsilon floor to avoid log(0) in the geometric-mean computation.
_EPSILON: float = 1e-6

# ---------------------------------------------------------------------------
# State → weight mapping
# ---------------------------------------------------------------------------
# Keys are the single-character state codes emitted by the State classifier.
# Values are (w1_uptake, w2_linguistic, w3_disclosure) tuples.
_STATE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "A": (0.35, 0.35, 0.30),  # Concept Teaching
    "B": (0.00, 0.30, 0.70),  # Error Remediation
    "C": (0.00, 0.25, 0.75),  # Socratic Assessment
}

_DEFAULT_WEIGHTS: tuple[float, float, float] = (0.33, 0.33, 0.34)


class PES(BaseMetric):
    """Pedagogical Evaluation Score — composite weighted-geometric-mean metric.

    Parameters
    ----------
    uptake : Uptake | None
        Pre-initialised Uptake metric instance (M1). If None, constructed automatically.
    linguistic_adaptation : LinguisticAdaptation | None
        Pre-initialised Linguistic Adaptation metric instance (M2). If None, constructed automatically.
    no_immediate_disclosure : NoImmediateDisclosure | None
        Pre-initialised No-Immediate-Disclosure metric instance (M3). If None, constructed automatically via *llm_client*.
    state : State | None
        Pre-initialised State classifier. If None, constructed automatically via *llm_client*.
    faithfulness : Faithfulness | None
        Optional pre-initialised Faithfulness metric.  When provided or constructed via *llm_client*, it acts
        as a safety gate: if the faithfulness score falls below
        *faithfulness_threshold*, the PES is forced to 0.0.
    llm_client : LLMClient | None
        The LLM client used to auto-instantiate LLM-based sub-metrics when not explicitly provided.
    faithfulness_threshold : float
        Minimum faithfulness score required for the gate to pass.
        Defaults to **0.5**.
    state_weights : dict[str, tuple[float, float, float]] | None
        Optional override for the state → weight mapping.  Keys should be
        ``"A"``, ``"B"``, ``"C"``; values are ``(w1, w2, w3)`` tuples.
    default_weights : tuple[float, float, float] | None
        Optional override for the fallback weights used when the state is
        unknown or classification fails.
    """

    def __init__(
        self,
        uptake: Uptake | None = None,
        linguistic_adaptation: LinguisticAdaptation | None = None,
        no_immediate_disclosure: NoImmediateDisclosure | None = None,
        state: State | None = None,
        faithfulness: Faithfulness | None = None,
        llm_client: LLMClient | None = None,
        faithfulness_threshold: float = 0.5,
        state_weights: dict[str, tuple[float, float, float]] | None = None,
        default_weights: tuple[float, float, float] | None = None,
    ) -> None:
        self.llm = llm_client

        self._uptake = uptake if uptake is not None else Uptake()
        self._linguistic = (
            linguistic_adaptation if linguistic_adaptation is not None else LinguisticAdaptation()
        )

        if no_immediate_disclosure is not None:
            self._disclosure = no_immediate_disclosure
        elif llm_client is not None:
            self._disclosure = NoImmediateDisclosure(llm_client=llm_client)
        else:
            raise ValueError(
                "Either 'no_immediate_disclosure' instance or 'llm_client' must be provided to PES."
            )

        if state is not None:
            self._state = state
        elif llm_client is not None:
            self._state = State(llm_client=llm_client)
        else:
            raise ValueError(
                "Either 'state' instance or 'llm_client' must be provided to PES."
            )

        if faithfulness is not None:
            self._faithfulness = faithfulness
        elif llm_client is not None:
            self._faithfulness = Faithfulness(llm_client=llm_client)
        else:
            self._faithfulness = None

        self._faithfulness_threshold = faithfulness_threshold
        self._state_weights = state_weights or _STATE_WEIGHTS
        self._default_weights = default_weights or _DEFAULT_WEIGHTS

    @property
    def name(self) -> str:
        return "pes"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_weights(self, state_code: str) -> tuple[float, float, float]:
        """Return ``(w1, w2, w3)`` for the given state code."""
        return self._state_weights.get(state_code, self._default_weights)

    @staticmethod
    def _weighted_geometric_mean(
        m1: float,
        m2: float,
        m3: float,
        w1: float,
        w2: float,
        w3: float,
    ) -> float:
        """Compute the weighted geometric mean of three scores.

        All inputs are clamped to [ε, 1.0] to avoid domain errors in ``log``.
        """
        m1_c = max(_EPSILON, min(1.0, m1))
        m2_c = max(_EPSILON, min(1.0, m2))
        m3_c = max(_EPSILON, min(1.0, m3))

        weight_sum = w1 + w2 + w3
        if weight_sum == 0:
            return 0.0

        log_sum = w1 * math.log(m1_c) + w2 * math.log(m2_c) + w3 * math.log(m3_c)
        return math.exp(log_sum / weight_sum)

    # ------------------------------------------------------------------
    # Public API (BaseMetric interface)
    # ------------------------------------------------------------------

    async def score(self, sample: TestSample) -> EvalResult:
        """Score a single sample with the composite PES metric.

        Execution order:

        1. **Faithfulness gate** (if provided) — short-circuits to 0.0 on
           failure.
        2. **State classifier** — determines dynamic weight configuration.
        3. **M1, M2, M3** — run in parallel via ``asyncio.gather``.
        4. **Weighted geometric mean** — final PES score.
        """
        if not sample.answer:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No answer provided.",
            )

        metadata: dict[str, Any] = {}

        # ---- Level 0: Faithfulness gate ----
        faithfulness_score: float | None = None
        gate_failed = False

        if self._faithfulness is not None:
            faith_result = await self._faithfulness.score(sample)
            faithfulness_score = faith_result.score
            metadata["faithfulness"] = {
                "score": round(faithfulness_score, 4),
                "threshold": self._faithfulness_threshold,
                "metadata": faith_result.metadata,
            }

            if faithfulness_score < self._faithfulness_threshold:
                gate_failed = True
                metadata["gate_failed"] = True
                return EvalResult(
                    metric_name=self.name,
                    score=0.0,
                    reason=(
                        f"Faithfulness gate FAILED "
                        f"(score={faithfulness_score:.4f} < "
                        f"threshold={self._faithfulness_threshold}). "
                        f"PES forced to 0.0."
                    ),
                    metadata=metadata,
                )

        metadata["gate_failed"] = False

        # ---- Level 1: State classification ----
        state_result = await self._state.score(sample)
        state_code = state_result.metadata.get("state", "")
        state_label = state_result.metadata.get("state_label", "")
        w1, w2, w3 = self._resolve_weights(state_code)

        metadata["state"] = {
            "code": state_code,
            "label": state_label,
            "confidence": state_result.metadata.get("confidence", 0.0),
        }
        metadata["weights"] = {"w1_uptake": w1, "w2_linguistic": w2, "w3_disclosure": w3}

        # ---- Level 2: Sub-metrics (parallel) ----
        m1_result, m2_result, m3_result = await asyncio.gather(
            self._uptake.score(sample),
            self._linguistic.score(sample),
            self._disclosure.score(sample),
        )

        m1 = m1_result.score
        m2 = m2_result.score
        m3 = m3_result.score

        metadata["M1_uptake"] = {"score": round(m1, 4), "metadata": m1_result.metadata}
        metadata["M2_linguistic_adaptation"] = {"score": round(m2, 4), "metadata": m2_result.metadata}
        metadata["M3_no_immediate_disclosure"] = {"score": round(m3, 4), "metadata": m3_result.metadata}

        # ---- Level 3: Weighted Geometric Mean ----
        pes_score = self._weighted_geometric_mean(m1, m2, m3, w1, w2, w3)
        pes_score = max(0.0, min(1.0, round(pes_score, 4)))

        # Build a human-readable reason string.
        reason_parts = [
            f"State={state_code or '?'} ({state_label or 'unknown'})",
            f"weights=({w1:.2f}, {w2:.2f}, {w3:.2f})",
            f"M1={m1:.4f}, M2={m2:.4f}, M3={m3:.4f}",
        ]
        if faithfulness_score is not None:
            reason_parts.append(f"faithfulness={faithfulness_score:.4f} (gate PASSED)")
        reason = " | ".join(reason_parts)

        return EvalResult(
            metric_name=self.name,
            score=pes_score,
            reason=reason,
            metadata=metadata,
        )
