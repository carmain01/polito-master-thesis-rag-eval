"""MRBench v3 loader — parse the MRBench devset into TestSample objects.

Transforms the nested MRBench JSON structure (conversations → tutor_responses)
into a flat list of :class:`TestSample` objects suitable for the PES pipeline,
together with a parallel list of annotation dictionaries for downstream
statistical validation.

Ground Truth Encoding
---------------------
Two numeric ground truth scores are computed per response:

- **providing_guidance** (primary):
    - ``"Yes"`` → 1.0
    - ``"To some extent"`` → 0.5
    - ``"No"`` → 0.0

- **actionability** (secondary):
    Same encoding as above.

- **combined_score** (weighted average):
    ``0.6 × providing_guidance + 0.4 × actionability``

    The 60/40 split reflects the fact that *Providing_Guidance* directly
    measures the pedagogical quality that PES is designed to capture, while
    *Actionability* adds a complementary "concreteness" dimension.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from rag_eval.core.types import TestSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label → numeric mapping
# ---------------------------------------------------------------------------
_LABEL_MAP: dict[str, float] = {
    "Yes": 1.0,
    "To some extent": 0.5,
    "No": 0.0,
}

# Weights for the combined ground truth score.
_PROVIDING_GUIDANCE_WEIGHT: float = 0.6
_ACTIONABILITY_WEIGHT: float = 0.4


def _label_to_numeric(label: str) -> float:
    """Convert a textual MRBench annotation label to a numeric score."""
    return _LABEL_MAP.get(label, 0.0)


def _extract_math_problem(conversation_history: str) -> str:
    """Extract the mathematical problem from the first Tutor turn.

    The first Tutor message in MRBench typically follows the pattern:
        "Tutor: Hi, could you please provide a step-by-step solution for
         the question below? The question is: <PROBLEM> \\n Student: ..."

    We extract everything between "The question is:" and the first
    "\\n Student:" (or end of first Tutor turn).
    """
    # Try to extract the math problem from "The question is: ... \n Student:"
    match = re.search(
        r"The question is:\s*(.+?)(?:\\n\s*Student:|$)",
        conversation_history,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: take the first Tutor turn entirely
    match = re.search(
        r"Tutor:\s*(.+?)(?:\\n\s*Student:|$)",
        conversation_history,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Last resort: return the first 500 chars
    return conversation_history[:500].strip()


def load_mrbench(
    path: str | Path,
    limit: int | None = None,
) -> tuple[list[TestSample], list[dict[str, Any]]]:
    """Load the MRBench v3 devset and flatten it into evaluation samples.

    Parameters
    ----------
    path : str | Path
        Path to ``mrbench_v3_devset.json``.
    limit : int | None
        If set, only the first *limit* samples are returned (useful for
        quick testing).

    Returns
    -------
    samples : list[TestSample]
        One sample per (conversation, model) pair.
    annotations : list[dict]
        Parallel list of annotation metadata, one per sample, containing:
        ``conversation_id``, ``model_name``, ``annotation`` (raw),
        ``providing_guidance``, ``actionability``, ``combined_score``,
        ``providing_guidance_label``, ``actionability_label``.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples: list[TestSample] = []
    annotations: list[dict[str, Any]] = []

    for conv in data:
        conversation_id = conv.get("conversation_id", "")
        conversation_history = conv.get("conversation_history", "")
        tutor_responses = conv.get("tutor_responses", {})

        # Extract the math problem as context
        math_problem = _extract_math_problem(conversation_history)

        for model_name, model_data in tutor_responses.items():
            response_text = model_data.get("response", "")
            annotation = model_data.get("annotation", {})

            # Compute numeric ground truth
            pg_label = annotation.get("Providing_Guidance", "No")
            act_label = annotation.get("Actionability", "No")

            pg_score = _label_to_numeric(pg_label)
            act_score = _label_to_numeric(act_label)
            combined = (
                _PROVIDING_GUIDANCE_WEIGHT * pg_score
                + _ACTIONABILITY_WEIGHT * act_score
            )

            sample = TestSample(
                question=conversation_history,
                answer=response_text,
                contexts=[math_problem],
                metadata={
                    "conversation_id": conversation_id,
                    "model_name": model_name,
                },
            )

            annot_record = {
                "conversation_id": conversation_id,
                "model_name": model_name,
                "annotation": annotation,
                "providing_guidance_label": pg_label,
                "actionability_label": act_label,
                "providing_guidance": pg_score,
                "actionability": act_score,
                "combined_score": round(combined, 2),
            }

            samples.append(sample)
            annotations.append(annot_record)

            if limit is not None and len(samples) >= limit:
                logger.info("Reached limit of %d samples.", limit)
                return samples, annotations

    logger.info(
        "Loaded %d samples from %d conversations (%s).",
        len(samples),
        len(data),
        path.name,
    )
    return samples, annotations
