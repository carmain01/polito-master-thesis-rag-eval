"""Load the MathDial dataset and convert it to TestSample objects for the PES pipeline.

MathDial (EMNLP 2023) contains ~2,861 one-to-one math tutoring dialogues, each with
teacher moves annotated as: *generic*, *focus*, *probing*, *telling*.

For PES validation, we extract **individual teacher turns** so that each sample
represents a single tutor response with its pedagogical label (teacher move).

Ground truth mapping for PES correlation
-----------------------------------------
The teacher moves form a natural pedagogical quality ordering:

    probing  (best scaffolding)   → 1.0
    focus    (good redirection)   → 0.75
    generic  (neutral opening)    → 0.5
    telling  (reveals answer)     → 0.0

This ordering directly aligns with PES's M3 (No-Immediate-Disclosure) sub-metric:
- ``probing`` and ``focus`` imply the tutor guides without spoiling
- ``telling`` means the tutor reveals the solution, which PES should penalize

Usage
-----
    from rag_eval.datasets.mathdial_loader import load_mathdial

    samples, annotations = load_mathdial("data/mathdial/test.jsonl", limit=100)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rag_eval.core.types import TestSample

logger = logging.getLogger(__name__)

# ── Ground truth mapping ─────────────────────────────────────────────────
# Maps teacher move labels to a numeric pedagogical quality score.
# Higher = better scaffolding; lower = more direct telling.
MOVE_SCORES: dict[str, float] = {
    "probing": 1.0,    # Socratic questioning — best scaffolding
    "focus": 0.75,     # Redirecting attention — good scaffolding
    "generic": 0.5,    # Neutral opening / generic prompt
    "telling": 0.0,    # Revealing the answer — worst pedagogically
}


def _parse_conversation(raw_conv: str) -> list[dict[str, str]]:
    """Parse MathDial ``|EOM|``-delimited conversation into structured turns.

    Returns a list of dicts with keys: ``role`` (Teacher/Student),
    ``move`` (teacher move tag or None), ``text`` (utterance content).
    """
    turns = []
    for segment in raw_conv.split("|EOM|"):
        segment = segment.strip()
        if not segment:
            continue

        if segment.startswith("Teacher:"):
            body = segment[len("Teacher:"):].strip()
            match = re.match(r"\((\w+)\)\s*(.*)", body, re.DOTALL)
            if match:
                move = match.group(1).lower()
                text = match.group(2).strip()
            else:
                move = None
                text = body
            turns.append({"role": "Teacher", "move": move, "text": text})

        elif segment.startswith("Student:"):
            text = segment[len("Student:"):].strip()
            turns.append({"role": "Student", "move": None, "text": text})

    return turns


def load_mathdial(
    path: str,
    limit: int | None = None,
    min_teacher_text_len: int = 20,
) -> tuple[list[TestSample], list[dict]]:
    """Load MathDial and produce one ``TestSample`` per teacher turn.

    Parameters
    ----------
    path : str
        Path to ``test.jsonl`` or ``train.jsonl``.
    limit : int, optional
        Maximum number of samples to return.
    min_teacher_text_len : int
        Skip teacher turns shorter than this (filters out trivial "Hi" turns).

    Returns
    -------
    samples : list[TestSample]
        Each sample has:
        - ``question``: The conversation history up to the student's last turn
        - ``answer``: The teacher's response (the turn being evaluated)
        - ``contexts``: [math problem text]
    annotations : list[dict]
        Parallel list with ground truth for each sample:
        - ``teacher_move``: raw label (probing/focus/generic/telling)
        - ``move_score``: numeric score (0.0–1.0)
        - ``qid``: problem identifier
        - ``self_correctness``: whether the student eventually self-corrected
        - ``conversation_id``: unique ID for this sample
        - ``turn_index``: position of this teacher turn in the dialogue
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"MathDial file not found: {filepath}")

    samples: list[TestSample] = []
    annotations: list[dict] = []
    count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            qid = entry.get("qid", str(line_num))
            question_text = entry.get("question", "")
            ground_truth_solution = entry.get("ground_truth", "")
            student_incorrect = entry.get("student_incorrect_solution", "")
            self_correctness = entry.get("self-correctness", "N/A")

            # Parse the conversation
            turns = _parse_conversation(entry.get("conversation", ""))

            # Extract individual teacher turns (skip first generic "Hi" turn)
            history_parts: list[str] = []

            for turn_idx, turn in enumerate(turns):
                if turn["role"] == "Student":
                    history_parts.append(f"Student: {turn['text']}")
                    continue

                # Teacher turn
                move = turn.get("move")
                text = turn["text"]

                # Skip very short turns (e.g., "Hi, talk me through your solution")
                if len(text) < min_teacher_text_len:
                    history_parts.append(f"Teacher: {text}")
                    continue

                # Skip turns without a recognized move tag
                if move not in MOVE_SCORES:
                    history_parts.append(f"Teacher: {text}")
                    continue

                # Build the conversation history (everything before this turn)
                conv_history = "\n".join(history_parts)

                # Build context: the math problem + student's incorrect solution
                context = (
                    f"Math Problem: {question_text}\n\n"
                    f"Correct Solution: {ground_truth_solution}\n\n"
                    f"Student's Incorrect Solution: {student_incorrect}"
                )

                sample = TestSample(
                    question=conv_history if conv_history else question_text,
                    answer=text,
                    contexts=[context],
                    metadata={
                        "conversation_id": f"mathdial-{qid}-t{turn_idx}",
                        "dataset": "mathdial",
                    },
                )

                annotation = {
                    "conversation_id": f"mathdial-{qid}-t{turn_idx}",
                    "qid": qid,
                    "teacher_move": move,
                    "move_score": MOVE_SCORES[move],
                    "self_correctness": self_correctness,
                    "turn_index": turn_idx,
                    "total_turns": len(turns),
                    "teacher_text": text[:200],
                }

                samples.append(sample)
                annotations.append(annotation)
                count += 1

                if limit is not None and count >= limit:
                    logger.info("Reached limit of %d samples.", limit)
                    return samples, annotations

                # Add this teacher turn to history for subsequent turns
                history_parts.append(f"Teacher: {text}")

    logger.info(
        "Loaded %d teacher-turn samples from MathDial (%s).",
        len(samples),
        filepath.name,
    )
    return samples, annotations
