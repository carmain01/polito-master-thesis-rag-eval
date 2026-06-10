"""Dataset loader — load evaluation datasets from various formats."""

from __future__ import annotations

from pathlib import Path

from rag_eval.core.types import TestSample


def load_dataset(path: str | Path) -> list[TestSample]:
    """Load a dataset from a JSON or JSONL file.

    Expected format (JSON):
        [
            {
                "question": "...",
                "answer": "...",
                "ground_truth": "...",
                "contexts": ["...", "..."]
            },
            ...
        ]
    """
    import json

    path = Path(path)
    samples: list[TestSample] = []

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(TestSample(**data))
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        samples = [TestSample(**item) for item in data]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .jsonl.")

    return samples
