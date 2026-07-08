"""Dataset loader — load evaluation datasets from various formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_eval.core.types import TestSample


def load_dataset(
    path: str | Path,
    format: str | None = None,
    column_mapping: dict[str, str] | None = None,
) -> list[TestSample]:
    """Load a dataset from a file or HuggingFace datasets.

    Args:
        path: Path to local file, or HuggingFace dataset name.
        format: 'json', 'jsonl', 'csv', 'hf'. If None, inferred from extension.
        column_mapping: Optional mapping from source columns to TestSample fields
            e.g. {"input_text": "question", "target_text": "ground_truth"}
    """
    path_str = str(path)

    if format is None:
        if path_str.endswith(".json"):
            format = "json"
        elif path_str.endswith(".jsonl"):
            format = "jsonl"
        elif path_str.endswith(".csv"):
            format = "csv"
        else:
            format = "hf"  # default fallback if no extension

    samples: list[TestSample] = []
    column_mapping = column_mapping or {}

    if format == "jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                data = _apply_mapping(data, column_mapping)
                samples.append(TestSample(**data))
    elif format == "json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "data" in data:
                data = data["data"]  # some json formats wrap in data array
            for item in data:
                item = _apply_mapping(item, column_mapping)
                samples.append(TestSample(**item))
    elif format == "csv":
        import pandas as pd

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            data = row.to_dict()
            data = _apply_mapping(data, column_mapping)
            # Ensure contexts is a list if it's a string representation
            if "contexts" in data and isinstance(data["contexts"], str):
                try:
                    data["contexts"] = json.loads(data["contexts"])
                except Exception:
                    data["contexts"] = [data["contexts"]]
            samples.append(TestSample(**data))
    elif format == "hf":
        from datasets import load_dataset as hf_load_dataset

        # path is treated as HF dataset name
        dataset = hf_load_dataset(path_str, split="train")  # default to train split
        for item in dataset:
            item = _apply_mapping(item, column_mapping)
            samples.append(TestSample(**item))
    else:
        raise ValueError(f"Unsupported dataset format: {format}")

    validate_dataset(samples)
    return samples


def _apply_mapping(data: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    if not mapping:
        return data
    result = data.copy()
    for src_col, dest_col in mapping.items():
        if src_col in result:
            result[dest_col] = result.pop(src_col)
    return result


def validate_dataset(samples: list[TestSample]) -> None:
    """Validate the loaded dataset and log warnings for missing optional fields."""
    import logging

    if not samples:
        logging.warning("Dataset is empty.")
        return

    missing_answers = sum(1 for s in samples if not s.answer)
    missing_contexts = sum(1 for s in samples if not s.contexts)
    missing_gt = sum(1 for s in samples if not s.ground_truth)

    total = len(samples)
    if missing_answers > 0:
        logging.warning(
            f"Dataset validation: {missing_answers}/{total} samples are missing 'answer'."
        )
    if missing_contexts > 0:
        logging.warning(
            f"Dataset validation: {missing_contexts}/{total} samples are missing 'contexts'."
        )
    if missing_gt > 0:
        logging.warning(
            f"Dataset validation: {missing_gt}/{total} samples are missing 'ground_truth'."
        )


def dataset_statistics(samples: list[TestSample]) -> dict[str, Any]:
    """Compute basic statistics for a dataset."""
    if not samples:
        return {"num_samples": 0}

    avg_context_len = 0
    avg_contexts_count = 0

    for s in samples:
        avg_contexts_count += len(s.contexts)
        for ctx in s.contexts:
            avg_context_len += len(ctx)

    total_contexts = sum(len(s.contexts) for s in samples)

    return {
        "num_samples": len(samples),
        "avg_contexts_per_sample": avg_contexts_count / len(samples),
        "avg_context_length_chars": avg_context_len / total_contexts if total_contexts > 0 else 0,
        "missing_answers": sum(1 for s in samples if not s.answer),
        "missing_ground_truth": sum(1 for s in samples if not s.ground_truth),
    }
