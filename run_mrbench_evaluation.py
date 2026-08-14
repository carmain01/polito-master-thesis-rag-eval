#!/usr/bin/env python3
"""Run the PES evaluation pipeline on the MRBench v3 devset.

Usage
-----
    # Quick test on 50 samples
    python run_mrbench_evaluation.py --limit 50

    # Full dataset (2476 samples — takes 1-2 hours on Ollama)
    python run_mrbench_evaluation.py

    # Custom model
    python run_mrbench_evaluation.py --model llama3.1 --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from rag_eval.core.config import LLMConfig
from rag_eval.core.types import EvalReport, EvalResult, TestSample
from rag_eval.datasets.mrbench_loader import load_mrbench
from rag_eval.metrics.pes import PES
from rag_eval.pipeline.evaluator import Evaluator
from rag_eval.utils.llm import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MRBench v3 tutor responses with the PES metric."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mrbench_v3_devset.json",
        help="Path to the MRBench v3 JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples (for quick testing).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="Ollama model to use as LLM judge.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        help="LLM provider (ollama, openai, anthropic, google).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Max concurrent LLM calls (keep low for Ollama).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=0.25,
        help="Minimum faithfulness score required for the gate to pass (default: 0.25).",
    )
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Disable the faithfulness gate to speed up evaluation.",
    )
    return parser.parse_args()


def build_results_dataframe(
    report: EvalReport,
    annotations: list[dict],
    samples: list[TestSample],
) -> pd.DataFrame:
    """Merge PES results with MRBench annotations into a single DataFrame.

    Returns a DataFrame with columns:
        conversation_id, model_name, pes_score, M1_uptake, M2_linguistic,
        M3_disclosure, state_code, providing_guidance_label,
        actionability_label, providing_guidance, actionability, combined_score
    """
    rows = []

    # Build a mapping from sample_index → PES result
    pes_by_index: dict[int, EvalResult] = {}
    for res in report.results:
        if res.metric_name == "pes":
            pes_by_index[res.sample_index] = res

    for idx, (sample, annot) in enumerate(zip(samples, annotations)):
        pes_result = pes_by_index.get(idx)
        if pes_result is None:
            logger.warning("No PES result for sample %d, skipping.", idx)
            continue

        meta = pes_result.metadata or {}
        m1_data = meta.get("M1_uptake", {})
        m2_data = meta.get("M2_linguistic_adaptation", {})
        m3_data = meta.get("M3_no_immediate_disclosure", {})
        state_data = meta.get("state", {})

        row = {
            "sample_index": idx,
            "conversation_id": annot["conversation_id"],
            "model_name": annot["model_name"],
            "pes_score": pes_result.score,
            "M1_uptake": m1_data.get("score", None),
            "M2_linguistic": m2_data.get("score", None),
            "M3_disclosure": m3_data.get("score", None),
            "state_code": state_data.get("code", ""),
            "state_label": state_data.get("label", ""),
            "gate_failed": meta.get("gate_failed", False),
            "providing_guidance_label": annot["providing_guidance_label"],
            "actionability_label": annot["actionability_label"],
            "providing_guidance": annot["providing_guidance"],
            "actionability": annot["actionability"],
            "combined_score": annot["combined_score"],
            "response_text": sample.answer[:200],  # truncated for CSV
            "reason": pes_result.reason,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step A: Load MRBench ──────────────────────────────────────────
    logger.info("Loading MRBench dataset from %s ...", args.dataset)
    samples, annotations = load_mrbench(args.dataset, limit=args.limit)
    logger.info("Loaded %d samples.", len(samples))

    # Print ground truth distribution
    from collections import Counter
    pg_dist = Counter(a["providing_guidance_label"] for a in annotations)
    act_dist = Counter(a["actionability_label"] for a in annotations)
    logger.info("Providing_Guidance distribution: %s", dict(pg_dist))
    logger.info("Actionability distribution:      %s", dict(act_dist))

    # ── Step B: Run PES evaluation ────────────────────────────────────
    logger.info("Initializing LLM client: provider=%s, model=%s", args.provider, args.model)
    llm_config = LLMConfig(provider=args.provider, model=args.model, temperature=0.0)
    llm_client = LLMClient(config=llm_config, enable_cache=True)

    if args.no_faithfulness:
        # Skip faithfulness to save LLM calls
        pes_metric = PES(llm_client=llm_client)
        pes_metric._faithfulness = None
    else:
        pes_metric = PES(
            llm_client=llm_client,
            faithfulness_threshold=args.faithfulness_threshold,
        )

    logger.info(
        "PES metric initialized (faithfulness_gate=%s, threshold=%.2f). Sub-metrics: Uptake, LinguisticAdaptation, NoImmediateDisclosure, State",
        "disabled" if args.no_faithfulness else "enabled",
        args.faithfulness_threshold,
    )

    evaluator = Evaluator(
        metrics=[pes_metric],
        cache_dir=str(output_dir / ".pes_cache"),
    )

    start_time = time.time()
    logger.info(
        "Starting PES evaluation on %d samples (batch_size=%d, max_concurrency=%d)...",
        len(samples), args.batch_size, args.max_concurrency,
    )

    report = evaluator.evaluate(
        samples,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
    )

    elapsed = time.time() - start_time
    logger.info("Evaluation completed in %.1f seconds (%.1f min).", elapsed, elapsed / 60)

    # ── Save raw results ──────────────────────────────────────────────
    raw_path = output_dir / "mrbench_pes_results.json"
    report_data = {
        "metadata": {
            "dataset": args.dataset,
            "model": args.model,
            "provider": args.provider,
            "num_samples": len(samples),
            "elapsed_seconds": round(elapsed, 1),
        },
        "report": report.model_dump(mode="json"),
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info("Raw results saved to %s", raw_path)

    # ── Build and save DataFrame ──────────────────────────────────────
    df = build_results_dataframe(report, annotations, samples)
    csv_path = output_dir / "mrbench_pes_dataframe.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("DataFrame saved to %s (%d rows, %d cols)", csv_path, len(df), len(df.columns))

    # ── Quick summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MRBench PES Evaluation — Summary")
    print("=" * 70)
    print(f"  Samples evaluated : {len(df)}")
    print(f"  PES mean          : {df['pes_score'].mean():.4f}")
    print(f"  PES median        : {df['pes_score'].median():.4f}")
    print(f"  PES std           : {df['pes_score'].std():.4f}")
    print()

    # PES by label group
    print("  PES by Providing_Guidance label:")
    for label in ["Yes", "To some extent", "No"]:
        subset = df[df["providing_guidance_label"] == label]
        if len(subset) > 0:
            print(f"    {label:20s}  n={len(subset):4d}  mean={subset['pes_score'].mean():.4f}  std={subset['pes_score'].std():.4f}")
    print()

    # PES by model
    print("  PES by Model:")
    for model, group in df.groupby("model_name"):
        print(f"    {model:15s}  n={len(group):4d}  mean={group['pes_score'].mean():.4f}")
    print("=" * 70)

    # Token/cost usage
    if hasattr(llm_client, "usage"):
        u = llm_client.usage
        print(f"\n  LLM Usage: {u.total_input_tokens} input + {u.total_output_tokens} output tokens")
        print(f"  Estimated cost: ${u.total_cost:.4f}")


if __name__ == "__main__":
    main()
