#!/usr/bin/env python3
"""Run the PES evaluation pipeline on MathDial teacher turns.

Usage
-----
    # Quick test on 50 samples
    python run_mathdial_evaluation.py --limit 50

    # Full test set (599 conversations → ~3k teacher turns)
    python run_mathdial_evaluation.py --split test

    # Custom model and threshold
    python run_mathdial_evaluation.py --model llama3.2 --faithfulness-threshold 0.25 --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from rag_eval.core.config import LLMConfig
from rag_eval.core.types import EvalReport, EvalResult, TestSample
from rag_eval.datasets.mathdial_loader import load_mathdial, MOVE_SCORES
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
        description="Evaluate MathDial teacher turns with the PES metric."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Which MathDial split to use.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/mathdial",
        help="Directory containing MathDial JSONL files.",
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
        help="Max concurrent LLM calls.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/mathdial",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=0.25,
        help="Minimum faithfulness score for the gate (default: 0.25).",
    )
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Disable the faithfulness gate entirely.",
    )
    return parser.parse_args()


def build_results_dataframe(
    report: EvalReport,
    annotations: list[dict],
    samples: list[TestSample],
) -> pd.DataFrame:
    """Merge PES results with MathDial annotations into a single DataFrame."""
    rows = []

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
            "qid": annot["qid"],
            "teacher_move": annot["teacher_move"],
            "move_score": annot["move_score"],
            "self_correctness": annot["self_correctness"],
            "turn_index": annot["turn_index"],
            "pes_score": pes_result.score,
            "M1_uptake": m1_data.get("score", None),
            "M2_linguistic": m2_data.get("score", None),
            "M3_disclosure": m3_data.get("score", None),
            "state_code": state_data.get("code", ""),
            "state_label": state_data.get("label", ""),
            "gate_failed": meta.get("gate_failed", False),
            "response_text": sample.answer[:200],
            "reason": pes_result.reason,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step A: Load MathDial ─────────────────────────────────────────
    dataset_path = f"{args.data_dir}/{args.split}.jsonl"
    logger.info("Loading MathDial (%s) from %s ...", args.split, dataset_path)
    samples, annotations = load_mathdial(dataset_path, limit=args.limit)
    logger.info("Loaded %d teacher-turn samples.", len(samples))

    # Print teacher move distribution
    move_dist = Counter(a["teacher_move"] for a in annotations)
    logger.info("Teacher move distribution: %s", dict(move_dist))

    # ── Step B: Run PES evaluation ────────────────────────────────────
    logger.info("Initializing LLM client: provider=%s, model=%s", args.provider, args.model)
    llm_config = LLMConfig(provider=args.provider, model=args.model, temperature=0.0)
    llm_client = LLMClient(config=llm_config, enable_cache=True)

    if args.no_faithfulness:
        pes_metric = PES(llm_client=llm_client)
        pes_metric._faithfulness = None
    else:
        pes_metric = PES(
            llm_client=llm_client,
            faithfulness_threshold=args.faithfulness_threshold,
        )

    logger.info(
        "PES metric initialized (faithfulness_gate=%s, threshold=%.2f).",
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
    raw_path = output_dir / "mathdial_pes_results.json"
    report_data = {
        "metadata": {
            "dataset": "mathdial",
            "split": args.split,
            "model": args.model,
            "provider": args.provider,
            "num_samples": len(samples),
            "elapsed_seconds": round(elapsed, 1),
            "faithfulness_threshold": args.faithfulness_threshold,
        },
        "report": report.model_dump(mode="json"),
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info("Raw results saved to %s", raw_path)

    # ── Build and save DataFrame ──────────────────────────────────────
    df = build_results_dataframe(report, annotations, samples)
    csv_path = output_dir / "mathdial_pes_dataframe.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("DataFrame saved to %s (%d rows, %d cols)", csv_path, len(df), len(df.columns))

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MathDial PES Evaluation — Summary")
    print("=" * 70)
    print(f"  Samples evaluated : {len(df)}")
    print(f"  PES mean          : {df['pes_score'].mean():.4f}")
    print(f"  PES median        : {df['pes_score'].median():.4f}")
    print(f"  PES std           : {df['pes_score'].std():.4f}")
    print(f"  Gate failures     : {df['gate_failed'].sum()} ({df['gate_failed'].mean()*100:.1f}%)")
    print()

    # PES by teacher move — the key validation table
    print("  PES by Teacher Move (Pedagogical Quality Ordering):")
    print("  " + "-" * 60)
    move_order = ["probing", "focus", "generic", "telling"]
    for move in move_order:
        subset = df[df["teacher_move"] == move]
        if len(subset) > 0:
            gt_score = MOVE_SCORES.get(move, "?")
            print(
                f"    {move:10s}  (GT={gt_score:.2f})  "
                f"n={len(subset):4d}  "
                f"PES_mean={subset['pes_score'].mean():.4f}  "
                f"PES_median={subset['pes_score'].median():.4f}  "
                f"std={subset['pes_score'].std():.4f}"
            )
    print()

    # Quick Spearman correlation
    from scipy import stats as sp_stats
    rho, p = sp_stats.spearmanr(df["pes_score"], df["move_score"])
    print(f"  Spearman ρ (PES vs move_score): {rho:+.4f}  (p = {p:.2e})")

    # M3 specifically vs move_score (expected strongest signal)
    m3_valid = df[["M3_disclosure", "move_score"]].dropna()
    if len(m3_valid) > 2:
        rho_m3, p_m3 = sp_stats.spearmanr(m3_valid["M3_disclosure"], m3_valid["move_score"])
        print(f"  Spearman ρ (M3 vs move_score):  {rho_m3:+.4f}  (p = {p_m3:.2e})")

    print("=" * 70)


if __name__ == "__main__":
    main()
