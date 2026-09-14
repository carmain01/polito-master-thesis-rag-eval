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
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Starting sample index (e.g. 300 to skip the first 300).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Automatically resume from the last evaluated sample index found in output-dir.",
    )
    return parser.parse_args()


def build_results_dataframe(
    report: EvalReport,
    annotations: list[dict],
    samples: list[TestSample],
    start_index: int = 0,
) -> pd.DataFrame:
    """Merge PES results with MathDial annotations into a single DataFrame."""
    rows = []

    pes_by_index: dict[int, EvalResult] = {}
    for res in report.results:
        if res.metric_name == "pes":
            pes_by_index[res.sample_index] = res

    for local_idx, (sample, annot) in enumerate(zip(samples, annotations)):
        global_idx = start_index + local_idx
        pes_result = pes_by_index.get(global_idx)
        if pes_result is None:
            logger.warning("No PES result for sample %d, skipping.", global_idx)
            continue

        meta = pes_result.metadata or {}
        m1_data = meta.get("M1_uptake", {})
        m2_data = meta.get("M2_linguistic_adaptation", {})
        m3_data = meta.get("M3_no_immediate_disclosure", {})
        state_data = meta.get("state", {})
        faith_data = meta.get("faithfulness", {})

        row = {
            "sample_index": global_idx,
            "conversation_id": annot["conversation_id"],
            "qid": annot["qid"],
            "teacher_move": annot["teacher_move"],
            "move_score": annot["move_score"],
            "self_correctness": annot["self_correctness"],
            "turn_index": annot["turn_index"],
            "pes_score": pes_result.score,
            "pes_before_gate": meta.get("pes_before_gate", pes_result.score),
            "faithfulness_score": faith_data.get("score", None),
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
    csv_path = output_dir / "mathdial_pes_dataframe.csv"
    raw_path = output_dir / "mathdial_pes_results.json"

    # ── Step A: Load MathDial ─────────────────────────────────────────
    dataset_path = f"{args.data_dir}/{args.split}.jsonl"
    logger.info("Loading MathDial (%s) from %s ...", args.split, dataset_path)
    all_samples, all_annotations = load_mathdial(dataset_path)
    logger.info("Loaded %d total teacher-turn samples from dataset.", len(all_samples))

    # Determine offset & resume logic
    offset = 0
    existing_df = None
    if args.resume:
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                if not existing_df.empty and "sample_index" in existing_df.columns:
                    max_idx = int(existing_df["sample_index"].max())
                    offset = max_idx + 1
                    logger.info(
                        "Found %d existing results in %s (max sample_index=%d). Resuming from offset %d.",
                        len(existing_df), csv_path, max_idx, offset,
                    )
                else:
                    logger.warning("Existing CSV is empty or has no sample_index; starting from 0.")
            except Exception as e:
                logger.warning("Could not read %s for resume: %s. Starting from 0.", csv_path, e)
        else:
            logger.info("--resume passed but %s does not exist. Starting from 0.", csv_path)
    elif args.offset is not None:
        offset = args.offset
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
            except Exception:
                pass

    if offset >= len(all_samples):
        logger.info("Offset (%d) >= total samples (%d). Evaluation is already complete!", offset, len(all_samples))
        return

    samples = all_samples[offset:]
    annotations = all_annotations[offset:]

    if args.limit is not None:
        samples = samples[: args.limit]
        annotations = annotations[: args.limit]

    logger.info("Selected %d samples for evaluation (offset=%d to %d).", len(samples), offset, offset + len(samples) - 1)

    # Print teacher move distribution for selected slice
    move_dist = Counter(a["teacher_move"] for a in annotations)
    logger.info("Teacher move distribution (current run): %s", dict(move_dist))

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

    def on_batch_complete(current_report: EvalReport, batch_num: int) -> None:
        try:
            curr_df = build_results_dataframe(current_report, annotations, samples, start_index=offset)
            if existing_df is not None and not existing_df.empty:
                full_df = pd.concat([existing_df, curr_df], ignore_index=True)
                full_df = full_df.drop_duplicates(subset=["sample_index"], keep="last").sort_values("sample_index").reset_index(drop=True)
            else:
                full_df = curr_df
            full_df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info("Incremental checkpoint batch %d saved to %s (%d rows)", batch_num, csv_path, len(full_df))
        except Exception as e:
            logger.warning("Failed to incrementally update CSV at batch %d: %s", batch_num, e)

    evaluator = Evaluator(
        metrics=[pes_metric],
        cache_dir=str(output_dir / ".pes_cache"),
        on_batch_complete=on_batch_complete,
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
        start_index=offset,
    )

    elapsed = time.time() - start_time
    logger.info("Evaluation run completed in %.1f seconds (%.1f min).", elapsed, elapsed / 60)

    # ── Build and merge DataFrame ──────────────────────────────────────
    new_df = build_results_dataframe(report, annotations, samples, start_index=offset)
    if existing_df is not None and not existing_df.empty and (args.resume or offset > 0):
        df = pd.concat([existing_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["sample_index"], keep="last").sort_values("sample_index").reset_index(drop=True)
        logger.info(
            "Merged %d new rows with %d existing rows -> %d total rows in DataFrame.",
            len(new_df), len(existing_df), len(df),
        )
    else:
        df = new_df

    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("DataFrame saved to %s (%d rows, %d cols)", csv_path, len(df), len(df.columns))

    # ── Save raw results ──────────────────────────────────────────────
    merged_report = report.model_dump(mode="json")
    prev_elapsed = 0.0
    if (args.resume or offset > 0) and raw_path.exists():
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                old_raw = json.load(f)
            old_results = old_raw.get("report", {}).get("results", [])
            prev_elapsed = float(old_raw.get("metadata", {}).get("elapsed_seconds", 0.0))
            new_indices = {res.sample_index for res in report.results}
            filtered_old = [r for r in old_results if r.get("sample_index") not in new_indices]
            all_raw_results = filtered_old + merged_report.get("results", [])
            all_raw_results.sort(key=lambda r: r.get("sample_index", 0))
            merged_report["results"] = all_raw_results
        except Exception as e:
            logger.warning("Could not merge with existing raw results: %s", e)

    report_data = {
        "metadata": {
            "dataset": "mathdial",
            "split": args.split,
            "model": args.model,
            "provider": args.provider,
            "num_samples": len(df),
            "elapsed_seconds": round(prev_elapsed + elapsed, 1),
            "faithfulness_threshold": args.faithfulness_threshold,
        },
        "report": merged_report,
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info("Raw results saved to %s", raw_path)
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
