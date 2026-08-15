#!/usr/bin/env python3
"""Ablation study: test different PES weight configurations post-hoc.

Reads the evaluation CSV (which contains M1, M2, M3 sub-scores) and
recomputes PES with different weight configurations WITHOUT re-running
any LLM calls. Then compares Spearman ρ correlation with human ground truth.

Usage
-----
    # MRBench ablation
    python ablation_pes_weights.py --csv output/mrbench_pes_dataframe.csv --gt-col combined_score

    # MathDial ablation
    python ablation_pes_weights.py --csv output/mathdial/mathdial_pes_dataframe.csv --gt-col move_score
"""

from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", font_scale=1.1)

_EPSILON = 1e-6


def weighted_geometric_mean(m1: float, m2: float, m3: float, w1: float, w2: float, w3: float) -> float:
    """Compute PES as a weighted geometric mean of M1, M2, M3."""
    m1 = max(m1, _EPSILON)
    m2 = max(m2, _EPSILON)
    m3 = max(m3, _EPSILON)
    w_sum = w1 + w2 + w3
    if w_sum == 0:
        return 0.0
    log_pes = (w1 * np.log(m1) + w2 * np.log(m2) + w3 * np.log(m3)) / w_sum
    return float(np.exp(log_pes))


def recompute_pes(df: pd.DataFrame, w1: float, w2: float, w3: float) -> pd.Series:
    """Recompute PES scores for all rows using given weights."""
    return df.apply(
        lambda row: (
            0.0 if row.get("gate_failed", False)
            else weighted_geometric_mean(
                row["M1_uptake"] if pd.notna(row["M1_uptake"]) else 0.5,
                row["M2_linguistic"] if pd.notna(row["M2_linguistic"]) else 0.5,
                row["M3_disclosure"] if pd.notna(row["M3_disclosure"]) else 0.5,
                w1, w2, w3,
            )
        ),
        axis=1,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study for PES weights.")
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV with M1, M2, M3 columns.")
    parser.add_argument("--gt-col", default="combined_score", help="Ground truth column name.")
    parser.add_argument("--output-dir", default=None, help="Output directory (auto-derived if not set).")
    parser.add_argument("--step", type=float, default=0.05, help="Weight grid step size.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    gt_col = args.gt_col

    if gt_col not in df.columns:
        logger.error("Ground truth column '%s' not found. Available: %s", gt_col, list(df.columns))
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.csv).parent / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out rows without sub-metric scores
    valid = df.dropna(subset=["M1_uptake", "M2_linguistic", "M3_disclosure", gt_col]).copy()
    logger.info("Using %d valid rows (of %d total) for ablation.", len(valid), len(df))

    # ── 1. Named configurations ──────────────────────────────────────
    named_configs = {
        "Default (State C)":     (0.15, 0.25, 0.60),
        "Default (State A)":     (0.40, 0.40, 0.20),
        "Default (State B)":     (0.20, 0.30, 0.50),
        "Equal weights":         (0.33, 0.33, 0.34),
        "M3-only":               (0.00, 0.00, 1.00),
        "M1-only":               (1.00, 0.00, 0.00),
        "M2-only":               (0.00, 1.00, 0.00),
        "No-M1":                 (0.00, 0.30, 0.70),
        "M3-heavy":              (0.05, 0.15, 0.80),
        "M2+M3 balanced":        (0.00, 0.50, 0.50),
        "M3-dominant":           (0.10, 0.10, 0.80),
        "Pedagogical-focused":   (0.05, 0.20, 0.75),
    }

    print("\n" + "═" * 90)
    print("  ABLATION STUDY: PES Weight Configurations")
    print(f"  Ground truth: {gt_col} | Valid samples: {len(valid)}")
    print("═" * 90)
    print(f"  {'Configuration':<25s}  {'w1(M1)':>6s}  {'w2(M2)':>6s}  {'w3(M3)':>6s}  {'ρ':>7s}  {'p-value':>10s}  {'sig':>4s}  {'PES_mean':>8s}")
    print("  " + "─" * 85)

    named_results = {}
    for name, (w1, w2, w3) in named_configs.items():
        pes_recomputed = recompute_pes(valid, w1, w2, w3)
        rho, p = stats.spearmanr(pes_recomputed, valid[gt_col])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        named_results[name] = {"w1": w1, "w2": w2, "w3": w3, "rho": rho, "p": p, "mean": pes_recomputed.mean()}
        print(f"  {name:<25s}  {w1:>6.2f}  {w2:>6.2f}  {w3:>6.2f}  {rho:>+7.4f}  {p:>10.2e}  {sig:>4s}  {pes_recomputed.mean():>8.4f}")

    # ── 2. Grid search ───────────────────────────────────────────────
    step = args.step
    weight_values = np.arange(0.0, 1.0 + step, step)
    best_rho = -1.0
    best_config = (0.33, 0.33, 0.34)
    grid_results = []

    for w1 in weight_values:
        for w2 in weight_values:
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < -0.001 or w3 > 1.001:
                continue
            w3 = max(0.0, w3)  # clamp floating point noise

            pes_recomputed = recompute_pes(valid, w1, w2, w3)
            rho, p = stats.spearmanr(pes_recomputed, valid[gt_col])

            grid_results.append({
                "w1_uptake": round(w1, 2),
                "w2_linguistic": round(w2, 2),
                "w3_disclosure": round(w3, 2),
                "rho": rho,
                "p_value": p,
                "pes_mean": pes_recomputed.mean(),
            })

            if rho > best_rho:
                best_rho = rho
                best_p = p
                best_config = (round(w1, 2), round(w2, 2), round(w3, 2))

    grid_df = pd.DataFrame(grid_results)

    print("\n" + "─" * 90)
    print(f"  🏆 BEST GRID SEARCH CONFIG:")
    print(f"     w1(M1)={best_config[0]:.2f}  w2(M2)={best_config[1]:.2f}  w3(M3)={best_config[2]:.2f}")
    print(f"     ρ = {best_rho:+.4f}  (p = {best_p:.2e})")

    # ── 3. Top 10 configurations ─────────────────────────────────────
    top10 = grid_df.nlargest(10, "rho")
    print("\n  Top 10 weight configurations (by Spearman ρ):")
    print(f"  {'#':>3s}  {'w1':>5s}  {'w2':>5s}  {'w3':>5s}  {'ρ':>7s}  {'p-value':>10s}")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"  {rank:>3d}  {row['w1_uptake']:>5.2f}  {row['w2_linguistic']:>5.2f}  {row['w3_disclosure']:>5.2f}  {row['rho']:>+7.4f}  {row['p_value']:>10.2e}")

    # ── 4. Save grid results ─────────────────────────────────────────
    grid_df.to_csv(output_dir / "weight_grid_search.csv", index=False)
    logger.info("Grid search results saved to %s", output_dir / "weight_grid_search.csv")

    # ── 5. Heatmap: ρ as function of w1 and w3 (w2 = 1 - w1 - w3) ──
    # Fix w2 at various levels and show w1 vs w3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, w2_fixed in enumerate([0.0, 0.15, 0.30]):
        sub = grid_df[np.isclose(grid_df["w2_linguistic"], w2_fixed, atol=step / 2)]
        if len(sub) < 3:
            continue
        pivot = sub.pivot_table(values="rho", index="w1_uptake", columns="w3_disclosure", aggfunc="first")
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            vmin=grid_df["rho"].min(), vmax=grid_df["rho"].max(),
            ax=axes[ax_idx], linewidths=0.5,
        )
        axes[ax_idx].set_title(f"w2(Linguistic) = {w2_fixed:.2f}", fontsize=12, fontweight="bold")
        axes[ax_idx].set_xlabel("w3 (Disclosure)")
        axes[ax_idx].set_ylabel("w1 (Uptake)")
        axes[ax_idx].invert_yaxis()

    fig.suptitle(f"Spearman ρ Heatmap: PES vs {gt_col}\n(Weight Grid Search)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_weight_search.png", dpi=200)
    plt.close(fig)
    logger.info("Saved weight search heatmap.")

    # ── 6. Bar chart: named configs comparison ───────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(named_results.keys())
    rhos = [named_results[n]["rho"] for n in names]
    colors = ["#27ae60" if r > 0 else "#e74c3c" for r in rhos]
    bars = ax.barh(names, rhos, color=colors, edgecolor="white")

    for bar, rho_val in zip(bars, rhos):
        x_pos = rho_val + 0.005 if rho_val >= 0 else rho_val - 0.005
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"ρ={rho_val:+.4f}", va="center", fontsize=9, fontweight="bold",
                ha="left" if rho_val >= 0 else "right")

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ", fontsize=13)
    ax.set_title(f"PES Correlation by Weight Configuration\n(vs {gt_col})", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "bar_named_configs.png", dpi=200)
    plt.close(fig)
    logger.info("Saved named configs comparison.")

    # ── 7. Recompute group stats with best weights ───────────────────
    pes_best = recompute_pes(valid, *best_config)
    pes_default = recompute_pes(valid, 0.15, 0.25, 0.60)  # State C default
    rho_default, p_default = stats.spearmanr(pes_default, valid[gt_col])

    print(f"\n{'═' * 90}")
    print(f"  CONFRONTO: Default vs Best")
    print(f"{'═' * 90}")
    print(f"  Default (0.15, 0.25, 0.60):  ρ = {rho_default:+.4f}  (p = {p_default:.2e})")
    print(f"  Best    {best_config}:  ρ = {best_rho:+.4f}  (p = {best_p:.2e})")
    print(f"  Miglioramento:               Δρ = {best_rho - rho_default:+.4f}")
    print(f"{'═' * 90}")

    # Check if providing_guidance_label or teacher_move is available for group analysis
    label_col = None
    if "providing_guidance_label" in valid.columns:
        label_col = "providing_guidance_label"
        label_order = ["Yes", "To some extent", "No"]
    elif "teacher_move" in valid.columns:
        label_col = "teacher_move"
        label_order = ["probing", "focus", "generic", "telling"]

    if label_col:
        print(f"\n  PES by {label_col} — Default vs Best weights:")
        valid_copy = valid.copy()
        valid_copy["pes_default"] = pes_default
        valid_copy["pes_best"] = pes_best
        for label in label_order:
            sub = valid_copy[valid_copy[label_col] == label]
            if len(sub) > 0:
                print(f"    {label:20s}  Default={sub['pes_default'].mean():.4f}  Best={sub['pes_best'].mean():.4f}  Δ={sub['pes_best'].mean() - sub['pes_default'].mean():+.4f}")


if __name__ == "__main__":
    main()
