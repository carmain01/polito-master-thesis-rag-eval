#!/usr/bin/env python3
"""Validate PES scores against MathDial teacher move annotations.

Reads the merged CSV from ``run_mathdial_evaluation.py`` and produces:
  - Spearman ρ correlation (PES vs move_score, M3 vs move_score)
  - ANOVA / Kruskal-Wallis between move groups
  - Post-hoc pairwise Mann-Whitney U tests
  - Publication-quality plots
  - Summary tables

Usage
-----
    python validate_pes_mathdial.py
    python validate_pes_mathdial.py --csv output/mathdial/mathdial_pes_dataframe.csv
"""

from __future__ import annotations

import argparse
import json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Visual configuration ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
MOVE_PALETTE = {
    "probing": "#27ae60",   # green — best scaffolding
    "focus": "#2980b9",     # blue  — good redirection
    "generic": "#f39c12",   # orange — neutral
    "telling": "#e74c3c",   # red   — worst (spoiling)
}
MOVE_ORDER = ["telling", "generic", "focus", "probing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PES against MathDial teacher moves.")
    parser.add_argument("--csv", default="output/mathdial/mathdial_pes_dataframe.csv")
    parser.add_argument("--output-dir", default="output/mathdial/plots")
    parser.add_argument("--score-col", default=None, help="Score column to analyze (default: pes_before_gate if present, else pes_score)")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════

def run_correlations(df: pd.DataFrame, score_col: str = "pes_score") -> dict:
    """Compute Spearman ρ for PES and sub-metrics vs move_score."""
    results = {}

    # PES vs move_score
    rho, p = stats.spearmanr(df[score_col], df["move_score"])
    results[f"{score_col}_vs_move_score"] = {"rho": rho, "p_value": p, "n": len(df)}

    # Sub-metrics vs move_score
    for col in ["M1_uptake", "M2_linguistic", "M3_disclosure"]:
        valid = df[[col, "move_score"]].dropna()
        if len(valid) > 2:
            rho, p = stats.spearmanr(valid[col], valid["move_score"])
            results[f"{col}_vs_move_score"] = {"rho": rho, "p_value": p, "n": len(valid)}

    return results


def run_group_tests(df: pd.DataFrame, score_col: str = "pes_score") -> dict:
    """Run ANOVA and Kruskal-Wallis between teacher move groups."""
    results = {}
    groups = [
        df[df["teacher_move"] == move][score_col].values
        for move in MOVE_ORDER
        if move in df["teacher_move"].values
    ]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
        results["anova"] = {"F_statistic": f_stat, "p_value": p_anova}

        h_stat, p_kruskal = stats.kruskal(*groups)
        results["kruskal_wallis"] = {"H_statistic": h_stat, "p_value": p_kruskal}

        # Effect size η²
        ss_between = sum(len(g) * (np.mean(g) - df[score_col].mean()) ** 2 for g in groups)
        ss_total = np.sum((df[score_col].values - df[score_col].mean()) ** 2)
        results["effect_size_eta_squared"] = ss_between / ss_total if ss_total > 0 else 0

        # Post-hoc pairwise Mann-Whitney U
        pairwise = {}
        for i, move_i in enumerate(MOVE_ORDER):
            for j, move_j in enumerate(MOVE_ORDER):
                if j <= i:
                    continue
                g_i = df[df["teacher_move"] == move_i][score_col].values
                g_j = df[df["teacher_move"] == move_j][score_col].values
                if len(g_i) > 0 and len(g_j) > 0:
                    u_stat, p_mw = stats.mannwhitneyu(g_i, g_j, alternative="two-sided")
                    pairwise[f"{move_i}_vs_{move_j}"] = {"U_statistic": u_stat, "p_value": p_mw}
        results["pairwise_mannwhitney"] = pairwise

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_boxplot(df: pd.DataFrame, output_dir: Path, score_col: str = "pes_score") -> None:
    """Box plot: PES distribution by teacher move."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=df, x="teacher_move", y=score_col,
        order=MOVE_ORDER, palette=MOVE_PALETTE,
        width=0.5, ax=ax,
    )
    sns.stripplot(
        data=df, x="teacher_move", y=score_col,
        order=MOVE_ORDER, color="black", alpha=0.1, size=2, jitter=True, ax=ax,
    )

    # Add mean markers
    means = df.groupby("teacher_move")[score_col].mean()
    for i, move in enumerate(MOVE_ORDER):
        if move in means.index:
            ax.scatter(i, means[move], color="white", edgecolors="black", s=80, zorder=5, marker="D")
            ax.annotate(f"μ={means[move]:.3f}", (i, means[move]),
                        textcoords="offset points", xytext=(25, 5), fontsize=10, fontweight="bold")

    ax.set_xlabel("Teacher Move (Pedagogical Strategy)", fontsize=13)
    ax.set_ylabel("PES Score", fontsize=13)
    ax.set_title("PES Distribution by Teacher Move\n(MathDial Validation)", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_pes_by_move.png", dpi=200)
    plt.close(fig)
    logger.info("Saved boxplot.")


def plot_violin(df: pd.DataFrame, output_dir: Path, score_col: str = "pes_score") -> None:
    """Violin plot: PES density by teacher move."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        data=df, x="teacher_move", y=score_col,
        order=MOVE_ORDER, palette=MOVE_PALETTE,
        inner="quartile", cut=0, ax=ax,
    )
    ax.set_xlabel("Teacher Move", fontsize=13)
    ax.set_ylabel("PES Score", fontsize=13)
    ax.set_title("PES Density by Teacher Move\n(MathDial Validation)", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "violin_pes_by_move.png", dpi=200)
    plt.close(fig)
    logger.info("Saved violin plot.")


def plot_m3_by_move(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plot: M3 (No-Immediate-Disclosure) by teacher move — expected strongest signal."""
    m3_valid = df.dropna(subset=["M3_disclosure"])
    if len(m3_valid) < 10:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=m3_valid, x="teacher_move", y="M3_disclosure",
        order=MOVE_ORDER, palette=MOVE_PALETTE,
        width=0.5, ax=ax,
    )

    means = m3_valid.groupby("teacher_move")["M3_disclosure"].mean()
    for i, move in enumerate(MOVE_ORDER):
        if move in means.index:
            ax.scatter(i, means[move], color="white", edgecolors="black", s=80, zorder=5, marker="D")
            ax.annotate(f"μ={means[move]:.3f}", (i, means[move]),
                        textcoords="offset points", xytext=(25, 5), fontsize=10, fontweight="bold")

    ax.set_xlabel("Teacher Move", fontsize=13)
    ax.set_ylabel("M3 Score (No-Immediate-Disclosure)", fontsize=13)
    ax.set_title("M3 (Scaffolding vs Telling) by Teacher Move\n(MathDial — Expected Strongest Signal)", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_m3_by_move.png", dpi=200)
    plt.close(fig)
    logger.info("Saved M3 boxplot.")


def plot_heatmap(df: pd.DataFrame, output_dir: Path, score_col: str = "pes_score") -> None:
    """Correlation heatmap: PES sub-metrics vs move_score."""
    cols = [c for c in [score_col, "M1_uptake", "M2_linguistic", "M3_disclosure", "move_score"] if c in df.columns]
    sub_df = df[cols].dropna()
    if len(sub_df) < 5:
        return

    corr_matrix = sub_df.corr(method="spearman")
    # Rename score_col to 'PES' in plot for clean presentation
    corr_renamed = corr_matrix.rename(columns={score_col: "pes_score"}, index={score_col: "pes_score"})
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr_renamed, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Spearman Correlation:\nPES Sub-metrics vs Teacher Move Score", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_mathdial.png", dpi=200)
    plt.close(fig)
    logger.info("Saved heatmap.")


def plot_scatter_pes_vs_move(df: pd.DataFrame, output_dir: Path, score_col: str = "pes_score") -> None:
    """Scatter: PES vs move_score with jitter and regression."""
    fig, ax = plt.subplots(figsize=(8, 6))
    jitter = np.random.RandomState(42).uniform(-0.05, 0.05, size=len(df))
    ax.scatter(
        df["move_score"] + jitter, df[score_col],
        c=df["teacher_move"].map(MOVE_PALETTE), alpha=0.3, s=15, edgecolors="none",
    )

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["move_score"], df[score_col])
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, label=f"R²={r_value**2:.3f}")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=MOVE_PALETTE[m], label=m) for m in MOVE_ORDER]
    legend_elements.append(plt.Line2D([0], [0], color="black", linestyle="--", label=f"R²={r_value**2:.3f}"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.set_xlabel("Teacher Move Score (Ground Truth)", fontsize=13)
    ax.set_ylabel("PES Score", fontsize=13)
    ax.set_title("PES vs Teacher Move Score\n(MathDial Validation)", fontsize=15, fontweight="bold")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_pes_vs_move.png", dpi=200)
    plt.close(fig)
    logger.info("Saved scatter plot.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    logger.info("Loaded %d rows from %s", len(df), args.csv)

    score_col = args.score_col
    if score_col is None:
        score_col = "pes_before_gate" if "pes_before_gate" in df.columns else "pes_score"
    logger.info("Evaluating using score column: %s", score_col)

    # ── Statistical Tests ──
    print("\n" + "═" * 80)
    print(f"  STATISTICAL VALIDATION: PES ({score_col}) vs MathDial Teacher Moves")
    print("═" * 80)

    # Correlations
    corr_results = run_correlations(df, score_col=score_col)
    print("\n── Spearman ρ Correlations ──")
    for key, res in corr_results.items():
        sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "n.s."
        print(f"  {key:35s}  ρ = {res['rho']:+.4f}  p = {res['p_value']:.2e}  n = {res['n']}  {sig}")

    # Group tests
    group_results = run_group_tests(df, score_col=score_col)
    print("\n── Group Comparison Tests (4 Teacher Moves) ──")
    if "anova" in group_results:
        a = group_results["anova"]
        print(f"  One-way ANOVA:     F = {a['F_statistic']:.4f},  p = {a['p_value']:.2e}")
    if "kruskal_wallis" in group_results:
        k = group_results["kruskal_wallis"]
        print(f"  Kruskal-Wallis:    H = {k['H_statistic']:.4f},  p = {k['p_value']:.2e}")
    if "effect_size_eta_squared" in group_results:
        print(f"  Effect size (η²):  {group_results['effect_size_eta_squared']:.4f}")

    if "pairwise_mannwhitney" in group_results:
        print("\n── Post-hoc Pairwise Mann-Whitney U ──")
        for pair, res in group_results["pairwise_mannwhitney"].items():
            sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "n.s."
            print(f"  {pair:30s}  U = {res['U_statistic']:.1f}  p = {res['p_value']:.2e}  {sig}")

    # Save stats JSON
    stats_output = {
        "score_column": score_col,
        "correlations": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in corr_results.items()},
        "group_tests": {
            k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v))
            for k, v in group_results.items()
            if k != "pairwise_mannwhitney"
        },
        "pairwise_mannwhitney": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in group_results.get("pairwise_mannwhitney", {}).items()
        },
    }
    with open(output_dir / "statistical_tests_mathdial.json", "w") as f:
        json.dump(stats_output, f, indent=2)

    # ── Plots ──
    logger.info("Generating plots...")
    plot_boxplot(df, output_dir, score_col=score_col)
    plot_violin(df, output_dir, score_col=score_col)
    plot_m3_by_move(df, output_dir)
    plot_heatmap(df, output_dir, score_col=score_col)
    plot_scatter_pes_vs_move(df, output_dir, score_col=score_col)

    # ── Summary Tables ──
    print("\n" + "=" * 80)
    print(f"  PES ({score_col}) by Teacher Move (ordered by pedagogical quality)")
    print("=" * 80)
    summary = df.groupby("teacher_move")[score_col].agg(["count", "mean", "median", "std"])
    summary = summary.reindex(MOVE_ORDER)
    print(summary.round(4).to_string())

    # Sub-metrics breakdown
    print("\n" + "=" * 80)
    print("  Sub-metrics by Teacher Move")
    print("=" * 80)
    for move in MOVE_ORDER:
        sub = df[df["teacher_move"] == move]
        print(f"  {move:10s}  M1={sub['M1_uptake'].mean():.3f}  M2={sub['M2_linguistic'].mean():.3f}  M3={sub['M3_disclosure'].mean():.3f}  PES={sub[score_col].mean():.3f}")

    print(f"\n  All plots saved to: {output_dir}/")
    print("═" * 80)


if __name__ == "__main__":
    main()
