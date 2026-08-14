#!/usr/bin/env python3
"""Validate PES scores against MRBench human annotations.

Reads the merged CSV from ``run_mrbench_evaluation.py`` and produces:
  - Spearman ρ correlation (global + per-model)
  - ANOVA / Kruskal-Wallis tests between label groups
  - Publication-quality plots (box, violin, bar, scatter, heatmap)
  - Summary tables

Usage
-----
    python validate_pes_mrbench.py                          # defaults
    python validate_pes_mrbench.py --csv output/mrbench_pes_dataframe.csv
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
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

# Suppress harmless warnings from seaborn/matplotlib
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Visual configuration ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {"No": "#e74c3c", "To some extent": "#f39c12", "Yes": "#27ae60"}
LABEL_ORDER = ["No", "To some extent", "Yes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PES against MRBench annotations.")
    parser.add_argument("--csv", default="output/mrbench_pes_dataframe.csv", help="Path to merged CSV.")
    parser.add_argument("--output-dir", default="output/plots", help="Directory for plots.")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════

def run_spearman(df: pd.DataFrame) -> dict:
    """Compute Spearman ρ globally and per model."""
    results = {}

    # Global: PES vs Providing_Guidance
    rho_pg, p_pg = stats.spearmanr(df["pes_score"], df["providing_guidance"])
    results["global_providing_guidance"] = {"rho": rho_pg, "p_value": p_pg, "n": len(df)}

    # Global: PES vs combined_score
    rho_comb, p_comb = stats.spearmanr(df["pes_score"], df["combined_score"])
    results["global_combined"] = {"rho": rho_comb, "p_value": p_comb, "n": len(df)}

    # Per model
    for model, group in df.groupby("model_name"):
        if len(group) < 3:
            continue
        rho, p = stats.spearmanr(group["pes_score"], group["providing_guidance"])
        results[f"model_{model}"] = {"rho": rho, "p_value": p, "n": len(group)}

    return results


def run_group_tests(df: pd.DataFrame) -> dict:
    """Run ANOVA and Kruskal-Wallis between label groups."""
    results = {}
    groups = [
        df[df["providing_guidance_label"] == label]["pes_score"].values
        for label in LABEL_ORDER
        if label in df["providing_guidance_label"].values
    ]

    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]

    if len(groups) >= 2:
        # One-way ANOVA
        f_stat, p_anova = stats.f_oneway(*groups)
        results["anova"] = {"F_statistic": f_stat, "p_value": p_anova}

        # Kruskal-Wallis (non-parametric, more robust)
        h_stat, p_kruskal = stats.kruskal(*groups)
        results["kruskal_wallis"] = {"H_statistic": h_stat, "p_value": p_kruskal}

        # Effect size: η² for ANOVA
        ss_between = sum(len(g) * (np.mean(g) - df["pes_score"].mean()) ** 2 for g in groups)
        ss_total = np.sum((df["pes_score"].values - df["pes_score"].mean()) ** 2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        results["effect_size_eta_squared"] = eta_squared

        # Post-hoc pairwise Mann-Whitney U tests
        pairwise = {}
        for i, label_i in enumerate(LABEL_ORDER):
            for j, label_j in enumerate(LABEL_ORDER):
                if j <= i:
                    continue
                g_i = df[df["providing_guidance_label"] == label_i]["pes_score"].values
                g_j = df[df["providing_guidance_label"] == label_j]["pes_score"].values
                if len(g_i) > 0 and len(g_j) > 0:
                    u_stat, p_mw = stats.mannwhitneyu(g_i, g_j, alternative="two-sided")
                    pairwise[f"{label_i}_vs_{label_j}"] = {
                        "U_statistic": u_stat,
                        "p_value": p_mw,
                    }
        results["pairwise_mannwhitney"] = pairwise

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plot: PES distribution by Providing_Guidance label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x="providing_guidance_label",
        y="pes_score",
        order=LABEL_ORDER,
        palette=PALETTE,
        width=0.5,
        ax=ax,
    )
    # Overlay individual points
    sns.stripplot(
        data=df,
        x="providing_guidance_label",
        y="pes_score",
        order=LABEL_ORDER,
        color="black",
        alpha=0.15,
        size=3,
        jitter=True,
        ax=ax,
    )

    # Add mean markers
    means = df.groupby("providing_guidance_label")["pes_score"].mean()
    for i, label in enumerate(LABEL_ORDER):
        if label in means.index:
            ax.scatter(i, means[label], color="white", edgecolors="black", s=80, zorder=5, marker="D")
            ax.annotate(
                f"μ={means[label]:.3f}",
                (i, means[label]),
                textcoords="offset points",
                xytext=(25, 5),
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("Providing Guidance (Human Label)", fontsize=13)
    ax.set_ylabel("PES Score", fontsize=13)
    ax.set_title("PES Distribution by Human Guidance Label", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot_pes_by_guidance.png", dpi=200)
    plt.close(fig)
    logger.info("Saved boxplot.")


def plot_violin(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot: PES density by Providing_Guidance label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=df,
        x="providing_guidance_label",
        y="pes_score",
        order=LABEL_ORDER,
        palette=PALETTE,
        inner="quartile",
        cut=0,
        ax=ax,
    )
    ax.set_xlabel("Providing Guidance (Human Label)", fontsize=13)
    ax.set_ylabel("PES Score", fontsize=13)
    ax.set_title("PES Density by Human Guidance Label", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "violin_pes_by_guidance.png", dpi=200)
    plt.close(fig)
    logger.info("Saved violin plot.")


def plot_bar_by_model(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar plot: mean PES per model, colored by label."""
    pivot = df.groupby(["model_name", "providing_guidance_label"])["pes_score"].mean().unstack(fill_value=0)
    # Reorder columns
    cols = [c for c in LABEL_ORDER if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", color=[PALETTE[c] for c in cols], ax=ax, width=0.8, edgecolor="white")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Mean PES Score", fontsize=13)
    ax.set_title("Mean PES Score by Model and Guidance Label", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Providing Guidance", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "bar_pes_by_model_and_label.png", dpi=200)
    plt.close(fig)
    logger.info("Saved bar plot by model.")


def plot_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot: PES vs combined ground truth with regression line."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Add jitter to the ground truth for visibility
    jitter = np.random.RandomState(42).uniform(-0.03, 0.03, size=len(df))

    ax.scatter(
        df["combined_score"] + jitter,
        df["pes_score"],
        c=df["providing_guidance_label"].map(PALETTE),
        alpha=0.4,
        s=25,
        edgecolors="none",
    )

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["combined_score"], df["pes_score"]
    )
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, label=f"R²={r_value**2:.3f}")

    # Custom legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PALETTE[l], label=l) for l in LABEL_ORDER]
    legend_elements.append(plt.Line2D([0], [0], color="black", linestyle="--", label=f"R²={r_value**2:.3f}"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.set_xlabel("Combined Ground Truth (0.6·Guidance + 0.4·Actionability)", fontsize=12)
    ax.set_ylabel("PES Score", fontsize=13)
    ax.set_title("PES vs Combined Human Ground Truth", fontsize=15, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_pes_vs_ground_truth.png", dpi=200)
    plt.close(fig)
    logger.info("Saved scatter plot.")


def plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: Spearman correlations between PES, sub-metrics, and human annotations."""
    cols = [c for c in [
        "pes_score", "M1_uptake", "M2_linguistic", "M3_disclosure",
        "providing_guidance", "actionability", "combined_score",
    ] if c in df.columns]

    # Drop rows with NaN in any of these columns
    sub_df = df[cols].dropna()
    if len(sub_df) < 5:
        logger.warning("Not enough data for heatmap (n=%d).", len(sub_df))
        return

    corr_matrix = sub_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Spearman Correlation Heatmap:\nPES Sub-metrics vs Human Annotations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_correlations.png", dpi=200)
    plt.close(fig)
    logger.info("Saved heatmap.")


def plot_model_ranking(df: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bar chart: models ranked by mean PES score."""
    model_means = df.groupby("model_name")["pes_score"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", n_colors=len(model_means))
    bars = ax.barh(model_means.index, model_means.values, color=colors, edgecolor="white")

    # Add value labels
    for bar, val in zip(bars, model_means.values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Mean PES Score", fontsize=13)
    ax.set_title("Model Ranking by Mean PES", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 1.1)
    fig.tight_layout()
    fig.savefig(output_dir / "model_ranking.png", dpi=200)
    plt.close(fig)
    logger.info("Saved model ranking plot.")


# ═══════════════════════════════════════════════════════════════════════════
# Summary Tables
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Print and save summary tables."""

    # ── Table 1: PES by (Model × Label) ──
    pivot = df.pivot_table(
        values="pes_score",
        index="model_name",
        columns="providing_guidance_label",
        aggfunc=["mean", "count"],
    )
    print("\n" + "=" * 80)
    print("  Table 1: Mean PES by Model × Providing_Guidance Label")
    print("=" * 80)

    # Flatten multi-level columns for display
    summary = df.groupby(["model_name", "providing_guidance_label"])["pes_score"].agg(
        ["mean", "std", "count"]
    ).round(4)
    print(summary.to_string())
    summary.to_csv(output_dir / "summary_model_x_label.csv")

    # ── Table 2: Global PES by Label ──
    print("\n" + "=" * 80)
    print("  Table 2: Global PES by Providing_Guidance Label")
    print("=" * 80)
    global_summary = df.groupby("providing_guidance_label")["pes_score"].agg(
        ["mean", "median", "std", "min", "max", "count"]
    ).round(4)
    global_summary = global_summary.reindex(LABEL_ORDER)
    print(global_summary.to_string())
    global_summary.to_csv(output_dir / "summary_global_by_label.csv")

    # ── Table 3: Model overall ranking ──
    print("\n" + "=" * 80)
    print("  Table 3: Model Ranking by Mean PES")
    print("=" * 80)
    ranking = df.groupby("model_name")["pes_score"].agg(
        ["mean", "median", "std", "count"]
    ).round(4).sort_values("mean", ascending=False)
    print(ranking.to_string())
    ranking.to_csv(output_dir / "summary_model_ranking.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    df = pd.read_csv(args.csv)
    logger.info("Loaded %d rows from %s", len(df), args.csv)

    # ── Statistical Tests ──
    print("\n" + "═" * 80)
    print("  STATISTICAL VALIDATION: PES vs MRBench Human Annotations")
    print("═" * 80)

    # Spearman ρ
    spearman_results = run_spearman(df)
    print("\n── Spearman ρ Correlations ──")
    for key, res in spearman_results.items():
        sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "n.s."
        print(f"  {key:40s}  ρ = {res['rho']:+.4f}  p = {res['p_value']:.2e}  n = {res['n']}  {sig}")

    # ANOVA / Kruskal-Wallis
    group_results = run_group_tests(df)
    print("\n── Group Comparison Tests ──")
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
            print(f"  {pair:35s}  U = {res['U_statistic']:.1f}  p = {res['p_value']:.2e}  {sig}")

    # Save stats to JSON
    import json
    stats_output = {
        "spearman": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in spearman_results.items()},
        "group_tests": {
            k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v))
            for k, v in group_results.items()
            if k != "pairwise_mannwhitney"
        },
    }
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(stats_output, f, indent=2)

    # ── Plots ──
    logger.info("Generating plots...")
    plot_boxplot(df, output_dir)
    plot_violin(df, output_dir)
    plot_bar_by_model(df, output_dir)
    plot_scatter(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_model_ranking(df, output_dir)

    # ── Summary Tables ──
    print_summary_table(df, output_dir)

    print("\n" + "═" * 80)
    print(f"  All plots saved to: {output_dir}/")
    print(f"  Statistical tests saved to: {output_dir}/statistical_tests.json")
    print("═" * 80)


if __name__ == "__main__":
    main()
