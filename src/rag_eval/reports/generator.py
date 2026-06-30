"""Report generator — format and export evaluation results."""

from __future__ import annotations

import csv
import importlib.metadata
import json
from typing import Any
import statistics
import string
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag_eval.core.types import EvalReport, EvalResult

_VENDOR_DIR = Path(__file__).parent / "vendor"


def _load_vendor_assets() -> dict[str, str]:
    """Read vendored CSS/JS files and return them as strings for inlining."""
    assets: dict[str, str] = {}
    for name, filename in [
        ("bootstrap_css", "bootstrap.min.css"),
        ("bootstrap_js", "bootstrap.bundle.min.js"),
        ("chartjs", "chart.min.js"),
    ]:
        vendor_file = _VENDOR_DIR / filename
        if vendor_file.exists():
            assets[name] = vendor_file.read_text(encoding="utf-8")
        else:
            assets[name] = f"/* {filename} not found — report may not render correctly offline */"
    return assets


class ReportGenerator:
    """Generate human-readable and machine-readable evaluation reports."""

    def __init__(self, report: EvalReport) -> None:
        self.report = report

    def to_json(self, path: str | Path) -> None:
        """Export the report as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            version = importlib.metadata.version("rag_eval")
        except importlib.metadata.PackageNotFoundError:
            version = "0.1.0"

        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "framework_version": version,
                "config": self.report.config,
            },
            "report": self.report.model_dump(mode="json"),
        }

        with open(path, "w") as f:
            json.dump(output_data, f, indent=2)

    def to_csv(self, path: str | Path) -> None:
        """Export per-sample scores as CSV in wide format for spreadsheet analysis.

        Each row represents one evaluation sample. Columns are ``Sample_Index``
        followed by ``{metric}_Score`` and ``{metric}_Reason`` pairs for every
        metric present in the report.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metrics = sorted({r.metric_name for r in self.report.results})

        # Group results by sample — a new sample starts when a metric name repeats.
        sample_results: dict[int, dict[str, EvalResult]] = defaultdict(dict)
        current_sample = 0
        seen_metrics_in_current: set[str] = set()

        for res in self.report.results:
            if res.metric_name in seen_metrics_in_current:
                current_sample += 1
                seen_metrics_in_current = set()
            seen_metrics_in_current.add(res.metric_name)
            sample_results[current_sample][res.metric_name] = res

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            header = ["Sample_Index"]
            for m in metrics:
                header.append(f"{m}_Score")
                header.append(f"{m}_Reason")
            writer.writerow(header)

            for s_idx, res_dict in sample_results.items():
                row: list[object] = [s_idx]
                for m in metrics:
                    if m in res_dict:
                        row.extend([res_dict[m].score, res_dict[m].reason])
                    else:
                        row.extend(["", ""])
                writer.writerow(row)

    def to_console(self) -> None:
        """Print a rich summary to the console."""
        console = Console()

        def sparkline(scores: list[float]) -> str:
            """Generate a simple sparkline for a list of scores."""
            if not scores:
                return ""
            ticks = " ▂▃▄▅▆▇█"
            # Create a 10-bin histogram
            bins = [0] * 10
            for s in scores:
                b = min(int(s * 10), 9)
                bins[b] += 1
            max_bin = max(bins)
            if max_bin == 0:
                return " " * 10
            line = "".join(ticks[int((c / max_bin) * 7)] for c in bins)
            return line

        # Group scores by metric
        grouped: dict[str, list[float]] = defaultdict(list)
        best_samples: dict[str, tuple[int, float]] = {}
        worst_samples: dict[str, tuple[int, float]] = {}

        current_sample = 0
        seen_metrics_in_current: set[str] = set()

        for res in self.report.results:
            if res.metric_name in seen_metrics_in_current:
                current_sample += 1
                seen_metrics_in_current = set()
            seen_metrics_in_current.add(res.metric_name)

            grouped[res.metric_name].append(res.score)

            # Track best/worst
            if res.metric_name not in best_samples or res.score > best_samples[res.metric_name][1]:
                best_samples[res.metric_name] = (current_sample, res.score)
            if (
                res.metric_name not in worst_samples
                or res.score < worst_samples[res.metric_name][1]
            ):
                worst_samples[res.metric_name] = (current_sample, res.score)

        table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", justify="right", style="green")
        table.add_column("Median", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Distribution", justify="center", style="blue")

        for metric, scores in grouped.items():
            mean = statistics.mean(scores)
            median = statistics.median(scores) if len(scores) > 0 else 0
            std_val = statistics.stdev(scores) if len(scores) > 1 else 0.0
            min_val = min(scores) if len(scores) > 0 else 0
            max_val = max(scores) if len(scores) > 0 else 0
            table.add_row(
                metric,
                f"{mean:.3f}",
                f"{median:.3f}",
                f"{std_val:.3f}",
                f"{min_val:.3f}",
                f"{max_val:.3f}",
                sparkline(scores),
            )

        console.print(table)

        # Print best/worst
        if best_samples or worst_samples:
            bw_text = Text()
            for m in grouped.keys():
                best_idx, best_val = best_samples.get(m, (-1, 0))
                worst_idx, worst_val = worst_samples.get(m, (-1, 0))
                bw_text.append(f"{m}:\n", style="bold cyan")
                bw_text.append(f"  Best: Sample {best_idx} ({best_val:.3f})\n", style="green")
                bw_text.append(f"  Worst: Sample {worst_idx} ({worst_val:.3f})\n", style="red")
            console.print(Panel(bw_text, title="Extremes (Best/Worst Samples)"))

        # Print cost/token summary if present
        cost_tokens = [k for k in self.report.summary.keys() if k.startswith("total_")]
        if cost_tokens:
            cost_text = Text()
            for key in cost_tokens:
                val = self.report.summary[key]
                if isinstance(val, float):
                    cost_text.append(f"{key}: ${val:.4f}\n", style="yellow")
                else:
                    cost_text.append(f"{key}: {val}\n", style="yellow")
            console.print(Panel(cost_text, title="Cost & Usage Summary"))

    def to_html(self, path: str | Path) -> None:
        """Generate an HTML report with charts and tables."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        template_dir = Path(__file__).parent / "templates"
        template_file = template_dir / "report_template.html"

        if not template_file.exists():
            raise FileNotFoundError(f"Template not found at {template_file}")

        with open(template_file) as f:
            template = string.Template(f.read())

        # Group scores by metric for charts
        grouped: dict[str, list[float]] = defaultdict(list)
        for res in self.report.results:
            grouped[res.metric_name].append(res.score)

        labels = list(grouped.keys())
        means = [statistics.mean(grouped[m]) for m in labels] if labels else []

        # Prepare distribution data (10 bins per metric)
        distributions = {}
        for metric, scores in grouped.items():
            bins = [0] * 10
            for s in scores:
                b = min(int(s * 10), 9)
                bins[b] += 1
            distributions[metric] = bins

        # Group by sample for drill-down table
        sample_results: dict[int, dict[str, EvalResult]] = defaultdict(dict)
        current_sample = 0
        seen_metrics_in_current: set[str] = set()
        for res in self.report.results:
            if res.metric_name in seen_metrics_in_current:
                current_sample += 1
                seen_metrics_in_current = set()
            seen_metrics_in_current.add(res.metric_name)
            sample_results[current_sample][res.metric_name] = res

        # Serialize for Chart.js
        chart_labels = json.dumps(labels)
        chart_data = json.dumps(means)
        chart_distributions = json.dumps(distributions)

        # Generate table rows with drill-down
        table_rows = ""
        for s_idx, res_dict in sample_results.items():
            avg_score = sum(r.score for r in res_dict.values()) / len(res_dict) if res_dict else 0

            # The main row
            table_rows += f'<tr data-bs-toggle="collapse" data-bs-target="#sample-{s_idx}" class="accordion-toggle" style="cursor: pointer;">'
            table_rows += (
                f'<td><button class="btn btn-sm btn-outline-secondary">+</button> {s_idx}</td>'
            )
            table_rows += f"<td><strong>{avg_score:.3f}</strong> (Avg)</td>"
            table_rows += f"<td>Click to expand {len(res_dict)} metrics</td>"
            table_rows += "</tr>\n"

            # The expanded details
            table_rows += f'<tr><td colspan="3" class="p-0 border-0"><div class="collapse" id="sample-{s_idx}"><table class="table table-sm mb-0 bg-light">'
            for m, res in res_dict.items():
                safe_reason = (
                    res.reason.replace("<", "&lt;").replace(">", "&gt;") if res.reason else ""
                )
                table_rows += f'<tr><td style="width: 20%;">{m}</td><td style="width: 10%;">{res.score:.3f}</td><td>{safe_reason}</td></tr>'
            table_rows += "</table></div></td></tr>\n"

        # Load vendored assets for standalone output
        assets = _load_vendor_assets()

        html_content = template.safe_substitute(
            bootstrap_css=assets["bootstrap_css"],
            bootstrap_js=assets["bootstrap_js"],
            chartjs=assets["chartjs"],
            chart_labels=chart_labels,
            chart_data=chart_data,
            chart_distributions=chart_distributions,
            table_rows=table_rows,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        with open(path, "w") as f:
            f.write(html_content)

    @staticmethod
    def to_html_comparative(
        reports: dict[str, EvalReport],
        path: str | Path,
    ) -> None:
        """Generate a comparative HTML report overlaying multiple evaluation runs.

        Args:
            reports: Mapping of *config label* → ``EvalReport``.  Each label
                     is shown in the legend (e.g. ``"gpt-4o"`` vs ``"llama-3"``).
            path: Destination file path for the HTML output.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        template_dir = Path(__file__).parent / "templates"
        template_file = template_dir / "comparative_template.html"

        if not template_file.exists():
            raise FileNotFoundError(f"Template not found at {template_file}")

        with open(template_file) as f:
            template = string.Template(f.read())

        # Collect all metric names across every report.
        all_metrics: set[str] = set()
        for report in reports.values():
            for res in report.results:
                all_metrics.add(res.metric_name)
        labels = sorted(all_metrics)

        # Build per-report mean scores for each metric.
        chart_datasets: list[dict[str, Any]] = []
        means_by_config: dict[str, dict[str, float]] = {}

        for config_label, report in reports.items():
            grouped: dict[str, list[float]] = defaultdict(list)
            for res in report.results:
                grouped[res.metric_name].append(res.score)
            means = {m: statistics.mean(grouped[m]) if m in grouped else 0.0 for m in labels}
            means_by_config[config_label] = means
            chart_datasets.append(
                {
                    "label": config_label,
                    "data": [round(means[m], 4) for m in labels],
                }
            )

        # Build config column headers.
        config_headers = "".join(f"<th>{label}</th>" for label in reports.keys())

        # Build summary rows with delta.
        summary_rows = ""
        for m in labels:
            scores_across = [means_by_config[c][m] for c in reports.keys()]
            delta = max(scores_across) - min(scores_across)
            delta_style = (
                'style="color: green;"'
                if delta < 0.05
                else 'style="color: red; font-weight: bold;"'
            )
            summary_rows += f"<tr><td>{m}</td>"
            for c in reports.keys():
                summary_rows += f"<td>{means_by_config[c][m]:.3f}</td>"
            summary_rows += f"<td {delta_style}>{delta:.3f}</td></tr>\n"

        assets = _load_vendor_assets()

        html_content = template.safe_substitute(
            bootstrap_css=assets["bootstrap_css"],
            bootstrap_js=assets["bootstrap_js"],
            chartjs=assets["chartjs"],
            chart_labels=json.dumps(labels),
            chart_datasets=json.dumps(chart_datasets),
            config_headers=config_headers,
            summary_rows=summary_rows,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        with open(path, "w") as f:
            f.write(html_content)
