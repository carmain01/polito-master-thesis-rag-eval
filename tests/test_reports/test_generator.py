"""Tests for the report generation capabilities."""

import csv
import json
from pathlib import Path

from rag_eval.core.types import EvalReport, EvalResult
from rag_eval.reports.generator import ReportGenerator


def test_to_json(tmp_path: Path) -> None:
    """Test generating a JSON report."""
    report = EvalReport()
    report.add_result(EvalResult(metric_name="faithfulness", score=0.9, reason="Good"))
    report.compute_summary()

    generator = ReportGenerator(report)
    out_path = tmp_path / "test_report.json"
    generator.to_json(out_path)

    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)

    assert "metadata" in data
    assert "config" in data["metadata"]
    assert "report" in data
    assert len(data["report"]["results"]) == 1
    assert data["report"]["results"][0]["metric_name"] == "faithfulness"
    assert data["report"]["results"][0]["score"] == 0.9


def test_to_csv(tmp_path: Path) -> None:
    """Test generating a CSV report."""
    report = EvalReport()
    report.add_result(EvalResult(metric_name="faithfulness", score=0.9, reason="Good"))

    generator = ReportGenerator(report)
    out_path = tmp_path / "test_report.csv"
    generator.to_csv(out_path)

    assert out_path.exists()
    with open(out_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 2  # Header + 1 data row
    assert rows[0] == ["Sample_Index", "faithfulness_Score", "faithfulness_Reason"]
    assert rows[1] == ["0", "0.9", "Good"]


def test_to_html(tmp_path: Path) -> None:
    """Test generating an HTML report."""
    report = EvalReport()
    report.add_result(EvalResult(metric_name="faithfulness", score=0.9, reason="Good"))

    generator = ReportGenerator(report)
    out_path = tmp_path / "test_report.html"
    generator.to_html(out_path)

    assert out_path.exists()
    with open(out_path) as f:
        html = f.read()

    assert "Evaluation Report" in html
    assert "faithfulness" in html
    assert "0.9" in html
    assert "Good" in html
    assert "data-bs-toggle" in html  # Check for drill-down


def test_to_console(capsys) -> None:
    """Test generating a console report."""
    report = EvalReport()
    report.add_result(EvalResult(metric_name="faithfulness", score=0.9, reason="Good"))
    report.add_result(EvalResult(metric_name="faithfulness", score=0.5, reason="OK"))
    report.compute_summary()

    generator = ReportGenerator(report)
    generator.to_console()

    captured = capsys.readouterr()
    assert "Evaluation Summary" in captured.out
    assert "faithfulness" in captured.out
    assert "0.700" in captured.out  # mean of 0.9 and 0.5
    assert "Std" in captured.out  # std column header
    assert "Extremes" in captured.out  # best/worst panel title
    assert "Best" in captured.out
    assert "Worst" in captured.out


def test_to_html_standalone(tmp_path: Path) -> None:
    """Test that the HTML report is fully standalone (no CDN links)."""
    report = EvalReport()
    report.add_result(EvalResult(metric_name="faithfulness", score=0.9, reason="Good"))

    generator = ReportGenerator(report)
    out_path = tmp_path / "standalone.html"
    generator.to_html(out_path)

    with open(out_path) as f:
        html = f.read()

    # Must NOT contain any CDN link/script tags
    assert "cdn.jsdelivr.net" not in html
    assert '<link href="https://' not in html
    assert '<script src="https://' not in html

    # Must contain inlined Bootstrap and Chart.js content
    assert "bootstrap" in html.lower()
    assert "Chart" in html


def test_to_html_comparative(tmp_path: Path) -> None:
    """Test generating a comparative HTML report from multiple configs."""
    report_a = EvalReport()
    report_a.add_result(EvalResult(metric_name="faithfulness", score=0.9, reason="Good"))
    report_a.add_result(EvalResult(metric_name="relevance", score=0.8, reason="OK"))

    report_b = EvalReport()
    report_b.add_result(EvalResult(metric_name="faithfulness", score=0.6, reason="Weak"))
    report_b.add_result(EvalResult(metric_name="relevance", score=0.7, reason="Fair"))

    out_path = tmp_path / "comparative.html"
    ReportGenerator.to_html_comparative(
        reports={"Config A": report_a, "Config B": report_b},
        path=out_path,
    )

    assert out_path.exists()
    with open(out_path) as f:
        html = f.read()

    assert "Comparative" in html
    assert "Config A" in html
    assert "Config B" in html
    assert "faithfulness" in html
    assert "relevance" in html
    # Should contain delta column
    assert "Delta" in html
    # Standalone — no CDN
    assert "cdn.jsdelivr.net" not in html
