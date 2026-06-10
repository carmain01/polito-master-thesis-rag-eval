"""Report generator — format and export evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

from rag_eval.core.types import EvalReport


class ReportGenerator:
    """Generate human-readable and machine-readable evaluation reports."""

    def __init__(self, report: EvalReport) -> None:
        self.report = report

    def to_json(self, path: str | Path) -> None:
        """Export the report as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report.model_dump(), f, indent=2)

    def to_console(self) -> None:
        """Print a rich summary to the console."""
        # TODO: Implement rich console output
        raise NotImplementedError

    def to_html(self, path: str | Path) -> None:
        """Generate an HTML report with charts and tables."""
        # TODO: Implement HTML report generation
        raise NotImplementedError
