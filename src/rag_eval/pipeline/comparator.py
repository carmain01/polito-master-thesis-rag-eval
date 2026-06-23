"""Comparator — compares evaluation reports from multiple RAG configurations."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from collections import defaultdict

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from rag_eval.core.types import EvalReport


class ConfigurationComparator:
    """Compares multiple evaluation reports to determine statistically significant differences."""

    def compare(
        self, 
        baseline_report: EvalReport, 
        candidate_reports: Dict[str, EvalReport]
    ) -> Dict[str, Any]:
        """
        Compare candidate configurations against a baseline configuration.
        
        Args:
            baseline_report: The baseline EvalReport.
            candidate_reports: A dictionary mapping config names to EvalReports.
            
        Returns:
            A nested dictionary containing metric comparisons, deltas, and p-values.
        """
        results: Dict[str, Any] = {}

        # Extract baseline scores per sample per metric
        # Assuming sample order is identical across reports for paired tests
        # We group by metric_name, then by sample (using list order assuming it matches)
        baseline_scores = self._group_scores_by_metric(baseline_report)

        for config_name, candidate_report in candidate_reports.items():
            candidate_scores = self._group_scores_by_metric(candidate_report)
            config_results = {}

            # Compare overlapping metrics
            all_metrics = set(baseline_scores.keys()).intersection(candidate_scores.keys())

            for metric in all_metrics:
                b_scores = baseline_scores[metric]
                c_scores = candidate_scores[metric]

                # If lengths mismatch, we truncate to the shortest, but warn (paired test requires equal length)
                min_len = min(len(b_scores), len(c_scores))
                b_scores = b_scores[:min_len]
                c_scores = c_scores[:min_len]

                b_mean = sum(b_scores) / min_len if min_len > 0 else 0.0
                c_mean = sum(c_scores) / min_len if min_len > 0 else 0.0
                delta = c_mean - b_mean
                
                # Compute statistical significance using paired t-test
                p_value: Optional[float] = None
                significant = False
                
                if HAS_SCIPY and min_len > 1:
                    # ttest_rel returns Ttest_relResult(statistic, pvalue)
                    t_stat, p_val = stats.ttest_rel(c_scores, b_scores)
                    p_value = float(p_val)
                    # Handle NaN p-values (e.g. when all scores are identical)
                    if math.isnan(p_value):
                        p_value = 1.0
                    
                    if p_value < 0.05:
                        significant = True

                config_results[metric] = {
                    "baseline_mean": b_mean,
                    "candidate_mean": c_mean,
                    "delta": delta,
                    "p_value": p_value,
                    "significant_improvement": significant and delta > 0,
                    "significant_degradation": significant and delta < 0,
                }
            
            results[config_name] = config_results

        return results

    def _group_scores_by_metric(self, report: EvalReport) -> Dict[str, List[float]]:
        """Groups sample scores by metric name."""
        grouped: Dict[str, List[float]] = defaultdict(list)
        for res in report.results:
            grouped[res.metric_name].append(res.score)
        return dict(grouped)
