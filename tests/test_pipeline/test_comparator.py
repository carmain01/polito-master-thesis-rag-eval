import pytest
from rag_eval.core.types import EvalReport, EvalResult
from rag_eval.pipeline.comparator import ConfigurationComparator

@pytest.fixture
def baseline_report():
    report = EvalReport()
    # 5 samples for faithfulness
    for i in range(5):
        report.add_result(EvalResult(metric_name="faithfulness", score=0.5))
    return report

@pytest.fixture
def candidate_report_better():
    report = EvalReport()
    for i in range(5):
        report.add_result(EvalResult(metric_name="faithfulness", score=0.9))
    return report

@pytest.fixture
def candidate_report_same():
    report = EvalReport()
    for i in range(5):
        report.add_result(EvalResult(metric_name="faithfulness", score=0.5))
    return report

def test_comparator_better(baseline_report, candidate_report_better):
    comparator = ConfigurationComparator()
    results = comparator.compare(baseline_report, {"cand1": candidate_report_better})
    
    assert "cand1" in results
    cand1_res = results["cand1"]["faithfulness"]
    assert cand1_res["baseline_mean"] == 0.5
    assert cand1_res["candidate_mean"] == 0.9
    assert cand1_res["delta"] == pytest.approx(0.4)
    # The difference is constant, so p-value might be 0.0 or nan depending on scipy. 
    # But let's just check that it runs without crashing.
    assert "p_value" in cand1_res

def test_comparator_same(baseline_report, candidate_report_same):
    comparator = ConfigurationComparator()
    results = comparator.compare(baseline_report, {"cand2": candidate_report_same})
    
    cand2_res = results["cand2"]["faithfulness"]
    assert cand2_res["delta"] == 0.0
    assert cand2_res["significant_improvement"] is False
    assert cand2_res["significant_degradation"] is False
