import pytest
from unittest.mock import patch

from rag_eval.datasets.benchmarks import (
    get_benchmark_adapter,
    HotpotQAAdapter,
    NaturalQuestionsAdapter,
    TriviaQAAdapter,
    Squad2Adapter,
    RGBAdapter,
    RecallAdapter
)
from rag_eval.core.types import TestSample


def test_registry():
    adapter = get_benchmark_adapter("hotpotqa")
    assert isinstance(adapter, HotpotQAAdapter)
    
    with pytest.raises(ValueError):
        get_benchmark_adapter("non_existent")


@patch("rag_eval.datasets.benchmarks.hotpotqa.HotpotQAAdapter._load_hf_dataset")
def test_hotpotqa_adapter(mock_load):
    mock_load.return_value = [
        {
            "question": "What is 2+2?",
            "answer": "4",
            "context": {"title": ["Math"], "sentences": [["2+2", " is 4."]]},
            "level": "easy",
            "type": "comparison"
        }
    ]
    
    adapter = get_benchmark_adapter("hotpotqa")
    samples = adapter.load()
    
    assert len(samples) == 1
    assert samples[0].question == "What is 2+2?"
    assert samples[0].ground_truth == "4"
    assert samples[0].contexts[0] == "Title: Math\n2+2 is 4."
    assert samples[0].metadata["level"] == "easy"


@patch("rag_eval.datasets.benchmarks.nq.NaturalQuestionsAdapter._load_hf_dataset")
def test_nq_adapter(mock_load):
    mock_load.return_value = [
        {
            "question": "who is the author of the book?",
            "answer": ["John Doe", "J. Doe"]
        }
    ]
    
    adapter = get_benchmark_adapter("nq")
    samples = adapter.load()
    
    assert len(samples) == 1
    assert samples[0].question == "who is the author of the book?"
    assert samples[0].ground_truth == "John Doe"
    assert len(samples[0].contexts) == 0


@patch("rag_eval.datasets.benchmarks.triviaqa.TriviaQAAdapter._load_hf_dataset")
def test_triviaqa_adapter(mock_load):
    mock_load.return_value = [
        {
            "question": "Capital of France?",
            "answer": {"value": "Paris", "aliases": ["City of Light"]},
            "search_results": {"search_context": ["Paris is the capital of France."]}
        }
    ]
    
    adapter = get_benchmark_adapter("triviaqa")
    samples = adapter.load()
    
    assert len(samples) == 1
    assert samples[0].ground_truth == "Paris"
    assert samples[0].contexts == ["Paris is the capital of France."]


@patch("rag_eval.datasets.benchmarks.squad.Squad2Adapter._load_hf_dataset")
def test_squad2_adapter_filtering(mock_load):
    mock_load.return_value = [
        {"question": "Answerable?", "answers": {"text": ["Yes"]}, "context": "Yes it is."},
        {"question": "Unanswerable?", "answers": {"text": []}, "context": "Nothing here."}
    ]
    
    adapter = get_benchmark_adapter("squad2")
    samples = adapter.load()
    
    # Unanswerable question should be filtered out
    assert len(samples) == 1
    assert samples[0].question == "Answerable?"
    assert samples[0].ground_truth == "Yes"


@patch("rag_eval.datasets.benchmarks.rgb.RGBAdapter._load_hf_dataset")
def test_rgb_adapter(mock_load):
    mock_load.return_value = [
        {
            "query": "Is RGB cool?",
            "answer": "Yes",
            "docs": [{"text": "RGB is cool."}]
        }
    ]
    
    adapter = get_benchmark_adapter("rgb")
    samples = adapter.load()
    
    assert len(samples) == 1
    assert samples[0].question == "Is RGB cool?"
    assert samples[0].ground_truth == "Yes"
    assert samples[0].contexts == ["RGB is cool."]


@patch("rag_eval.datasets.benchmarks.recall.RecallAdapter._load_hf_dataset")
def test_recall_adapter(mock_load):
    mock_load.return_value = [
        {
            "question": "What does RECALL test?",
            "ground_truth": "LLMs",
            "context": "RECALL tests LLMs."
        }
    ]
    
    adapter = get_benchmark_adapter("recall")
    samples = adapter.load()
    
    assert len(samples) == 1
    assert samples[0].question == "What does RECALL test?"
    assert samples[0].ground_truth == "LLMs"
    assert samples[0].contexts == ["RECALL tests LLMs."]
