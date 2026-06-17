import json

import pytest

from rag_eval.core.types import TestSample
from rag_eval.datasets.loader import (
    _apply_mapping,
    dataset_statistics,
    load_dataset,
    validate_dataset,
)


@pytest.fixture
def sample_data():
    return [
        {"question": "Q1", "answer": "A1", "ground_truth": "GT1", "contexts": ["C1"]},
        {"question": "Q2", "answer": "A2", "ground_truth": "GT2", "contexts": ["C2", "C3"]}
    ]

@pytest.fixture
def jsonl_file(tmp_path, sample_data):
    file_path = tmp_path / "test.jsonl"
    with open(file_path, "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    return file_path

@pytest.fixture
def json_file(tmp_path, sample_data):
    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(sample_data, f)
    return file_path

@pytest.fixture
def csv_file(tmp_path, sample_data):
    file_path = tmp_path / "test.csv"
    import pandas as pd
    df = pd.DataFrame(sample_data)
    # Contexts list to string representation
    df['contexts'] = df['contexts'].apply(json.dumps)
    df.to_csv(file_path, index=False)
    return file_path

def test_load_dataset_jsonl(jsonl_file):
    samples = load_dataset(jsonl_file)
    assert len(samples) == 2
    assert samples[0].question == "Q1"

def test_load_dataset_json(json_file):
    samples = load_dataset(json_file)
    assert len(samples) == 2
    assert samples[1].ground_truth == "GT2"

def test_load_dataset_csv(csv_file):
    samples = load_dataset(csv_file)
    assert len(samples) == 2
    assert isinstance(samples[0].contexts, list)
    assert samples[0].contexts[0] == "C1"

def test_apply_mapping():
    data = {"input_col": "text", "target_col": "ans"}
    mapping = {"input_col": "question", "target_col": "ground_truth"}
    result = _apply_mapping(data, mapping)
    assert "question" in result
    assert result["question"] == "text"
    assert result["ground_truth"] == "ans"
    assert "input_col" not in result

def test_load_dataset_with_mapping(tmp_path):
    file_path = tmp_path / "test_map.json"
    data = [{"q": "test?", "a": "ans", "gt": "ref", "ctx": ["c1"]}]
    with open(file_path, "w") as f:
        json.dump(data, f)

    mapping = {"q": "question", "a": "answer", "gt": "ground_truth", "ctx": "contexts"}
    samples = load_dataset(file_path, column_mapping=mapping)
    assert len(samples) == 1
    assert samples[0].question == "test?"

def test_dataset_statistics(sample_data):
    samples = [TestSample(**d) for d in sample_data]
    stats = dataset_statistics(samples)
    assert stats["num_samples"] == 2
    assert stats["avg_contexts_per_sample"] == 1.5
    assert stats["missing_answers"] == 0

def test_validate_dataset_warnings(caplog, sample_data):
    # Missing answer in one
    sample_data[0]["answer"] = ""
    samples = [TestSample(**d) for d in sample_data]

    import logging
    with caplog.at_level(logging.WARNING):
        validate_dataset(samples)

    assert "missing 'answer'" in caplog.text
