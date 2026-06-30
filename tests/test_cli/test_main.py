import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rag_eval.cli.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "rag-eval" in result.stdout


def test_metrics_command():
    result = runner.invoke(app, ["metrics"])
    assert result.exit_code == 0
    assert "Available Metrics:" in result.stdout


@patch("rag_eval.cli.main.Evaluator")
@patch("rag_eval.cli.main.ReportGenerator")
@patch("rag_eval.cli.main.load_dataset")
@patch("rag_eval.cli.main._instantiate_metrics")
def test_evaluate_command_dry_run(
    mock_instantiate, mock_load, mock_report_gen, mock_evaluator, tmp_path
):
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text("[]")

    result = runner.invoke(
        app, ["evaluate", "--dataset", str(dataset_file), "--dry-run", "--metrics", "exact_match"]
    )

    assert result.exit_code == 0
    assert "Dry-run completed successfully" in result.stdout
    mock_load.assert_called_once()
    mock_evaluator.assert_not_called()


@patch("rag_eval.cli.main.Evaluator")
@patch("rag_eval.cli.main.ReportGenerator")
@patch("rag_eval.cli.main.load_dataset")
@patch("rag_eval.cli.main._instantiate_metrics")
def test_evaluate_command_success(
    mock_instantiate, mock_load, mock_report_gen, mock_evaluator, tmp_path
):
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text("[]")

    mock_load.return_value = []
    mock_eval_instance = MagicMock()
    mock_evaluator.return_value = mock_eval_instance

    output_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--dataset",
            str(dataset_file),
            "--output",
            str(output_dir),
            "--metrics",
            "exact_match",
        ],
    )

    assert result.exit_code == 0
    assert "Evaluation complete!" in result.stdout
    mock_evaluator.assert_called_once()
    mock_eval_instance.evaluate.assert_called_once()
    mock_report_gen.assert_called_once()


@patch("rag_eval.cli.main.SyntheticDataGenerator")
def test_generate_command(mock_gen_class, tmp_path):
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("Test document")

    out_file = tmp_path / "synthetic.json"

    mock_gen_instance = MagicMock()

    # We need to mock the async generate_qa_pairs method
    async def mock_generate(*args, **kwargs):
        mock_sample = MagicMock()
        mock_sample.model_dump.return_value = {"question": "Q?"}
        return [mock_sample]

    mock_gen_instance.generate_qa_pairs = mock_generate
    mock_gen_class.return_value = mock_gen_instance

    result = runner.invoke(
        app,
        ["generate", "--documents", str(doc_file), "--output", str(out_file), "--num-samples", "1"],
    )

    assert result.exit_code == 0
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert len(data) == 1
    assert data[0] == {"question": "Q?"}


@patch("rag_eval.cli.main.ReportGenerator")
def test_report_command(mock_gen_class, tmp_path):
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps({"report": {"summary": {"score": 1.0}, "samples": []}}))

    out_dir = tmp_path / "output_regenerated"

    mock_gen_instance = MagicMock()
    mock_gen_class.return_value = mock_gen_instance

    result = runner.invoke(
        app,
        [
            "report",
            "--input",
            str(report_file),
            "--output-dir",
            str(out_dir),
            "--format",
            "console",
        ],
    )

    assert result.exit_code == 0
    mock_gen_instance.to_console.assert_called_once()
