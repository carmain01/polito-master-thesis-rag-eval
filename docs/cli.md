# CLI Reference

The `rag-eval` command line tool provides access to every core feature of the
framework without writing Python code.

## Installation

After installing the package in editable mode the CLI is automatically
available inside your virtual environment:

```bash
uv pip install -e .
rag-eval --help
```

## Global Options

| Flag | Description |
|---|---|
| `--verbose` / `-v` | Enable **debug** logging (shows LLM calls, cache hits, etc.). |
| `--quiet` / `-q` | Suppress all output except errors. |
| `--version` | Print the framework version and exit. |

---

## Commands

### `evaluate`

Run evaluation on a dataset using the specified metrics.

```bash
rag-eval evaluate \
  --dataset examples/synthetic_dataset.json \
  --metrics faithfulness,token_f1 \
  --output output/ \
  --format console,json,html
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--dataset` | `-d` | *(required)* | Path to the dataset file (JSON or CSV). |
| `--config` | `-c` | – | Path to a YAML configuration file. |
| `--output` | `-o` | `output` | Directory where report files are written. |
| `--metrics` | `-m` | *(from config)* | Comma-separated metric names. Run `rag-eval metrics` for the full list. |
| `--format` | `-f` | `json,html,csv,console` | Comma-separated output formats to generate. |
| `--provider` | `-p` | *(from config)* | Override the LLM provider (e.g. `openai`, `ollama`). |
| `--max-concurrency` | – | `5` | Maximum number of concurrent evaluations. Set to `1` for local models. |
| `--dry-run` | – | `false` | Validate config & dataset, then exit without running. |

#### Examples

```bash
# Quick check with lightweight metrics only
rag-eval evaluate -d data.json -m token_f1,exactmatch

# Full LLM-judge evaluation with Ollama, sequential
rag-eval evaluate -d data.json -m faithfulness,answerrelevance --max-concurrency 1

# Generate only the HTML report
rag-eval evaluate -d data.json -m token_f1 --format html

# Dry-run: check your config is valid
rag-eval evaluate -d data.json --dry-run
```

---

### `generate`

Generate a synthetic QA dataset from a source text document.

```bash
rag-eval generate \
  --documents my_document.txt \
  --output synthetic.json \
  --num-samples 3 \
  --question-types factual,reasoning \
  --difficulty hard
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--documents` | `-d` | *(required)* | Path to the source text file. |
| `--output` | `-o` | `synthetic_dataset.json` | Output path for the generated JSON dataset. |
| `--num-samples` | `-n` | `1` | Number of QA pairs to generate **per chunk**. |
| `--question-types` | `-q` | `factual` | Comma-separated types: `factual`, `multi-hop`, `reasoning`, `comparative`. |
| `--difficulty` | – | `medium` | Question difficulty: `easy`, `medium`, `hard`. |

---

### `report`

Regenerate visual reports from a previously saved `report.json`.

```bash
rag-eval report --input output/report.json --output-dir new_reports/ --format html,csv
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *(required)* | Path to an existing `report.json`. |
| `--output-dir` | `-o` | `output_regenerated` | Directory for the regenerated report files. |
| `--format` | `-f` | `html,csv,console` | Comma-separated output formats to generate. |

---

### `metrics`

List all available evaluation metrics with their dependency requirements.

```bash
rag-eval metrics
```

Output shows each metric name, the Python class, and whether it requires an
**LLM** client, an **Embedding** client, or runs **standalone** (no external
dependencies).
