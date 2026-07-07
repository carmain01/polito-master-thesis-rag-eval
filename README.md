# RAG Evaluation Framework

A modular, extensible framework for evaluating Retrieval-Augmented Generation (RAG) systems.
Supports LLM-as-judge metrics, traditional NLP metrics, synthetic dataset generation, and rich
HTML/CSV/JSON reporting — all accessible via a CLI or the Python API.

## Features

- **10 evaluation metrics** — mix LLM-judge and traditional NLP metrics in a single run
- **Synthetic QA generation** — create evaluation datasets from raw text documents
- **Multi-provider LLM support** — Ollama (local), OpenAI, Anthropic, Google GenAI, vLLM
- **Rich reporting** — interactive HTML dashboards, CSV exports, JSON, and console tables
- **Caching** — disk-based LLM response cache to avoid redundant API calls
- **LangChain integration** — evaluate LangChain RAG chains directly
- **Full CLI** — every feature available without writing Python code

## Project Structure

```
├── src/rag_eval/
│   ├── cli/               # Typer-based command line interface
│   ├── core/              # Data models, config, metric registry
│   ├── metrics/           # Evaluation metrics (10 built-in)
│   ├── datasets/          # Dataset loading and synthetic generation
│   ├── pipeline/          # Evaluation orchestrator
│   ├── reports/           # Report generation (HTML, CSV, JSON, console)
│   ├── integrations/      # LangChain adapter
│   ├── prompts/           # LLM prompt templates
│   └── utils/             # LLM clients, embeddings, caching, chunking
├── tests/                 # Unit and integration tests
├── examples/              # Runnable usage examples
├── docs/                  # Extended documentation
└── configs/               # Default YAML configuration
```

## Installation

Requires **Python ≥ 3.10**.

```bash
# Clone the repository
git clone <repo-url> && cd Tesi

# Install in editable mode (with uv or pip)
uv pip install -e .

# Or, with pip
pip install -e .

# Install dev dependencies (pytest, ruff, mypy)
pip install -e ".[dev]"
```

After installation the `rag-eval` command is available inside your virtual environment.

## Environment Setup

Copy `.env` and fill in the keys for any provider you plan to use.
Ollama requires no API key — just make sure the server is running locally.

```bash
cp .env .env.local
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google GenAI API key |
| `OLLAMA_BASE_URL` | Ollama server URL (default `http://127.0.0.1:11434`) |
| `VLLM_BASE_URL` | vLLM server URL (default `http://localhost:8000`) |
| `RAG_EVAL_CACHE_DIR` | Cache directory (default `.cache/rag_eval`) |

---

## CLI Usage

The `rag-eval` CLI exposes four commands. Run `rag-eval --help` for the full overview.

### Global Options

```
--verbose / -v    Enable debug logging (LLM calls, cache hits, …)
--quiet   / -q    Suppress all output except errors
--version         Print the framework version and exit
```

---

### `rag-eval evaluate` — Run Evaluation

Evaluate a dataset against one or more metrics and generate reports.

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
| `--provider` | `-p` | *(from config)* | Override the LLM provider (`openai`, `ollama`, `anthropic`, `google`, `vllm`). |
| `--max-concurrency` | – | `5` | Maximum concurrent evaluations. Set to `1` for local models. |
| `--dry-run` | – | `false` | Validate config & dataset, then exit without running. |

**Examples:**

```bash
# Quick check with lightweight (no-LLM) metrics only
rag-eval evaluate -d data.json -m token_f1,exactmatch

# Full LLM-judge evaluation with Ollama, sequential
rag-eval evaluate -d data.json -m faithfulness,answerrelevance --max-concurrency 1

# Use a custom config and output only the HTML report
rag-eval evaluate -d data.json -c configs/default.yaml --format html

# Dry-run: validate your dataset and configuration without calling any LLM
rag-eval evaluate -d data.json --dry-run
```

---

### `rag-eval generate` — Synthetic Dataset Generation

Generate a synthetic QA dataset from a source text document using an LLM.

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
| `--documents` | `-d` | *(required)* | Path to the source text file (`.txt`). |
| `--output` | `-o` | `synthetic_dataset.json` | Output path for the generated JSON dataset. |
| `--num-samples` | `-n` | `1` | Number of QA pairs to generate **per chunk**. |
| `--question-types` | `-q` | `factual` | Comma-separated types: `factual`, `multi-hop`, `reasoning`, `comparative`. |
| `--difficulty` | – | `medium` | Question difficulty: `easy`, `medium`, `hard`. |
| `--provider` | `-p` | `ollama` | LLM provider to use for generation. |
| `--model` | `-m` | `qwen2.5:7b` | LLM model to use. |
| `--max-chunks` | – | `10` | Maximum number of document chunks to process (`0` = unlimited). |
| `--max-concurrency` | – | `3` | Maximum concurrent LLM requests. |

**Examples:**

```bash
# Generate 2 factual QA pairs per chunk, easy difficulty
rag-eval generate -d paper.txt -n 2 --difficulty easy

# Multi-hop + reasoning questions using OpenAI
rag-eval generate -d paper.txt -q multi-hop,reasoning -p openai -m gpt-4o

# Process entire document (no chunk limit), save to custom path
rag-eval generate -d corpus.txt --max-chunks 0 -o datasets/eval_set.json
```

---

### `rag-eval report` — Regenerate Reports

Regenerate HTML, CSV, or console reports from a previously saved `report.json`.

```bash
rag-eval report \
  --input output/report.json \
  --output-dir new_reports/ \
  --format html,csv
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *(required)* | Path to an existing `report.json`. |
| `--output-dir` | `-o` | `output_regenerated` | Directory for the regenerated report files. |
| `--format` | `-f` | `html,csv,console` | Comma-separated output formats to generate. |

**Examples:**

```bash
# Regenerate only the console summary from an existing report
rag-eval report -i output/report.json --format console

# Export to all formats in a new directory
rag-eval report -i output/report.json -o results/ --format json,html,csv,console
```

---

### `rag-eval metrics` — List Available Metrics

Print all registered evaluation metrics and their dependency requirements.

```bash
rag-eval metrics
```

Each metric shows whether it requires an **LLM** client, an **Embedding** client,
or runs **standalone** (no external dependencies).

### Available Metrics

| Metric | Class | Requires |
|---|---|---|
| `faithfulness` | `Faithfulness` | LLM |
| `answer_relevance` | `AnswerRelevance` | LLM |
| `context_precision` | `ContextPrecision` | LLM |
| `context_recall` | `ContextRecall` | LLM |
| `semantic_similarity` | `SemanticSimilarity` | Embedding |
| `token_f1` | `TokenF1` | Standalone |
| `exact_match` | `ExactMatch` | Standalone |
| `rouge` | `Rouge` | Standalone |
| `bleu` | `BLEU` | Standalone |
| `bert_score` | `BERTScore` | Standalone |

---

## Python API

The framework can also be used as a library. See `examples/` for full scripts.

```python
from pathlib import Path

from rag_eval.core.config import LLMConfig, EmbeddingConfig
from rag_eval.datasets.loader import load_dataset
from rag_eval.metrics.faithfulness import Faithfulness
from rag_eval.metrics.f1 import TokenF1
from rag_eval.metrics.semantic_similarity import SemanticSimilarity
from rag_eval.pipeline.evaluator import Evaluator
from rag_eval.reports.generator import ReportGenerator
from rag_eval.utils.llm import LLMClient
from rag_eval.utils.embeddings import EmbeddingClient

# 1. Load dataset
samples = load_dataset("examples/synthetic_dataset.json")

# 2. Create clients
llm_client = LLMClient(config=LLMConfig())               # defaults to Ollama
embed_client = EmbeddingClient(config=EmbeddingConfig())   # sentence-transformers

# 3. Choose metrics
metrics = [
    Faithfulness(llm_client=llm_client),
    SemanticSimilarity(embed_client=embed_client),
    TokenF1(),
]

# 4. Run evaluation
evaluator = Evaluator(metrics=metrics)
report = evaluator.evaluate(samples, max_concurrency=1)

# 5. Generate reports
generator = ReportGenerator(report)
generator.to_console()
generator.to_html(Path("output/report.html"))
generator.to_json(Path("output/report.json"))
generator.to_csv(Path("output/report.csv"))
```

## Configuration

The default configuration lives in `configs/default.yaml`. Settings can be
overridden via CLI flags or by passing a custom YAML file with `--config`.

```yaml
llm:
  provider: ollama          # ollama | openai | anthropic | google | vllm
  model: llama3.2
  temperature: 0.0
  max_tokens: 1024

embedding:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2

cache:
  enabled: true
  directory: .cache/rag_eval

metrics:
  - faithfulness
  - answer_relevance
  - context_precision
  - context_recall
  - semantic_similarity

output_dir: ./output
max_concurrency: 5
```

## Testing

```bash
# Run the full test suite
pytest

# With coverage
pytest --cov=rag_eval --cov-report=term-missing
```

## Documentation

Extended documentation is available in the `docs/` directory:

- [CLI Reference](docs/cli.md) — full command reference with all options
- [Synthetic Generation](docs/SYNTHETIC_GENERATION.md) — methodology behind QA pair generation
- [System Overview](docs/SYSTEM_OVERVIEW.md) — architecture and design decisions
- [Project Roadmap](docs/PROJECT_ROADMAP.md) — planned features and progress

## License

MIT — see [LICENSE](LICENSE) for details.
