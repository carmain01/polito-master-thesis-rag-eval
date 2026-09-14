# Command Line Interface (CLI) Reference

The `rag-eval` CLI provides direct access to every core feature of the framework — dataset evaluation, synthetic data generation, report visualization, and metric discovery — without writing boilerplate Python code.

---

## 1. Installation & Verification

When installed in editable mode inside your virtual environment, the CLI binary is immediately available:

```bash
# Using uv or pip
uv pip install -e .

# Verify installation
rag-eval --version
rag-eval --help
```

---

## 2. Global Options

These flags apply to all commands:

| Option | Short | Description |
|---|---|---|
| `--verbose` | `-v` | Enable **debug** logging (shows full LLM prompts, token counts, cache hits). |
| `--quiet` | `-q` | Suppress all output except critical errors. |
| `--version` | – | Print the framework version (`rag-eval 0.1.0`) and exit. |

---

## 3. Command Reference

### 3.1 `evaluate`

Runs evaluation on a dataset using selected metrics, LLM judges, and export formats.

```bash
rag-eval evaluate \
  --dataset data/golden_dataset_pes_compiled.json \
  --metrics pes,faithfulness \
  --provider azure \
  --model gpt-5-nano \
  --reasoning-effort high \
  --output output/rag_pes_azure/ \
  --format console,json,html,csv
```

#### Options & Flags

| Option | Short | Default | Description |
|---|---|---|---|
| `--dataset` | `-d` | *(required)* | Path to the evaluation dataset (`.json`, `.jsonl`, or `.csv`). |
| `--config` | `-c` | `None` | Path to a YAML configuration file (e.g. `configs/default.yaml`). |
| `--output` | `-o` | `output` | Directory where report files will be written. |
| `--metrics` | `-m` | *(from config)* | Comma-separated list of metric names (e.g. `pes,faithfulness,context_precision`). |
| `--format` | `-f` | `json,html,csv,console` | Comma-separated output formats to generate. |
| `--provider` | `-p` | *(from config)* | LLM provider override: `azure`, `openai`, `anthropic`, `google`, `ollama`, `vllm`. |
| `--model` | – | *(from config)* | LLM model override (e.g. `gpt-5-nano`, `gpt-4o`, `llama3.2`, `qwen2.5:7b`). |
| `--reasoning-effort`| – | *(from config)* | Reasoning level for reasoning models: `low`, `medium`, `high`. |
| `--max-concurrency` | – | `5` | Max parallel async requests. Set to `1` for local models (Ollama). |
| `--dry-run` | – | `false` | Validates configuration, dataset schema, and metric dependencies without making LLM calls. |

#### Evaluation Examples

```bash
# 1. Quick sanity check with lightweight deterministic metrics (no LLM costs)
rag-eval evaluate -d data/eval.json -m exact_match,f1,bleu

# 2. Complete Pedagogical Evaluation Score (PES) run on Azure OpenAI
rag-eval evaluate \
  -d data/golden_dataset_pes_compiled.json \
  -m pes,faithfulness,linguistic_adaptation,uptake,no_immediate_disclosure \
  -p azure \
  --model gpt-5-nano \
  --reasoning-effort high \
  -o output/pes_run/

# 3. Local RAG Triad evaluation using Ollama (sequential to prevent GPU OOM)
rag-eval evaluate \
  -d data/golden_dataset_compiled.json \
  -m faithfulness,relevance,context_precision,context_recall \
  -p ollama \
  --model llama3.2 \
  --max-concurrency 1 \
  -o output/rag_triad_local/

# 4. Dry-run to inspect configuration resolution
rag-eval evaluate -d data/golden_dataset_pes_compiled.json -c configs/default.yaml --dry-run
```

---

### 3.2 `generate`

Generates synthetic Question-Answer pairs from reference documents, with automatic semantic chunking, distractor context injection, and simulated RAG imperfections.

```bash
rag-eval generate \
  --documents source_document.txt \
  --output data/synthetic_dataset.json \
  --num-samples 2 \
  --question-types factual,multi-hop,reasoning \
  --difficulty hard \
  --provider ollama \
  --model qwen2.5:7b \
  --max-chunks 15
```

#### Options & Flags

| Option | Short | Default | Description |
|---|---|---|---|
| `--documents` | `-d` | *(required)* | Path to source text file (`.txt`, UTF-8 encoded). |
| `--output` | `-o` | `synthetic_dataset.json`| Path for the generated JSON dataset. |
| `--num-samples` | `-n` | `1` | Number of QA pairs to generate **per chunk**. |
| `--question-types` | `-q` | `factual` | Comma-separated types: `factual`, `multi-hop`, `reasoning`, `comparative`. |
| `--difficulty` | – | `medium` | Question complexity level: `easy`, `medium`, `hard`. |
| `--provider` | `-p` | `ollama` | LLM provider to use for dataset creation (`ollama`, `openai`, `azure`, etc.). |
| `--model` | `-m` | `qwen2.5:7b` | Model name. |
| `--max-chunks` | – | `10` | Maximum number of chunks to process (`0` for unlimited). |
| `--max-concurrency` | – | `3` | Maximum concurrent LLM generation calls. |

> **Note**: Input files must be plain text (`.txt`). For PDFs, extract text beforehand using `pdftotext` or Python's `pypdf`.

---

### 3.3 `report`

Regenerates visual HTML dashboards, CSV tables, or console views from an existing `report.json` without rerunning evaluation metrics.

```bash
rag-eval report \
  --input output/rag_triad_azure/report.json \
  --output-dir output/regenerated_dashboard/ \
  --format html,csv,console
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *(required)* | Path to an existing `report.json` output file. |
| `--output-dir` | `-o` | `output_regenerated`| Target directory for newly generated report files. |
| `--format` | `-f` | `html,csv,console` | Formats to regenerate. |

---

### 3.4 `metrics`

Lists all metrics registered in the `MetricRegistry`, displaying their associated Python classes and runtime requirements.

```bash
rag-eval metrics
```

#### Sample Output

```text
Available Metrics:

  bert_score                BERTScore              (Embedding)
  bleu                      BLEU                   (standalone)
  context_precision         ContextPrecision       (LLM)
  context_recall            ContextRecall          (LLM)
  exact_match               ExactMatch             (standalone)
  f1                        TokenF1                (standalone)
  faithfulness              Faithfulness           (LLM)
  linguistic_adaptation     LinguisticAdaptation   (standalone)
  no_immediate_disclosure   NoImmediateDisclosure  (LLM)
  pes                       PES                    (LLM, Embedding)
  relevance                 Relevance              (LLM)
  rouge_1                   Rouge1                 (standalone)
  rouge_2                   Rouge2                 (standalone)
  rouge_l                   RougeL                 (standalone)
  semantic_similarity       SemanticSimilarity     (Embedding)
  state                     State                  (LLM)
  uptake                    Uptake                 (Embedding)
```

- **`LLM`**: Requires an active LLM provider client (`--provider`).
- **`Embedding`**: Requires an active embedding client (local `sentence-transformers` or OpenAI embeddings).
- **`standalone`**: Pure Python/NLP calculations (runs instantly with zero external dependencies).
