# RAG Evaluation Framework — System Overview

## 1. Introduction

### 1.1 Purpose

This framework provides a **modular, extensible toolkit for evaluating Retrieval-Augmented Generation (RAG) systems**. It is designed to support experimental research by offering a unified interface for measuring the quality of both the *retrieval* and *generation* stages of a RAG pipeline across a wide range of metrics, datasets, and LLM providers.

### 1.2 Context

The framework is built as part of a **thesis project** focused on evaluating and comparing RAG systems. It serves as the experimental backbone — enabling reproducible evaluations, side-by-side configuration comparisons, and automated report generation. While it supports the thesis research, the framework is designed to be general-purpose and reusable.

### 1.3 Core Principles

| Principle | Description |
|---|---|
| **Modularity** | Every component (metrics, datasets, LLM clients, reporters) is a self-contained, swappable module |
| **Extensibility** | Adding a new metric or LLM provider requires implementing a single abstract class |
| **Reproducibility** | All evaluation runs are fully configured via YAML files and produce deterministic, exportable results |
| **Async-first** | Built on Python's `asyncio` for concurrent LLM calls and efficient batch evaluation |
| **Multi-provider** | Supports OpenAI, Anthropic, Google, and local models (Ollama, vLLM) as LLM judges |

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              User Interface                              │
│                          (Python API + CLI)                              │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Pipeline / Evaluator                             │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐              │
│  │   Datasets    │───▶│   Evaluator   │───▶│    Reports     │              │
│  │   (loader,    │    │  (orchestrate │    │  (JSON, HTML,  │              │
│  │   synthetic)  │    │   metrics)    │    │   console)     │              │
│  └──────────────┘    └──────┬───────┘    └───────────────┘              │
│                             │                                            │
│                     ┌───────┴────────┐                                   │
│                     │    Metrics      │                                   │
│                     │                │                                   │
│                     │  ┌───────────┐ │                                   │
│                     │  │ LLM-Judge │ │                                   │
│                     │  ├───────────┤ │                                   │
│                     │  │ NLP-Based │ │                                   │
│                     │  ├───────────┤ │                                   │
│                     │  │ Embedding │ │                                   │
│                     │  └───────────┘ │                                   │
│                     └───────┬────────┘                                   │
│                             │                                            │
│                     ┌───────┴────────┐                                   │
│                     │     Utils       │                                   │
│                     │  (LLM client,  │                                   │
│                     │   embeddings,  │                                   │
│                     │   text proc.)  │                                   │
│                     └────────────────┘                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Layer Summary

| Layer | Package | Responsibility |
|---|---|---|
| **Interface** | CLI + Python API | Entry points for users: run evaluations, configure settings, view results |
| **Pipeline** | `rag_eval.pipeline` | Orchestrates the full evaluation flow: load data → run metrics → produce reports |
| **Metrics** | `rag_eval.metrics` | Implements individual evaluation metrics (LLM-judge, NLP, embedding-based) |
| **Datasets** | `rag_eval.datasets` | Load/validate datasets and generate synthetic evaluation data |
| **Reports** | `rag_eval.reports` | Aggregate results and export in various formats |
| **Core** | `rag_eval.core` | Shared data models (`TestSample`, `EvalResult`) and configuration |
| **Utils** | `rag_eval.utils` | Cross-cutting utilities: LLM clients, embedding clients, text processing |

---

## 3. Core Data Models

All data flows through a small set of **Pydantic models** defined in `rag_eval.core.types`:

### 3.1 TestSample

Represents a single evaluation instance — the atomic unit of evaluation.

```python
class TestSample(BaseModel):
    question: str           # The user query
    answer: str             # Generated answer from the RAG system
    ground_truth: str       # Reference / expected answer
    contexts: list[str]     # Retrieved context chunks
    metadata: dict          # Arbitrary metadata (source, config, timestamps...)
```

### 3.2 EvalResult

The output of running a single metric on a single sample.

```python
class EvalResult(BaseModel):
    metric_name: str        # e.g. "faithfulness"
    score: float            # Normalized score in [0.0, 1.0]
    reason: str             # LLM-generated explanation for the score
    metadata: dict          # Additional info (token counts, latency, etc.)
```

### 3.3 EvalReport

Aggregated results across all samples and metrics, with summary statistics.

```python
class EvalReport(BaseModel):
    results: list[EvalResult]
    summary: dict[str, float]    # metric_name → average score
```

---

## 4. Metrics System

Metrics are the heart of the framework. Every metric extends `BaseMetric`:

```python
class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def score(self, sample: TestSample) -> EvalResult: ...

    async def score_batch(self, samples: list[TestSample]) -> list[EvalResult]: ...
```

### 4.1 Metric Categories

The framework provides three categories of metrics:

#### A. LLM-as-Judge Metrics

These use an LLM to evaluate quality by reasoning over the inputs. They are the most powerful but also the most expensive.

| Metric | What it measures | Inputs used |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the contexts? | `answer`, `contexts` |
| **Answer Relevance** | Does the answer actually address the question? | `question`, `answer` |
| **Context Precision** | Are the retrieved contexts relevant (signal vs. noise)? | `question`, `contexts`, `ground_truth` |
| **Context Recall** | Do the contexts cover all information in the ground truth? | `contexts`, `ground_truth` |

**Approach**: Each metric uses structured prompts to extract judgments from the LLM. For example, Faithfulness works in two steps:
1. **Claim extraction** — decompose the answer into atomic claims
2. **Verification** — for each claim, check if it's supported by any context chunk

#### B. Traditional NLP Metrics

Classical reference-based metrics that compare the generated answer against the ground truth using string/token overlap.

| Metric | Description |
|---|---|
| **BLEU** | N-gram precision between answer and ground truth |
| **ROUGE** (ROUGE-1, ROUGE-2, ROUGE-L) | Recall-oriented n-gram overlap |
| **F1 Score** | Token-level F1 between answer and ground truth |
| **Exact Match** | Binary: does the answer exactly match the ground truth? |

These are deterministic, fast, and free (no LLM calls).

#### C. Embedding-Based Metrics

Use pre-trained embedding models to compare semantic meaning.

| Metric | Description |
|---|---|
| **Semantic Similarity** | Cosine similarity between answer and ground truth embeddings |
| **BERTScore** | Token-level embedding similarity using contextual embeddings |

---

### 4.2 Metric Registry

Metrics are registered by name and can be instantiated from configuration:

```python
# Config-driven usage
config = EvalConfig.from_yaml("configs/default.yaml")
# config.metrics = ["faithfulness", "answer_relevance", "bleu", "rouge_l"]
# The evaluator resolves names → metric instances via the registry
```

---

## 5. Dataset System

### 5.1 Dataset Loading

The framework supports loading evaluation datasets from:

- **JSON** — array of sample objects
- **JSONL** — one sample per line (streaming-friendly)
- **CSV** — tabular format with configurable column mapping
- **HuggingFace Datasets** — direct loading from the HF Hub

Expected schema:
```json
{
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "ground_truth": "Paris",
    "contexts": ["France is a country in Europe. Its capital is Paris.", "..."]
}
```

### 5.2 Built-in Benchmark Support

The framework includes loaders and adapters for standard benchmarks:

| Category | Datasets |
|---|---|
| **Standard QA** | HotpotQA, Natural Questions (NQ), TriviaQA, SQuAD 2.0 |
| **RAG-specific** | RGB, RECALL, CRUD-RAG |

Each benchmark has a dedicated adapter that maps its native format into `TestSample` objects.

### 5.3 Synthetic Data Generation

A central feature of the framework. The `SyntheticGenerator` creates evaluation datasets from raw source documents using LLMs:

```
Source Documents → [Chunk] → [LLM: Generate QA pairs] → TestSample[]
```

**Capabilities**:
- **Question type control**: factual, multi-hop, reasoning, comparative
- **Difficulty levels**: simple (single-context) to hard (requires synthesis across chunks)
- **Distractor injection**: generate plausible but irrelevant contexts to test retrieval precision
- **Ground truth generation**: automatically generate reference answers from the source
- **Metadata annotation**: tag each sample with source document, chunk IDs, question type

---

## 6. LLM Client System

The `LLMClient` provides a unified interface across multiple providers:

```
┌─────────────┐
│  LLMClient   │  ← Unified async interface
├─────────────┤
│  OpenAI      │  GPT-4o, GPT-4o-mini, o3, ...
│  Anthropic   │  Claude Sonnet, Opus, Haiku, ...
│  Google      │  Gemini 2.5 Pro, Flash, ...
│  Local       │  Ollama, vLLM, HF Transformers
└─────────────┘
```

### 6.1 Key Features

- **Provider abstraction**: switch between providers by changing a config value
- **Retry logic**: exponential backoff with jitter for rate limits
- **Token tracking**: count input/output tokens for cost estimation
- **Caching**: optional response caching to avoid redundant LLM calls during development
- **Concurrency control**: configurable max concurrent requests per provider

### 6.2 Embedding Client

A parallel `EmbeddingClient` supports embedding generation for semantic metrics:
- OpenAI `text-embedding-3-small/large`
- Sentence Transformers (local)
- Custom embedding endpoints

---

## 7. Pipeline / Evaluator

The `Evaluator` is the orchestration layer that ties everything together:

```
1. Load config (YAML / programmatic)
2. Load dataset (file / benchmark / synthetic)
3. Instantiate metrics (from config or explicit list)
4. Run evaluation:
   for each sample:
       for each metric:
           result = await metric.score(sample)
5. Aggregate results into EvalReport
6. Export report (JSON / HTML / console)
```

### 7.1 Execution Modes

| Mode | Description |
|---|---|
| **Sequential** | Evaluate one sample at a time (simple, debuggable) |
| **Batch** | Process samples in configurable batch sizes |
| **Concurrent** | Run multiple metric evaluations in parallel using `asyncio` |

### 7.2 Configuration Comparison (Nice-to-have)

Support running the same dataset through multiple RAG configurations and producing a comparative report:

```python
evaluator.compare(
    samples_a=samples_config_a,
    samples_b=samples_config_b,
    metrics=[Faithfulness(), AnswerRelevance()],
)
# → Comparative table with per-metric deltas
```

---

## 8. Reporting System

### 8.1 Output Formats

| Format | Use Case |
|---|---|
| **Console** (Rich) | Quick inspection during development |
| **JSON** | Machine-readable, for downstream processing |
| **HTML** | Visual report with charts for thesis/presentation |
| **CSV** | Import into spreadsheets or data analysis tools |

### 8.2 Report Contents

- Per-sample scores for every metric
- Aggregate statistics (mean, median, std, min, max per metric)
- Score distributions (histograms)
- Failure case analysis (lowest-scoring samples)
- Cost tracking (tokens used, estimated cost)

---

## 9. CLI Interface

The framework provides a command-line interface for common operations:

```bash
# Run a full evaluation
rag-eval evaluate --config configs/default.yaml --dataset data/eval.json

# Generate synthetic data
rag-eval generate --documents docs/ --output data/synthetic.json --num-samples 100

# View results
rag-eval report --input output/results.json --format html

# List available metrics
rag-eval metrics --list
```

---

## 10. LangChain Integration

For users with existing LangChain RAG pipelines, the framework provides an adapter:

```python
from rag_eval.integrations.langchain import LangChainAdapter

# Wrap your existing chain
adapter = LangChainAdapter(chain=my_rag_chain)

# Automatically extracts question, answer, and contexts from the chain
samples = adapter.run(questions=["What is X?", "How does Y work?"])

# Evaluate normally
report = evaluator.evaluate(samples)
```

The adapter hooks into LangChain's callback system to capture:
- The user query
- Retrieved documents (with metadata)
- The final generated answer

---

## 11. Package Structure

```
src/rag_eval/
├── __init__.py
├── core/                       # Foundation layer
│   ├── types.py                # TestSample, EvalResult, EvalReport
│   ├── config.py               # LLMConfig, EvalConfig, YAML loading
│   └── registry.py             # Metric/provider registry
├── metrics/                    # All evaluation metrics
│   ├── base.py                 # BaseMetric ABC
│   ├── faithfulness.py         # LLM-judge: answer grounding
│   ├── relevance.py            # LLM-judge: answer relevance
│   ├── context_precision.py    # LLM-judge: retrieval precision
│   ├── context_recall.py       # LLM-judge: retrieval recall
│   ├── semantic_similarity.py  # Embedding: cosine similarity
│   ├── bert_score.py           # Embedding: BERTScore
│   ├── bleu.py                 # NLP: BLEU score
│   ├── rouge.py                # NLP: ROUGE variants
│   ├── f1.py                   # NLP: token-level F1
│   └── exact_match.py          # NLP: exact match
├── datasets/                   # Data ingestion
│   ├── loader.py               # JSON/JSONL/CSV loading
│   ├── synthetic.py            # LLM-based QA generation
│   └── benchmarks/             # Built-in benchmark adapters
│       ├── hotpotqa.py
│       ├── natural_questions.py
│       ├── triviaqa.py
│       └── squad.py
├── pipeline/                   # Orchestration
│   ├── evaluator.py            # Main Evaluator class
│   └── comparator.py           # Multi-config comparison
├── integrations/               # External framework adapters
│   └── langchain.py            # LangChain adapter
├── reports/                    # Result export
│   └── generator.py            # Multi-format report generation
├── cli/                        # Command-line interface
│   └── main.py                 # CLI entry point (click/typer)
└── utils/                      # Shared utilities
    ├── llm.py                  # Multi-provider LLM client
    ├── embeddings.py           # Embedding client
    ├── text.py                 # Text processing helpers
    └── cache.py                # Response caching
```

---

## 12. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data models | Pydantic v2 |
| Async runtime | asyncio |
| LLM clients | openai, anthropic, google-genai SDKs |
| Local models | Ollama, vLLM |
| Embeddings | sentence-transformers, OpenAI |
| NLP metrics | nltk (BLEU), rouge-score, bert-score |
| CLI | Typer + Rich |
| Data | pandas, numpy |
| ML utilities | scikit-learn |
| Config | PyYAML |
| Testing | pytest, pytest-asyncio |
| Linting | Ruff, mypy |
