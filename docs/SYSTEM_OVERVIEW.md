# RAG Evaluation Framework — System Overview

## 1. Introduction

### 1.1 Purpose

The **`rag_eval`** framework provides a **modular, extensible, and mathematically grounded toolkit for evaluating Retrieval-Augmented Generation (RAG) systems**. Designed to support rigorous academic research and enterprise audit, it unifies the measurement of retrieval precision, factual grounding, and answer quality across multiple LLM providers, benchmark datasets, and evaluation metrics.

Beyond standard factual evaluation, the framework introduces the **Pedagogical Evaluation Score (PES)**, a novel composite metric specifically designed to evaluate conversational AI tutors, socratic scaffolding, and educational dialogue systems in e-learning environments.

### 1.2 Context & Thesis Background

Developed as part of a Master's Thesis in Computer Engineering at Politecnico di Torino, the framework serves as the experimental backbone for:
- Evaluating enterprise RAG deployments across institutional knowledge bases and continuous learning platforms.
- Validating the pedagogical quality of LLMs in tutorial interactions.
- Benchmarking state-of-the-art models (OpenAI GPT-4o, Azure GPT-5-nano reasoning models, Meta LLaMA 3.1/3.2, Qwen 2.5) on real-world and synthetic datasets.

### 1.3 Core Architectural Principles

| Principle | Description |
|---|---|
| **Modularity** | Every component (metrics, datasets, LLM clients, reports) is a self-contained, swappable module registered in dynamic registries. |
| **Pedagogical Awareness** | Native support for multi-turn pedagogical evaluation (scaffolding vs. spoiling, uptake, readability, and socratic dialogue). |
| **Reproducibility** | Full configuration via YAML/CLI with deterministic seed controls, structured JSON schemas, and audit logs. |
| **Fault Tolerance & Checkpointing** | Automatic batch checkpointing allowing long evaluation runs to be safely interrupted and resumed without data loss. |
| **Multi-Provider & Async-First** | Built on Python's `asyncio` with concurrent execution across commercial cloud engines (Azure, OpenAI, Anthropic, Google) and local runtimes (Ollama, vLLM). |

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              User Interface                              │
│                 (Python API + CLI: `rag-eval evaluate / generate`)        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Pipeline / Evaluator Layer                       │
│                                                                          │
│  ┌──────────────┐    ┌────────────────────────────────┐    ┌───────────┐ │
│  │   Datasets   │───▶│   Evaluator & Batch Runner     │───▶│  Reports  │ │
│  │ (MathDial,   │    │  (Concurrency, Checkpointing,  │    │  (HTML,   │ │
│  │  MR-Bench,   │    │   DiskCache, Error Handling)   │    │   JSON,   │ │
│  │  Golden JSON)│    └──────────────┬─────────────────┘    │   CSV)    │ │
│  └──────────────┘                   │                      └───────────┘ │
│                                     │                                    │
│                 ┌───────────────────┴───────────────────┐                │
│                 │            Metrics Engine             │                │
│                 │                                       │                │
│                 │  ┌─────────────────────────────────┐  │                │
│                 │  │ PES Framework (Pedagogical)     │  │                │
│                 │  │  • Level 0: Faithfulness Gate   │  │                │
│                 │  │  • Level 1: Pedagogical State   │  │                │
│                 │  │  • Level 2: Submetrics (M1..M3) │  │                │
│                 │  │  • Level 3: Dynamic GeoMean     │  │                │
│                 │  ├─────────────────────────────────┤  │                │
│                 │  │ RAG Triad (LLM-as-a-Judge)      │  │                │
│                 │  │  • Faithfulness, Relevance      │  │                │
│                 │  │  • Context Precision / Recall   │  │                │
│                 │  ├─────────────────────────────────┤  │                │
│                 │  │ Classical NLP & Embedding       │  │                │
│                 │  │  • BLEU, ROUGE, F1, Exact Match │  │                │
│                 │  │  • BERTScore, Cosine Similarity │  │                │
│                 │  └─────────────────────────────────┘  │                │
│                 └───────────────────┬───────────────────┘                │
│                                     │                                    │
│                 ┌───────────────────┴───────────────────┐                │
│                 │           Cross-Cutting Utils         │                │
│                 │  ┌─────────────────────────────────┐  │                │
│                 │  │ Multi-Provider LLM Client       │  │                │
│                 │  │ (Azure, OpenAI, Claude, Gemini, │  │                │
│                 │  │  Ollama, vLLM with Reasoning)   │  │                │
│                 │  ├─────────────────────────────────┤  │                │
│                 │  │ Embeddings & Text Processing    │  │                │
│                 │  │ (sentence-transformers, MiniLM) │  │                │
│                 │  └─────────────────────────────────┘  │                │
│                 └───────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Data Models

The framework relies on strict Pydantic v2 schemas defined in `rag_eval.core.types`:

### 3.1 `TestSample`
Represents a single evaluation unit (query, answer, context, references, and metadata):
```python
class TestSample(BaseModel):
    id: str | None = None
    question: str                  # User query or student prompt
    answer: str                    # Answer generated by the RAG system
    ground_truth: str | None       # Expected reference answer or pedagogical target
    contexts: list[str]            # Retrieved context chunks
    metadata: dict[str, Any] = {}  # Source document, difficulty, pedagogical state, tokens
```

### 3.2 `EvalResult`
The output of a single metric evaluated on a single sample:
```python
class EvalResult(BaseModel):
    metric_name: str               # e.g., "pes", "faithfulness", "uptake"
    score: float                   # Normalized score in [0.0, 1.0]
    reason: str | None = None      # Qualitative explanation generated by the LLM judge
    metadata: dict[str, Any] = {}  # Sub-scores, token counts, latency, weights used
```

### 3.3 `EvalReport`
Aggregated results across the entire dataset with statistical rollups:
```python
class EvalReport(BaseModel):
    results: list[EvalResult]
    summary: dict[str, float]      # metric_name -> mean score
    metadata: dict[str, Any] = {}  # Execution duration, model name, provider, timestamp
```

---

## 4. The Metrics System

Metrics extend the abstract base class `BaseMetric`:
```python
class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def score(self, sample: TestSample) -> EvalResult: ...

    async def score_batch(self, samples: list[TestSample]) -> list[EvalResult]: ...
```

### 4.1 The Pedagogical Evaluation Score (PES) Framework

The **PES** is a multi-tier composite metric designed to assess whether a conversational RAG system acts as an effective tutor rather than a passive answering engine.

#### Level 0: Faithfulness Gate
An educational answer containing factual errors or hallucinations is pedagogically unacceptable. If `Faithfulness` drops below a configurable threshold (default $\tau = 0.50$):
$$\text{PES} = 0.0$$

#### Level 1: Pedagogical State Classification
The `State` classifier identifies the active pedagogical scenario $S$:
- **State A (Concept Teaching):** Introducing new domain concepts.
- **State B (Error Remediation):** Identifying and rectifying student misconceptions.
- **State C (Socratic Assessment):** Guiding the student toward deductive reasoning through probing questions.

#### Level 2: Sub-Metrics Evaluation
- **$M_1$ — Uptake (`Uptake`):** Evaluates how effectively the assistant incorporates student context, previous conversational turns, and misconceptions (via NUC-BERT and semantic similarity).
- **$M_2$ — Linguistic Adaptation (`LinguisticAdaptation`):** Evaluates lexical readability (Flesch-Kincaid / Gulpease), tentative phrasing (*tentativeness*), and constructive tone.
- **$M_3$ — No Immediate Disclosure (`NoImmediateDisclosure`):** Evaluates scaffolding quality vs. spoiling the solution, penalizing premature disclosure of the complete answer.

#### Level 3: Weighted Geometric Mean
The final score is synthesized via a state-dependent weighted geometric mean:
$$\text{PES}(S) = \left( \prod_{i=1}^{3} M_i^{w_i(S)} \right)^{\frac{1}{\sum_{i=1}^3 w_i(S)}}$$

| State $S$ | Intent Label | $w_1$ (Uptake) | $w_2$ (Adaptation) | $w_3$ (No Disclosure) |
|---|---|:---:|:---:|:---:|
| **A** | Concept Teaching | $0.35$ | $0.35$ | $0.30$ |
| **B** | Error Remediation | $0.00$ | $0.30$ | $0.70$ |
| **C** | Socratic Assessment | $0.00$ | $0.25$ | $0.75$ |
| **Other** | Default / General | $0.33$ | $0.33$ | $0.34$ |

---

### 4.2 Standard RAG Triad Metrics (LLM-as-a-Judge)

Implemented with robust prompting, atomic decomposition, and reasoning engines:

1. **Faithfulness (`faithfulness.py`):** Decomposes the answer into atomic statements and verifies each against the retrieved contexts.
2. **Answer Relevance (`relevance.py`):** Measures whether the answer directly addresses the core intent of the question.
3. **Context Precision (`context_precision.py`):** Evaluates whether relevant information is concentrated at top retrieval ranks (signal-to-noise ratio).
4. **Context Recall (`context_recall.py`):** Verifies that all ground-truth facts are captured within the retrieved contexts.

---

### 4.3 Deterministic & Embedding-Based NLP Metrics

- **N-gram Overlap:** `BLEU` (via NLTK), `ROUGE-1`, `ROUGE-2`, `ROUGE-L` (via `rouge-score`).
- **Exact & Token-Level:** `Exact Match`, `Token F1`.
- **Semantic Representation:** `Semantic Similarity` (Cosine distance on sentence-transformers embeddings), `BERTScore` (Contextual token alignment).

---

## 5. LLM Client & Provider Architecture

The `LLMClient` provides a unified async interface across cloud and local engines:

```
┌─────────────────────────────────────────────────────────────┐
│                          LLMClient                          │
├─────────────────────────────────────────────────────────────┤
│  • Azure OpenAI Provider  (GPT-5-nano, high reasoning)      │
│  • OpenAI Provider        (GPT-4o, GPT-4o-mini, o3-mini)    │
│  • Anthropic Provider     (Claude 3.5 Sonnet, Haiku)        │
│  • Google Provider        (Gemini 2.5 Pro, Flash)           │
│  • Ollama Provider        (LLaMA 3.1/3.2, Qwen 2.5 local)   │
│  • vLLM Provider          (High-throughput vLLM cluster)    │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 Azure OpenAI & Reasoning Models
The `AzureOpenAIProvider` supports:
- Authentication via Azure API keys or Microsoft Entra ID (`DefaultAzureCredential`).
- Deployment name mapping and configurable API versions (`2024-02-15-preview`, `2024-08-01-preview`, etc.).
- Reasoning effort control (`reasoning_effort: "high"`, `"medium"`, `"low"`) and extended token ceilings (`max_completion_tokens: 32768`) for deep-reasoning models.

### 5.2 Resiliency, Caching & Concurrency
- **DiskCache (`utils/cache.py`):** Persistent disk cache keyed by prompt, system message, provider, model, and temperature to avoid redundant API spend.
- **Backoff & Jitter:** Exponential retry logic for HTTP 429 (Rate Limits) and 5xx transient server errors.
- **Concurrency Limiter:** Per-provider `asyncio.Semaphore` preventing rate-limit saturation.

---

## 6. Dataset Ingestion & Benchmarks

### 6.1 Supported Formats
- Standard JSON arrays of `TestSample`.
- Streaming JSON Lines (`.jsonl`).
- Comma-Separated Values (`.csv`) with automatic column mapping.

### 6.2 Specialized Tutoring & Research Loaders
- **`MathDialLoader` (`datasets/mathdial_loader.py`):** Loads teacher-student mathematical dialogues, mapping teacher moves (*probing*, *focus*, *telling*) and conversational state to evaluation samples.
- **`MRBenchLoader` (`datasets/mrbench_loader.py`):** Ingests the MRBench v3 dataset for multi-model reasoning comparisons.
- **Standard QA Benchmarks:** Built-in adapters for HotpotQA, Natural Questions, TriviaQA, SQuAD 2.0, RGB, and RECALL.

### 6.3 Synthetic Data Generator (`SyntheticDataGenerator`)
Enables zero-ground-truth evaluation by extracting semantic chunks from text, injecting distractor contexts, and stochastically simulating realistic RAG behavior (grounded answers, hallucinations, and unanswerability).

---

## 7. Pipeline, Evaluator & Fault Tolerance

The `Evaluator` orchestrates data loading, metric instantiation, execution, and reporting:

```
Config / CLI ──▶ Load Samples ──▶ Batch Evaluation Loop ──▶ Checkpoint Save ──▶ Report Generator
```

### 7.1 Fault-Tolerant Checkpointing
For large-scale evaluations (such as the 2,475 samples of MRBench taking over 50 hours of compute), the pipeline periodically dumps batch checkpoints into `.rag_eval_cache/`. In case of network interruption or hardware failure, the evaluator resumes seamlessly from the last completed batch.

### 7.2 Multi-Configuration Comparator
The `Comparator` (`pipeline/comparator.py`) supports side-by-side comparative analysis of different RAG pipelines (e.g., Naive RAG vs. Advanced RAG with reranking), computing delta metrics and statistical distributions.

---

## 8. Reporting & Visualization

Results are exported via `ReportGenerator`:
- **Interactive HTML Dashboard:** Features Chart.js visualizations, metric histograms, radar comparison charts, and per-sample drill-downs.
- **Machine-Readable JSON:** Full evaluation payload including prompt tokens, reasoning tokens, latency, and judgment explanations.
- **CSV Export:** Tabular summary suitable for statistical tools (R, Pandas, SPSS).
- **Rich Console Tables:** Color-coded terminal summaries for rapid CLI debugging.

---

## 9. Package Structure

```
src/rag_eval/
├── __init__.py
├── core/                         # Core abstractions & configuration
│   ├── config.py                 # EvalConfig, LLMConfig, EmbeddingConfig
│   ├── registry.py               # MetricRegistry with autodiscovery
│   └── types.py                  # TestSample, EvalResult, EvalReport
├── metrics/                      # Metric implementations
│   ├── base.py                   # BaseMetric abstract base class
│   ├── pes.py                    # Composite Pedagogical Evaluation Score
│   ├── no_immediate_disclosure.py# M3: Scaffolding vs. spoiling metric
│   ├── linguistic_adaptation.py  # M2: Readability, tentativeness, critique
│   ├── uptake.py                 # M1: Conversational context elaboration
│   ├── state.py                  # Pedagogical state classifier (A, B, C)
│   ├── faithfulness.py           # Atomic claim verification (RAG Triad)
│   ├── relevance.py              # Query-response alignment (RAG Triad)
│   ├── context_precision.py      # Signal-to-noise ranking (RAG Triad)
│   ├── context_recall.py         # Ground truth coverage (RAG Triad)
│   ├── semantic_similarity.py    # Embedding cosine similarity
│   ├── bert_score.py             # Contextual token alignment
│   ├── bleu.py                   # BLEU-1 to BLEU-4
│   ├── rouge.py                  # ROUGE-1, ROUGE-2, ROUGE-L
│   ├── f1.py                     # Token-level F1 score
│   └── exact_match.py            # Exact string matching
├── datasets/                     # Data ingestion & generation
│   ├── loader.py                 # Generic JSON/JSONL/CSV loader
│   ├── mathdial_loader.py        # MathDial tutorial dialogue loader
│   ├── mrbench_loader.py         # MRBench v3 dataset loader
│   ├── synthetic.py              # Semantic chunking & QA generator
│   └── benchmarks/               # Adapters for standard QA benchmarks
├── pipeline/                     # Execution engine
│   ├── evaluator.py              # Orchestrator with batch checkpointing
│   └── comparator.py             # A/B configuration comparison
├── reports/                      # Export engines
│   ├── generator.py              # JSON, HTML, CSV, Console exports
│   ├── templates/                # Jinja2 HTML dashboard templates
│   └── vendor/                   # Embedded Bootstrap and Chart.js
├── cli/                          # Command-line interface
│   └── main.py                   # Typer application (evaluate, generate, report)
└── utils/                        # Infrastructure & helpers
    ├── llm.py                    # Unified LLMClient
    ├── cache.py                  # Disk response caching
    ├── chunking.py               # Semantic-aware text chunker
    ├── embeddings.py             # SentenceTransformers embedding client
    ├── text.py                   # Text cleaning & normalization
    └── providers/                # Cloud & local LLM providers
        ├── azure_provider.py     # Azure OpenAI (GPT-5-nano reasoning)
        ├── openai_provider.py    # OpenAI (GPT-4o, o3-mini)
        ├── anthropic_provider.py # Anthropic (Claude 3.5)
        ├── google_provider.py    # Google (Gemini 2.5)
        ├── ollama_provider.py    # Ollama local runtime
        └── vllm_provider.py      # vLLM high-concurrency server
```
