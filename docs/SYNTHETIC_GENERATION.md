# Synthetic Data Generation Methodology

## 1. Overview & Research Motivation

In many real-world and enterprise settings — including institutional e-learning platforms and corporate knowledge bases — labeled ground-truth datasets (aligned triplets of query, retrieved context, and reference answer) are scarce or completely absent. Manually annotating thousands of educational queries is prohibitively expensive and difficult to standardize.

To overcome this bottleneck, the **`rag_eval`** framework incorporates an automated **Synthetic Data Generation Engine** (`SyntheticDataGenerator` in `src/rag_eval/datasets/synthetic.py`). The pipeline ingests unstructured domain documents and produces high-quality, diverse, and stochastically realistic `TestSample` datasets ready for evaluation.

---

## 2. End-to-End Generation Pipeline

```
Raw Text Document (.txt)
         │
         ▼
┌────────────────────────────────────────┐
│    1. Semantic-Aware Chunking          │
│   (SentenceTransformers / MiniLM)      │
└──────────────────┬─────────────────────┘
                   │ Cohesive Chunks
                   ▼
┌────────────────────────────────────────┐
│    2. Prompt & Difficulty Engine       │
│   (Factual, Multi-Hop, Reasoning...)   │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│    3. Distractor Context Injection     │
│   (Sample & shuffle noise chunks)      │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│    4. Stochastic RAG Simulation        │
│   (70% Grounded, 15% Hallucination,    │
│    15% Abstention/Unanswerable)        │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│    5. Quality Filtering & Dedup        │
│   (Length, tautology, hash checks)     │
└──────────────────┬─────────────────────┘
                   │
                   ▼
       Exported JSON Dataset
```

---

## 3. Two-Tier Document Chunking

Proper chunking is critical: naively splitting by character count cuts across sentences and breaks conceptual integrity. The `ChunkingService` implements:

1. **Semantic-Aware Chunking (Primary Tier):**
   - The document is segmented into grammatical sentences using NLTK / regex boundaries.
   - Embeddings are computed for adjacent sentences using `sentence-transformers/paraphrase-MiniLM-L6-v2`.
   - Sentences are merged into a cohesive chunk as long as cosine similarity between consecutive sentences remains above an empirical threshold ($\ge 0.65$) and the total token budget is not exceeded.
   - When a semantic topic shift is detected, a new chunk boundary is established.

2. **Fast Fallback Chunking (Secondary Tier):**
   - If embedding computation fails (e.g. out of memory or missing model weights), the system falls back to overlapping token/character chunking with configurable stride, guaranteeing uninterrupted execution.

---

## 4. LLM-Based QA Pair Generation

Each semantic chunk is provided to an LLM prompted as an expert dataset author. The generation is parameterized by **Question Type** and **Difficulty**:

### 4.1 Question Taxonomies

| Type | Instruction Focus | Evaluation Goal |
|---|---|---|
| **Factual** | Direct extraction of explicit facts from the chunk. | Baseline recall & precision. |
| **Multi-Hop** | Combines disparate facts spread across different sentences. | Retrieval completeness & synthesis. |
| **Reasoning** | Derives logical inferences implied but not explicitly stated. | Deep reasoning & context comprehension. |
| **Comparative**| Contrasts two or more entities, algorithms, or concepts. | Multi-entity discrimination. |

### 4.2 Difficulty Modulation

The prompt instruction dynamically incorporates a difficulty modifier:
- **`easy`**: Explicit, surface-level queries matching the wording of the source text closely.
- **`medium`**: Synthesized queries requiring paraphrasing.
- **`hard`**: Abstract queries requiring high-level abstraction, vocabulary variation, or complex deduction.

---

## 5. Context Handling & Distractor Injection

To properly evaluate **Context Precision** (measuring whether a retriever ranks signal above noise), each sample cannot consist solely of the true context.

The pipeline implements **Distractor Context Injection**:
1. For each generated question, the originating chunk is marked as the *Ground Truth Context*.
2. One or more unrelated chunks from different parts of the same document are randomly selected as *Distractors*.
3. The true context and distractors are merged into `contexts: list[str]` and pseudo-randomly shuffled.
4. During evaluation, metrics like `ContextPrecision` verify whether the retriever places the true chunk at Rank 1.

---

## 6. Stochastic RAG Behavior Simulation

A unique and powerful capability of the `rag_eval` generator is that it does not assume the RAG system produces perfect answers. Instead, it stochastically **emulates realistic failure modes** seen in real-world deployments:

```python
behavior_roll = random.random()

if behavior_roll < 0.15:
    # 15% Hallucination Mode:
    # Generates a plausible, well-formed, but factually ungrounded response.
    system_prompt = "You are an AI assistant. Provide a plausible but entirely incorrect or hallucinated answer to the question, ignoring the facts in the context."

elif behavior_roll < 0.30:
    # 15% Abstention Mode:
    # Simulates a conservative system refusing to answer.
    system_prompt = "You are an AI assistant. State exactly that the provided context does not contain enough information to answer the question."

else:
    # 70% Grounded Faithful Mode:
    # Answers strictly adhering to the provided context.
    system_prompt = "You are a helpful AI assistant. Answer the question using ONLY the provided context."
```

This controlled distribution ($70\%$ faithful, $15\%$ hallucinated, $15\%$ unanswerable) creates a benchmark with known ground truth for validating whether metrics like `Faithfulness` and `Relevance` can reliably detect hallucinations and abstentions.

---

## 7. Automated Quality Filtering & Deduplication

Generated pairs undergo rigorous validation before being appended to the dataset:
- **Length Thresholds:** Discards empty queries, questions shorter than 10 characters, or answers shorter than 2 characters.
- **Tautology Filtering:** Discards samples where the question is verbatim identical to the answer.
- **Runtime Deduplication:** Maintains a normalized hash set of lowercased question strings to prevent semantic duplicates across chunks.

---

## 8. Output Schema (`TestSample`)

Each generated sample adheres to the standard `TestSample` JSON schema:

```json
{
  "question": "What is the primary role of the U.S. Office of Technology Assessment?",
  "answer": "The Office of Technology Assessment identified key organizational barriers...",
  "ground_truth": "The U.S. Office of Technology Assessment (OTA) identified four main problems...",
  "contexts": [
    "Context Chunk A (True Context)...",
    "Context Chunk B (Injected Distractor)..."
  ],
  "metadata": {
    "source": "synthetic",
    "question_type": "factual",
    "difficulty": "medium",
    "chunk_id": 4,
    "is_synthetic": true
  }
}
```

---

## 9. Usage

### Via CLI

```bash
rag-eval generate \
  --documents source_materials.txt \
  --output data/synthetic_evaluation_set.json \
  --num-samples 2 \
  --question-types factual,reasoning,multi-hop \
  --difficulty medium \
  --provider ollama \
  --model qwen2.5:7b \
  --max-chunks 20
```

### Programmatic Python API

```python
import asyncio
from pathlib import Path
from rag_eval.datasets.synthetic import SyntheticDataGenerator
from rag_eval.utils.llm import LLMClient
from rag_eval.core.config import LLMConfig

async def main():
    llm_client = LLMClient(config=LLMConfig(provider="ollama", model="qwen2.5:7b"))
    generator = SyntheticDataGenerator(llm_client=llm_client)
    
    text = Path("document.txt").read_text(encoding="utf-8")
    samples = await generator.generate_qa_pairs(
        text=text,
        num_questions_per_chunk=2,
        question_types=["factual", "reasoning"],
        difficulty="hard",
        max_chunks=10,
    )
    print(f"Generated {len(samples)} synthetic evaluation samples.")

asyncio.run(main())
```
