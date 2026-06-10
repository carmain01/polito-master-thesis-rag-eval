# RAG Evaluation Framework

A modular, extensible framework for evaluating Retrieval-Augmented Generation (RAG) systems.

## Project Structure

```
├── src/rag_eval/          # Main package
│   ├── core/              # Data models, config, base classes
│   ├── metrics/           # Evaluation metrics (faithfulness, relevance, etc.)
│   ├── datasets/          # Dataset loading and synthetic generation
│   ├── pipeline/          # Evaluation orchestration
│   ├── reports/           # Report generation and visualization
│   └── utils/             # Shared utilities (LLM clients, embeddings, etc.)
├── tests/                 # Unit and integration tests
├── examples/              # Usage examples and notebooks
├── docs/                  # Documentation
└── configs/               # Default and sample configurations
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from rag_eval.pipeline import Evaluator
from rag_eval.metrics import Faithfulness, AnswerRelevance
from rag_eval.datasets import load_dataset

# Load evaluation dataset
dataset = load_dataset("path/to/dataset.json")

# Configure evaluator
evaluator = Evaluator(
    metrics=[Faithfulness(), AnswerRelevance()],
)

# Run evaluation
results = evaluator.evaluate(dataset)
results.summary()
```

## License

MIT
