"""Basic evaluation example — demonstrates the core workflow."""

from rag_eval.utils.embeddings import EmbeddingClient
from rag_eval.metrics.semantic_similarity import SemanticSimilarity
from pathlib import Path

from rag_eval.core.config import LLMConfig, EmbeddingConfig
from rag_eval.datasets.loader import load_dataset, dataset_statistics
from rag_eval.metrics.exact_match import ExactMatch
from rag_eval.metrics.f1 import TokenF1
from rag_eval.metrics.faithfulness import Faithfulness
from rag_eval.metrics.relevance import AnswerRelevance
from rag_eval.pipeline.evaluator import Evaluator
from rag_eval.reports.generator import ReportGenerator
from rag_eval.utils.llm import LLMClient
from rag_eval.metrics.base import BaseMetric



def main() -> None:
    # 1. Load the synthetic dataset
    dataset_path = Path(__file__).parent / "synthetic_dataset.json"
    samples = load_dataset(dataset_path)

    stats = dataset_statistics(samples)
    print(f"Loaded {stats['num_samples']} samples")
    print(f"  Avg contexts/sample: {stats['avg_contexts_per_sample']:.1f}")
    print(f"  Avg context length:  {stats['avg_context_length_chars']:.0f} chars")
    print()

    # 2. Create an LLM client (defaults to Ollama/llama3.2 — local and free)
    llm_config = LLMConfig()
    llm_client = LLMClient(config=llm_config)
    embedding_config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config=embedding_config)

    # 3. Choose metrics
    #    - LLM-as-judge metrics require an llm_client
    #    - Traditional NLP metrics (ExactMatch, TokenF1) work without an LLM
    metrics: list[BaseMetric] = [
        Faithfulness(llm_client=llm_client),
        AnswerRelevance(llm_client=llm_client),
        SemanticSimilarity(embed_client=embedding_client),
        TokenF1(),
    ]

    # 4. Run evaluation
    evaluator = Evaluator(metrics=metrics)
    report = evaluator.evaluate(samples)

    # 5. Generate reports
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating Reports...")
    generator = ReportGenerator(report)
    
    generator.to_console()
    
    generator.to_json(output_dir / "report.json")
    print(f" - Saved JSON report to {output_dir / 'report.json'}")
    
    generator.to_csv(output_dir / "report.csv")
    print(f" - Saved CSV report to {output_dir / 'report.csv'}")
    
    generator.to_html(output_dir / "report.html")
    print(f" - Saved HTML report to {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
