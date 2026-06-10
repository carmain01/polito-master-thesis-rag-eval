"""Basic evaluation example — demonstrates the core workflow."""

from rag_eval.core.types import TestSample
from rag_eval.metrics.faithfulness import Faithfulness
from rag_eval.metrics.relevance import AnswerRelevance
from rag_eval.pipeline.evaluator import Evaluator


def main() -> None:
    # 1. Define evaluation samples
    samples = [
        TestSample(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            ground_truth="Paris is the capital of France.",
            contexts=[
                "France is a country in Western Europe. Its capital is Paris.",
                "Paris is known for the Eiffel Tower and the Louvre Museum.",
            ],
        ),
    ]

    # 2. Choose metrics
    metrics = [Faithfulness(), AnswerRelevance()]

    # 3. Run evaluation
    evaluator = Evaluator(metrics=metrics)
    report = evaluator.evaluate(samples)

    # 4. Print results
    print("=== Evaluation Summary ===")
    for metric_name, avg_score in report.summary.items():
        print(f"  {metric_name}: {avg_score:.3f}")


if __name__ == "__main__":
    main()
