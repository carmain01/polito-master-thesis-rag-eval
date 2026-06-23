"""Basic evaluation example — demonstrates the core workflow."""

from rag_eval.core.config import LLMConfig
from rag_eval.core.types import TestSample
from rag_eval.metrics.exact_match import ExactMatch
from rag_eval.metrics.f1 import TokenF1
from rag_eval.metrics.faithfulness import Faithfulness
from rag_eval.metrics.relevance import AnswerRelevance
from rag_eval.pipeline.evaluator import Evaluator
from rag_eval.utils.llm import LLMClient


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

    # 2. Create an LLM client (defaults to Ollama/llama3.2 — local and free)
    llm_config = LLMConfig()
    llm_client = LLMClient(config=llm_config)

    # 3. Choose metrics
    #    - LLM-as-judge metrics require an llm_client
    #    - Traditional NLP metrics (ExactMatch, TokenF1) work without an LLM
    metrics = [
        Faithfulness(llm_client=llm_client),
        AnswerRelevance(llm_client=llm_client),
        ExactMatch(),
        TokenF1(),
    ]

    # 4. Run evaluation
    evaluator = Evaluator(metrics=metrics)
    report = evaluator.evaluate(samples)

    # 5. Print results
    print("=== Evaluation Summary ===")
    for key, value in report.summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
