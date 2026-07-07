"""Example demonstrating how to evaluate a benchmark dataset."""

from pathlib import Path

from rag_eval.datasets.benchmarks import get_benchmark_adapter
from rag_eval.pipeline.evaluator import Evaluator
from rag_eval.metrics.exact_match import ExactMatch
from rag_eval.metrics.f1 import TokenF1
from rag_eval.reports.generator import ReportGenerator
# NOTE: If you want to use LLM-based metrics (e.g. Faithfulness), 
# you will need to instantiate an LLMClient.
# from rag_eval.utils.llm import LLMClient
# from rag_eval.metrics.faithfulness import Faithfulness

def dummy_rag_pipeline(question: str) -> str:
    """
    This is a dummy RAG pipeline. 
    In a real scenario, this function would call your vector database, 
    retrieve contexts, and query your LLM to generate the final answer.
    """
    # Just a mock response for the sake of the example
    if "Normandy" in question:
        return "Normandy is a region in northern France."
    return "I don't know the answer."

def main() -> None:
    # 1. Carichiamo il dataset dal benchmark desiderato tramite la factory
    print("Loading SQuAD 2.0 benchmark...")
    squad_adapter = get_benchmark_adapter("squad2")
    
    # Carichiamo solo 5 sample per un test veloce
    samples = squad_adapter.load(split="validation", max_samples=5)
    
    # 2. Generiamo le risposte usando il nostro sistema RAG
    # Il benchmark ci fornisce 'question' e 'ground_truth', ma 'answer' è vuoto.
    # Dobbiamo popolarlo noi facendo rispondere il nostro RAG.
    print("\nGenerating answers through the RAG pipeline...")
    for i, sample in enumerate(samples):
        # Passiamo la domanda al nostro sistema
        generated_answer = dummy_rag_pipeline(sample.question)
        
        # Salviamo la risposta generata nel TestSample
        sample.answer = generated_answer
        print(f"[{i+1}/5] Q: {sample.question}")
        print(f"      A: {sample.answer}")
        
    # 3. Scegliamo le metriche di valutazione
    # (Usiamo metriche testuali semplici per evitare di configurare l'LLM in questo esempio)
    metrics = [
        ExactMatch(),
        TokenF1(),
    ]
    
    # 4. Facciamo partire l'evaluator
    print("\nStarting evaluation...")
    evaluator = Evaluator(metrics=metrics)
    report = evaluator.evaluate(samples, max_concurrency=1)
    
    # 5. Generiamo e visualizziamo i report
    print("\nEvaluation completed! Generating reports...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    generator = ReportGenerator(report)
    
    # Stampa a schermo i risultati
    generator.to_console()
    
    # Salva in formato JSON e HTML
    generator.to_json(output_dir / "benchmark_report.json")
    generator.to_html(output_dir / "benchmark_report.html")
    print(f"\nReports saved to {output_dir}/")

if __name__ == "__main__":
    main()
