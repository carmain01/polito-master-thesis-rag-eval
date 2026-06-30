"""Example demonstrating how to evaluate a LangChain RAG pipeline with rag-eval."""


from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag_eval.integrations.langchain import RagEvalCallbackHandler
from rag_eval.metrics.f1 import TokenF1
from rag_eval.pipeline.evaluator import Evaluator
from rag_eval.reports.generator import ReportGenerator

# 1. Setup a Mock LangChain RAG Pipeline
# In a real scenario, you'd use a real VectorStoreRetriever and LLM (e.g. ChatOpenAI, ChatOllama)

def mock_retrieve(query: str) -> list[Document]:
    """A mock retriever that returns a static document."""
    return [
        Document(
            page_content="Paris is the capital and most populous city of France.",
            metadata={"source": "wikipedia"}
        )
    ]

from typing import Any

def mock_llm(prompt: Any) -> str:
    """A mock LLM that returns a static answer."""
    return "The capital of France is Paris."

# Build the LCEL (LangChain Expression Language) chain
prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context:\nContext: {context}\nQuestion: {question}"
)

retriever = RunnableLambda(mock_retrieve).with_config(run_name="retriever")
llm = RunnableLambda(mock_llm).with_config(run_name="llm")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def main() -> None:
    print("--- 1. Running LangChain Pipeline ---")

    # 2. Attach the RagEvalCallbackHandler
    eval_handler = RagEvalCallbackHandler()

    # Run the chain, passing the handler to callbacks
    question = "What is the capital of France?"
    result = rag_chain.invoke(
        question,
        config={"callbacks": [eval_handler]}
    )

    print(f"Question: {question}")
    print(f"Answer: {result}")

    # 3. Extract the TestSample
    sample = eval_handler.get_sample()
    if not sample:
        print("Failed to capture sample!")
        return

    print("\n--- 2. Captured Evaluation Sample ---")
    print(f"Question: {sample.question}")
    print(f"Contexts: {sample.contexts}")
    print(f"Answer:   {sample.answer}")

    # (Optional) Add ground truth if you are evaluating against a dataset
    sample.ground_truth = "Paris"

    # 4. Evaluate the captured sample
    print("\n--- 3. Evaluating with rag-eval ---")

    metrics = [TokenF1()]
    evaluator = Evaluator(metrics=metrics)

    # Evaluator expects a list of samples
    report = evaluator.evaluate([sample])

    # Generate the report
    generator = ReportGenerator(report)
    generator.to_console()

if __name__ == "__main__":
    main()
