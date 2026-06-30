"""Tests for the LangChain integration."""

from typing import Any

import pytest

try:
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    from rag_eval.integrations.langchain import RagEvalCallbackHandler

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")


@pytest.fixture
def mock_chain():
    """Create a mock LCEL chain for testing."""

    def mock_retrieve(query: str) -> list[Document]:
        return [
            Document(page_content="doc1 content", metadata={"source": "test", "page": 1}),
            Document(page_content="doc2 content", metadata={"source": "test", "page": 2}),
        ]

    def mock_llm(prompt: Any) -> str:
        return f"Mock answer based on prompt: {prompt}"

    prompt = ChatPromptTemplate.from_template("Context: {context} Question: {question}")
    retriever = RunnableLambda(mock_retrieve).with_config(run_name="retriever")
    llm = RunnableLambda(mock_llm).with_config(run_name="llm")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    return chain


def test_sync_callback_captures_query_and_answer(mock_chain):
    """Test that query and answer are captured from a sync chain invocation."""
    handler = RagEvalCallbackHandler()
    res = mock_chain.invoke("What is test 1?", config={"callbacks": [handler]})

    sample = handler.get_sample()
    assert sample is not None
    assert sample.question == "What is test 1?"
    assert sample.answer == res


def test_sync_callback_captures_contexts(mock_chain):
    """Test that retrieved document contents appear in contexts."""
    handler = RagEvalCallbackHandler()
    mock_chain.invoke("What is test?", config={"callbacks": [handler]})

    sample = handler.get_sample()
    assert sample is not None
    assert "doc1 content" in sample.contexts
    assert "doc2 content" in sample.contexts


def test_sync_callback_preserves_document_metadata(mock_chain):
    """Test that full document metadata is preserved in sample.metadata."""
    handler = RagEvalCallbackHandler()
    mock_chain.invoke("metadata test", config={"callbacks": [handler]})

    sample = handler.get_sample()
    assert sample is not None
    retrieved_docs = sample.metadata.get("retrieved_documents", [])
    assert len(retrieved_docs) >= 2

    # Check that the original metadata was kept
    doc1 = next(d for d in retrieved_docs if d["page_content"] == "doc1 content")
    assert doc1["metadata"]["source"] == "test"
    assert doc1["metadata"]["page"] == 1


def test_sync_callback_deduplicates_contexts(mock_chain):
    """Test that contexts are deduplicated (same doc captured via retriever + chain_end)."""
    handler = RagEvalCallbackHandler()
    mock_chain.invoke("dedup test", config={"callbacks": [handler]})

    sample = handler.get_sample()
    assert sample is not None
    # Each document should appear exactly once, not duplicated
    assert sample.contexts.count("doc1 content") == 1
    assert sample.contexts.count("doc2 content") == 1


def test_multiple_invocations(mock_chain):
    """Test that multiple invocations accumulate separate samples."""
    handler = RagEvalCallbackHandler()

    res1 = mock_chain.invoke("Q1?", config={"callbacks": [handler]})
    res2 = mock_chain.invoke("Q2?", config={"callbacks": [handler]})

    samples = handler.get_samples()
    assert len(samples) == 2

    assert samples[0].question == "Q1?"
    assert samples[0].answer == res1
    assert samples[1].question == "Q2?"
    assert samples[1].answer == res2

    # get_sample returns the last one
    last_sample = handler.get_sample()
    assert last_sample is not None
    assert last_sample.question == "Q2?"


@pytest.mark.asyncio
async def test_async_callback_handler(mock_chain):
    """Test the callback handler with asynchronous chain execution."""
    handler = RagEvalCallbackHandler()
    res = await mock_chain.ainvoke("Async question?", config={"callbacks": [handler]})

    sample = handler.get_sample()
    assert sample is not None
    assert sample.question == "Async question?"
    assert sample.answer == res
    assert "doc1 content" in sample.contexts


def test_callback_handler_reset(mock_chain):
    """Test resetting the callback handler clears all state."""
    handler = RagEvalCallbackHandler()
    mock_chain.invoke("Question?", config={"callbacks": [handler]})

    assert len(handler.get_samples()) == 1

    handler.reset()
    assert len(handler.get_samples()) == 0
    assert handler.get_sample() is None


def test_langchain_run_id_in_metadata(mock_chain):
    """Test that the LangChain run_id is stored in sample metadata."""
    handler = RagEvalCallbackHandler()
    mock_chain.invoke("run id test", config={"callbacks": [handler]})

    sample = handler.get_sample()
    assert sample is not None
    assert "langchain_run_id" in sample.metadata
    assert isinstance(sample.metadata["langchain_run_id"], str)
    assert len(sample.metadata["langchain_run_id"]) > 0
