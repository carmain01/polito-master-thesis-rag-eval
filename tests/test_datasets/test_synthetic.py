from unittest.mock import AsyncMock, patch

import pytest

from rag_eval.datasets.synthetic import SyntheticDataGenerator


@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    # By default, mock successful JSON generation
    client.complete_json.return_value = {
        "question": "What is the capital of France?",
        "answer": "Paris"
    }
    return client

@pytest.mark.asyncio
async def test_generate_qa_pairs(mock_llm_client):
    mock_llm_client.complete_json.side_effect = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is the capital of Italy?", "answer": "Rome"}
    ]
    generator = SyntheticDataGenerator(llm_client=mock_llm_client)
    # Mock chunking so it returns predictable chunks
    with patch.object(generator.chunking_service, 'semantic_aware_chunking', return_value=["Chunk 1", "Chunk 2"]):
        samples = await generator.generate_qa_pairs("Text here", num_questions_per_chunk=1)
        assert len(samples) == 2
        assert samples[0].question == "What is the capital of France?"
        assert samples[1].question == "What is the capital of Italy?"
        assert samples[0].metadata["is_synthetic"] is True

@pytest.mark.asyncio
async def test_deduplication(mock_llm_client):
    generator = SyntheticDataGenerator(llm_client=mock_llm_client)
    # We ask for 3 questions from 1 chunk. The LLM always returns the exact same question.
    # Deduplication logic should filter out the 2nd and 3rd ones.
    with patch.object(generator.chunking_service, 'semantic_aware_chunking', return_value=["Single Chunk"]):
        samples = await generator.generate_qa_pairs("Text", num_questions_per_chunk=3)
        assert len(samples) == 1
        assert samples[0].question == "What is the capital of France?"

@pytest.mark.asyncio
async def test_quality_filtering(mock_llm_client):
    # Setup LLM to return bad quality on first call, good on second
    mock_llm_client.complete_json.side_effect = [
        {"question": "What?", "answer": "Yes"}, # Too short
        {"question": "Same string", "answer": "Same string"}, # Identical
        {"question": "What is a good valid question?", "answer": "Valid answer"} # Good
    ]
    generator = SyntheticDataGenerator(llm_client=mock_llm_client)

    with patch.object(generator.chunking_service, 'semantic_aware_chunking', return_value=["Chunk"]):
        samples = await generator.generate_qa_pairs("Text", num_questions_per_chunk=3)
        assert len(samples) == 1
        assert samples[0].question == "What is a good valid question?"

@pytest.mark.asyncio
async def test_empty_chunks(mock_llm_client):
    generator = SyntheticDataGenerator(llm_client=mock_llm_client)
    with patch.object(generator.chunking_service, 'semantic_aware_chunking', return_value=[]):
        samples = await generator.generate_qa_pairs("Empty or not chunkable text")
        assert len(samples) == 0
