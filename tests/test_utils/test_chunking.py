from unittest.mock import patch

import numpy as np

from rag_eval.utils.chunking import ChunkingService, _split_sentences


def test_split_sentences():
    text = "Hello world. This is a test! Is it working? Mr. Smith says yes. Yes, he does."
    sentences = _split_sentences(text)
    assert len(sentences) == 5
    assert sentences[0].strip() == "Hello world."
    assert sentences[1].strip() == "This is a test!"
    assert sentences[2].strip() == "Is it working?"
    assert sentences[3].strip() == "Mr. Smith says yes."
    assert sentences[4].strip() == "Yes, he does."


def test_split_sentences_fallback():
    text = "Hello world\nThis is a test"
    sentences = _split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == "Hello world\nThis is a test"


class MockSemanticModel:
    def encode(self, sentences, normalize_embeddings=False):
        # Return dummy embeddings (random vectors)
        return np.random.rand(len(sentences), 384)


@patch("rag_eval.utils.chunking.get_chunking_model", return_value=MockSemanticModel())
def test_chunking_service_fast(mock_get):
    service = ChunkingService()
    text = "A" * 2000
    # Will use fallback fast_chunking if semantic chunking is skipped
    chunks = service.fast_chunking(text)
    assert len(chunks) == 3
    assert len(chunks[0]) == 800


@patch("rag_eval.utils.chunking.get_chunking_model", return_value=MockSemanticModel())
def test_chunking_service_short_text(mock_get):
    service = ChunkingService()
    text = "This is a short text."
    chunks = service.chunk_text(text, max_chunk_chars=1500)
    assert chunks == [text]


@patch("rag_eval.utils.chunking.get_chunking_model")
def test_chunking_service_semantic(mock_get):
    class DeterministicMockModel:
        def encode(self, sentences, normalize_embeddings=False):
            # Return identity-like matrix so that dot product between different sentences is 0
            n = len(sentences)
            arr = np.zeros((n, max(n, 2)))
            for i in range(n):
                arr[i, i] = 1.0
            return arr

    mock_get.return_value = DeterministicMockModel()

    service = ChunkingService()
    # Provide multiple sentences
    # Because similarity will be 0 (threshold=0.3), they will be split if current_length >= min_chunk_chars
    # But min_chunk_chars defaults to 100
    text = "Sentence one is long enough. " * 5 + "Sentence two is also long enough. " * 5
    chunks = service.semantic_aware_chunking(text, min_chunk_chars=10, overlap=0)

    # We expect multiple chunks since they are completely dissimilar and pass min_chunk_chars
    assert len(chunks) > 1


@patch("rag_eval.utils.chunking.get_chunking_model", return_value=MockSemanticModel())
def test_chunking_service_empty(mock_get):
    service = ChunkingService()
    assert service.chunk_text("") == []
    assert service.semantic_aware_chunking("") == []
