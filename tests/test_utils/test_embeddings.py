"""Tests for the EmbeddingClient and cosine similarity utilities."""

from __future__ import annotations

import math

import pytest

from rag_eval.utils.embeddings import (
    EmbeddingClient,
    cosine_similarity,
    pairwise_cosine_similarity,
)

# ---------------------------------------------------------------------------
# Cosine similarity tests (pure math, no model needed)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Test cosine similarity utility function."""

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-7

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-7

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-7

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_known_value(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # Manual calculation
        dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
        norm_a = math.sqrt(1 + 4 + 9)  # sqrt(14)
        norm_b = math.sqrt(16 + 25 + 36)  # sqrt(77)
        expected = dot / (norm_a * norm_b)
        assert abs(cosine_similarity(a, b) - expected) < 1e-7


class TestPairwiseCosineSimilarity:
    """Test pairwise cosine similarity."""

    def test_basic(self):
        a = [[1.0, 0.0], [0.0, 1.0]]
        b = [[1.0, 0.0], [0.0, 1.0]]
        result = pairwise_cosine_similarity(a, b)
        assert len(result) == 2
        assert abs(result[0] - 1.0) < 1e-7
        assert abs(result[1] - 1.0) < 1e-7

    def test_mismatched_lengths_raises(self):
        a = [[1.0, 0.0]]
        b = [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(ValueError, match="same length"):
            pairwise_cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# EmbeddingClient tests (requires sentence-transformers model download)
# ---------------------------------------------------------------------------


class TestEmbeddingClient:
    """Test the EmbeddingClient with Sentence Transformers.

    These tests will download the model on first run (~80MB for all-MiniLM-L6-v2).
    """

    @pytest.fixture
    def client(self):
        return EmbeddingClient()

    def test_embed_single(self, client):
        embedding = client.embed_single("Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_batch(self, client):
        texts = ["Hello world", "Goodbye world", "Testing embeddings"]
        embeddings = client.embed(texts)
        assert len(embeddings) == 3
        assert all(len(e) == len(embeddings[0]) for e in embeddings)

    def test_embed_empty_list(self, client):
        result = client.embed([])
        assert result == []

    def test_similar_texts_high_similarity(self, client):
        emb_a = client.embed_single("The cat sat on the mat.")
        emb_b = client.embed_single("A cat was sitting on a mat.")
        similarity = cosine_similarity(emb_a, emb_b)
        assert similarity > 0.7  # Similar texts should have high similarity

    def test_different_texts_lower_similarity(self, client):
        emb_a = client.embed_single("The cat sat on the mat.")
        emb_b = client.embed_single("Quantum physics is complex.")
        similarity = cosine_similarity(emb_a, emb_b)
        assert similarity < 0.5  # Very different texts

    def test_dimensions(self, client):
        dims = client.dimensions
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert dims == 384

    def test_repr(self, client):
        r = repr(client)
        assert "all-MiniLM-L6-v2" in r
