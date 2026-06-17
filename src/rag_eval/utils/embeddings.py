"""Embedding client — local embedding generation via Sentence Transformers."""

from __future__ import annotations

import numpy as np

from rag_eval.core.config import EmbeddingConfig


class EmbeddingClient:
    """Generate text embeddings using Sentence Transformers (local, free).

    Uses the ``sentence-transformers`` library to run embedding models
    directly on your machine — no API calls, no cost, fully offline.

    Default model: ``all-MiniLM-L6-v2`` (fast, lightweight, 384 dimensions).
    For higher quality, use ``all-mpnet-base-v2`` (768 dimensions).

    Example::

        client = EmbeddingClient()
        embeddings = client.embed(["Hello world", "Another sentence"])
        similarity = cosine_similarity(embeddings[0], embeddings[1])
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = None

    def _load_model(self):
        """Lazy-load the Sentence Transformer model (avoids slow import at startup)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        # Convert torch tensors to plain Python lists
        # (avoids numpy compatibility issues with certain torch versions)
        return [e.tolist() for e in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        results = self.embed([text])
        return results[0]

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding model."""
        model = self._load_model()
        # Use get_embedding_dimension (renamed from get_sentence_embedding_dimension)
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        return f"EmbeddingClient(model={self.config.model!r})"


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score in [-1.0, 1.0]. Higher means more similar.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def pairwise_cosine_similarity(
    embeddings_a: list[list[float]],
    embeddings_b: list[list[float]],
) -> list[float]:
    """Compute pairwise cosine similarity between two lists of embeddings.

    Args:
        embeddings_a: First list of embedding vectors.
        embeddings_b: Second list of embedding vectors (must be same length).

    Returns:
        List of cosine similarity scores, one per pair.

    Raises:
        ValueError: If the lists have different lengths.
    """
    if len(embeddings_a) != len(embeddings_b):
        raise ValueError(
            f"Embedding lists must have the same length, "
            f"got {len(embeddings_a)} and {len(embeddings_b)}"
        )

    return [cosine_similarity(a, b) for a, b in zip(embeddings_a, embeddings_b)]
