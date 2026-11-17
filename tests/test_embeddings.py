"""Tests for the embedding service without downloading a real model."""

from app.retrieval.embeddings import EmbeddingService


class FakeEmbeddingModel:
    """Small predictable stand-in for SentenceTransformer during tests."""

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return [[float(len(text)), 1.0] for text in texts]


def test_embed_documents_returns_one_vector_per_text() -> None:
    """Every input text should receive one embedding vector."""
    service = object.__new__(EmbeddingService)
    service.model = FakeEmbeddingModel()

    embeddings = service.embed_documents(["aircraft inspection", "fatigue analysis"])

    assert embeddings == [[19.0, 1.0], [16.0, 1.0]]