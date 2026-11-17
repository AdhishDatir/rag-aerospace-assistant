"""Tests for query embedding and vector retrieval coordination."""

import pytest

from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import SearchResult


class FakeEmbeddingService:
    """Returns a known query vector and records the requested text."""

    def __init__(self) -> None:
        self.received_question: str | None = None

    def embed_query(self, question: str) -> list[float]:
        self.received_question = question
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    """Returns a known search result and records search arguments."""

    def __init__(self) -> None:
        self.received_embedding: list[float] | None = None
        self.received_top_k: int | None = None

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        self.received_embedding = query_embedding
        self.received_top_k = top_k

        return [
            SearchResult(
                chunk_id="faa.pdf-page-1-chunk-1",
                document_name="faa.pdf",
                page_number=1,
                text="Visual inspection provides technical information.",
                distance=0.12,
            )
        ]


def test_retrieve_embeds_question_and_searches_store() -> None:
    """A question should be embedded and used in a top-k vector search."""
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedding_service, vector_store)

    results = retriever.retrieve("  What is visual inspection?  ", top_k=3)

    assert embedding_service.received_question == "What is visual inspection?"
    assert vector_store.received_embedding == [0.1, 0.2, 0.3]
    assert vector_store.received_top_k == 3
    assert results[0].page_number == 1


def test_retrieve_rejects_an_empty_question() -> None:
    """Blank questions should fail before embedding or database access."""
    retriever = Retriever(FakeEmbeddingService(), FakeVectorStore())

    with pytest.raises(ValueError, match="question must not be empty"):
        retriever.retrieve("   ")