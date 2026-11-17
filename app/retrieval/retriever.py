"""Retrieve the most relevant document chunks for a user question."""

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import ChromaVectorStore, SearchResult


class Retriever:
    """Coordinate query embedding and vector similarity search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
    ) -> None:
        """Receive dependencies explicitly for easier testing."""
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 4) -> list[SearchResult]:
        """Return the top matching chunks for a non-empty question."""
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("question must not be empty")

        query_embedding = self.embedding_service.embed_query(cleaned_question)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )