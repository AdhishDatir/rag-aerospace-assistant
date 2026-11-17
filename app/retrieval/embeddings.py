"""Create local semantic embeddings for document chunks and user questions."""

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wrap a Sentence Transformers model used for semantic search."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Load the embedding model once for reuse."""
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert document chunk text into normalized embedding vectors."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in embedding] for embedding in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Convert one user question into a normalized embedding vector."""
        return self.embed_documents([query])[0]