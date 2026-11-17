"""Persist and search page-aware document chunks with Chroma."""

from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.ingestion.chunker import TextChunk


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk together with its semantic distance."""

    chunk_id: str
    document_name: str
    page_number: int
    text: str
    distance: float


class ChromaVectorStore:
    """Local Chroma collection for aerospace document embeddings."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "aerospace_documents",
    ) -> None:
        """Open or create a persistent Chroma collection."""
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Save chunks and matching embeddings, replacing duplicate chunk IDs."""
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks must match the number of embeddings")

        if not chunks:
            return 0

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                }
                for chunk in chunks
            ],
            embeddings=embeddings,
        )

        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[SearchResult]:
        """Return the nearest chunks for a question embedding."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            SearchResult(
                chunk_id=chunk_id,
                document_name=str(metadata["document_name"]),
                page_number=int(metadata["page_number"]),
                text=str(document),
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()