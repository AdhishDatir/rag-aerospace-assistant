"""Tests for Chroma payload construction without creating a real database."""

import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.vector_store import ChromaVectorStore


class FakeCollection:
    """Captures Chroma calls so tests remain fast and isolated."""

    def __init__(self) -> None:
        self.upsert_payload: dict[str, object] | None = None

    def upsert(self, **kwargs: object) -> None:
        self.upsert_payload = kwargs


def test_upsert_chunks_sends_text_embeddings_and_metadata() -> None:
    """Stored records must include text and citation metadata."""
    store = object.__new__(ChromaVectorStore)
    store.collection = FakeCollection()

    chunks = [
        TextChunk(
            chunk_id="faa.pdf-page-2-chunk-1",
            document_name="faa.pdf",
            page_number=2,
            text="Inspect the aircraft surface under proper lighting.",
        )
    ]

    stored_count = store.upsert_chunks(chunks, [[0.1, 0.2, 0.3]])

    assert stored_count == 1
    assert store.collection.upsert_payload == {
        "ids": ["faa.pdf-page-2-chunk-1"],
        "documents": ["Inspect the aircraft surface under proper lighting."],
        "metadatas": [{"document_name": "faa.pdf", "page_number": 2}],
        "embeddings": [[0.1, 0.2, 0.3]],
    }


def test_upsert_chunks_rejects_mismatched_lengths() -> None:
    """Each chunk must have exactly one embedding."""
    store = object.__new__(ChromaVectorStore)

    with pytest.raises(ValueError, match="must match"):
        store.upsert_chunks(
            [
                TextChunk(
                    chunk_id="example",
                    document_name="example.pdf",
                    page_number=1,
                    text="Example text.",
                )
            ],
            [],
        )