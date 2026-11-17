"""Tests for text chunking with citation metadata."""

import pytest

from app.ingestion.chunker import chunk_pages, split_text
from app.ingestion.pdf_loader import PageText


def test_chunk_pages_keeps_source_metadata() -> None:
    """Every chunk should retain the source document and page number."""
    text = " ".join(["Aircraft inspection requires careful observation."] * 30)
    pages = [
        PageText(
            document_name="inspection_guide.pdf",
            page_number=7,
            text=text,
        )
    ]

    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.document_name == "inspection_guide.pdf" for chunk in chunks)
    assert all(chunk.page_number == 7 for chunk in chunks)
    assert chunks[0].chunk_id == "inspection_guide.pdf-page-7-chunk-1"
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_split_text_rejects_invalid_overlap() -> None:
    """Overlap must be smaller than the chunk size to ensure progress."""
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        split_text("sample text", chunk_size=100, chunk_overlap=100)