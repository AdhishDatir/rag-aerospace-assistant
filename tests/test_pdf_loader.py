"""Tests for page-aware PDF extraction."""

from pathlib import Path

import pymupdf
import pytest

from app.ingestion.pdf_loader import clean_text, extract_pdf_pages


def test_extract_pdf_pages_preserves_text_and_page_number(tmp_path: Path) -> None:
    """Text extracted from a PDF should retain its document and page metadata."""
    pdf_path = tmp_path / "sample_aircraft.pdf"

    document =  pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Aircraft inspection starts with visual checks.")
    second_page = document.new_page()
    second_page.insert_text((72, 72), "Record each inspection finding accurately.")
    document.save(pdf_path)
    document.close()

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0].document_name == "sample_aircraft.pdf"
    assert pages[0].page_number == 1
    assert "Aircraft inspection" in pages[0].text
    assert pages[1].page_number == 2
    assert "Record each inspection" in pages[1].text


def test_extract_pdf_pages_rejects_missing_file(tmp_path: Path) -> None:
    """A clear error should be raised for a PDF path that does not exist."""
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_pdf_pages(missing_pdf)


def test_clean_text_repairs_common_pdf_word_splits() -> None:
    """Legacy PDF extraction artifacts should not break retrieval terms."""
    assert clean_text("in� formation and dis- bonding") == "information and disbonding"
