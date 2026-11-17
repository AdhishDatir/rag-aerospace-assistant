"""Extract text from PDF documents while preserving page metadata."""

from dataclasses import dataclass
from pathlib import Path
import re

import  pymupdf  # PyMuPDF


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page, together with its source metadata."""

    document_name: str
    page_number: int
    text: str


def clean_text(text: str) -> str:
    """Normalize whitespace so extracted text is easier to search later."""
    # Some legacy technical PDFs split words around replacement characters or
    # line-wrap hyphens (for example, "in� formation" and "dis- bonding").
    text = re.sub(r"(\w)�\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = text.replace("�", "")
    return " ".join(text.split())


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    """Extract non-empty text from every page in a PDF.

    Args:
        pdf_path: Path to a readable PDF document.

    Returns:
        One PageText object for each page containing extractable text.

    Raises:
        FileNotFoundError: If the supplied PDF path does not exist.
        ValueError: If the PDF has no extractable text.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    extracted_pages: list[PageText] = []

    with  pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = clean_text(page.get_text())

            if text:
                extracted_pages.append(
                    PageText(
                        document_name=pdf_path.name,
                        page_number=page_number,
                        text=text,
                    )
                )

    if not extracted_pages:
        raise ValueError(f"No extractable text found in: {pdf_path}")

    return extracted_pages
