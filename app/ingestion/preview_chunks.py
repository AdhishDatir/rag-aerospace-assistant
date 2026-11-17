"""Preview chunks created from a source PDF."""

import argparse
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import extract_pdf_pages


def main() -> None:
    """Extract a PDF, split it into chunks, and print a short preview."""
    parser = argparse.ArgumentParser(
        description="Preview page-aware chunks created from a PDF."
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the source PDF.")
    args = parser.parse_args()

    pages = extract_pdf_pages(args.pdf)
    chunks = chunk_pages(pages)

    print(f"Document: {args.pdf.name}")
    print(f"Extracted pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")
    print()
    print(f"First citation: {chunks[0].document_name}, page {chunks[0].page_number}")
    print("First chunk preview:")
    print(chunks[0].text[:500])


if __name__ == "__main__":
    main()