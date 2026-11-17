"""Print a short summary of text extracted from a PDF."""

import argparse
from pathlib import Path

from app.ingestion.pdf_loader import extract_pdf_pages


def main() -> None:
    """Extract a PDF and display page-level extraction results."""
    parser = argparse.ArgumentParser(
        description="Inspect page-aware text extracted from a PDF."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the PDF to inspect.",
    )
    args = parser.parse_args()

    pages = extract_pdf_pages(args.pdf)

    print(f"Document: {args.pdf.name}")
    print(f"Pages with extracted text: {len(pages)}")
    print()
    print(f"First extracted page: {pages[0].page_number}")
    print("Preview:")
    print(pages[0].text[:500])


if __name__ == "__main__":
    main()