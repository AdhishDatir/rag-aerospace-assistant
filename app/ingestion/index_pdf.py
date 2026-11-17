"""Index a PDF into the local Chroma vector database."""

import argparse
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import extract_pdf_pages
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import ChromaVectorStore


def main() -> None:
    """Build or update the local vector index for one PDF."""
    parser = argparse.ArgumentParser(
        description="Extract, chunk, embed, and store a PDF in Chroma."
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the source PDF.")
    parser.add_argument(
        "--database-path",
        type=Path,
        default=Path("data/chroma"),
        help="Directory where Chroma stores vectors.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before indexing this document.",
    )
    args = parser.parse_args()

    pages = extract_pdf_pages(args.pdf)
    chunks = chunk_pages(pages)

    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_documents(
        [chunk.text for chunk in chunks]
    )

    vector_store = ChromaVectorStore(args.database_path)
    if args.reset:
        vector_store.client.delete_collection("aerospace_documents")
        vector_store = ChromaVectorStore(args.database_path)
    stored_count = vector_store.upsert_chunks(chunks, embeddings)

    print(f"Document: {args.pdf.name}")
    print(f"Extracted pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Stored chunks in this run: {stored_count}")
    print(f"Total chunks in database: {vector_store.count()}")


if __name__ == "__main__":
    main()
