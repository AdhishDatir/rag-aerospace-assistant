"""Search the local aerospace document database from the command line."""

import argparse
from pathlib import Path

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore


def main() -> None:
    """Run a semantic search and print retrieved chunks with citations."""
    parser = argparse.ArgumentParser(
        description="Search indexed aerospace documents."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Natural-language question to search for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of relevant chunks to return.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=Path("data/chroma"),
        help="Directory containing the Chroma database.",
    )
    args = parser.parse_args()

    embedding_service = EmbeddingService()
    vector_store = ChromaVectorStore(args.database_path)
    retriever = Retriever(embedding_service, vector_store)

    results = retriever.retrieve(args.question, top_k=args.top_k)

    if not results:
        print("No matching chunks found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"[{rank}] {result.document_name}, page {result.page_number}")
        print(f"Distance: {result.distance:.4f}")
        print(result.text[:700])
        print("-" * 80)


if __name__ == "__main__":
    main()