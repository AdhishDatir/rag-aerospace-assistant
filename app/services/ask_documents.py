"""Ask a grounded question against the local document database."""

import argparse
from pathlib import Path

import httpx

from app.core.config import settings
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore
from app.services.qa_service import QAService


def main() -> None:
    """Retrieve evidence and generate one local grounded answer."""
    parser = argparse.ArgumentParser(
        description="Ask a grounded question about indexed aerospace documents."
    )
    parser.add_argument("--question", required=True, help="Question to answer.")
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

    with httpx.Client(timeout=120.0) as client:
        qa_service = QAService(
            client=client,
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

        chunks = retriever.retrieve(args.question)
        result = qa_service.answer_question(args.question, chunks)

    print(f"Answer:\n{result.answer}")
    print("\nSources:")

    if not result.citations:
        print("No sources because the answer was not found in the documents.")
        return

    for citation in result.citations:
        print(f"- {citation.document_name}, page {citation.page_number}")


if __name__ == "__main__":
    main()