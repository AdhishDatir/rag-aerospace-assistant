"""Measure whether expected evidence appears in the top-k retrieval results."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore, SearchResult


class RetrievalClient(Protocol):
    """Protocol that makes the evaluator easy to test with a fake retriever."""

    def retrieve(self, question: str, top_k: int = 4) -> list[SearchResult]:
        """Return the top matching chunks for a question."""


@dataclass(frozen=True)
class EvaluationRecord:
    """Outcome of one manually labelled retrieval test."""

    question_id: str
    question: str
    hit: bool


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load manually labelled questions from a JSON file."""
    with path.open(encoding="utf-8") as file:
        questions = json.load(file)
    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation questions must be a non-empty JSON list")
    return questions


def contains_expected_passage(results: list[SearchResult], expected_document: str, expected_terms: list[str]) -> bool:
    """Return whether one result contains all labelled evidence terms."""
    for result in results:
        if result.document_name == expected_document and all(term.casefold() in result.text.casefold() for term in expected_terms):
            return True
    return False


def evaluate_retrieval(retriever: RetrievalClient, questions: list[dict[str, Any]], top_k: int) -> list[EvaluationRecord]:
    """Evaluate top-k evidence retrieval for every labelled question."""
    records: list[EvaluationRecord] = []
    for item in questions:
        hit = contains_expected_passage(retriever.retrieve(item["question"], top_k=top_k), item["expected_document"], item["expected_terms"])
        records.append(EvaluationRecord(question_id=item["id"], question=item["question"], hit=hit))
    return records


def main() -> None:
    """Run the retrieval benchmark and print a concise accuracy report."""
    parser = argparse.ArgumentParser(description="Evaluate top-k retrieval accuracy.")
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.json"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--database-path", type=Path, default=Path("data/chroma"))
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    records = evaluate_retrieval(Retriever(EmbeddingService(), ChromaVectorStore(args.database_path)), load_questions(args.questions), args.top_k)
    hits = sum(record.hit for record in records)
    print(f"Questions evaluated: {len(records)}")
    print(f"Top-{args.top_k} evidence hits: {hits}")
    print(f"Top-{args.top_k} retrieval accuracy: {hits / len(records) * 100:.1f}%")
    print("Missed questions:")
    for record in (record for record in records if not record.hit):
        print(f"- {record.question_id}: {record.question}")


if __name__ == "__main__":
    main()
