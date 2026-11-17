"""Tests for evidence-based retrieval evaluation."""

from app.retrieval.vector_store import SearchResult
from evaluation.evaluate_retrieval import contains_expected_passage, evaluate_retrieval


class FakeRetriever:
    def retrieve(self, question: str, top_k: int = 4) -> list[SearchResult]:
        return [SearchResult("faa-page-7", "faa_ac_43_204.pdf", 7, "Visual inspection provides early detection of defects before critical size.", 0.1)]


def test_contains_expected_passage_requires_document_and_all_terms() -> None:
    results = FakeRetriever().retrieve("example")
    assert contains_expected_passage(results, "faa_ac_43_204.pdf", ["early detection", "critical size"])
    assert not contains_expected_passage(results, "other.pdf", ["early detection"])


def test_evaluate_retrieval_reports_hits_and_misses() -> None:
    questions = [
        {"id": "hit", "question": "How are defects found?", "expected_document": "faa_ac_43_204.pdf", "expected_terms": ["early detection"]},
        {"id": "miss", "question": "What is the speed?", "expected_document": "faa_ac_43_204.pdf", "expected_terms": ["maximum speed"]},
    ]
    assert [record.hit for record in evaluate_retrieval(FakeRetriever(), questions, 3)] == [True, False]
