"""Tests for the deterministic low-relevance answer fallback."""

from app.retrieval.vector_store import SearchResult
from app.services.qa_service import NOT_FOUND_MESSAGE, QAService


class FailingClient:
    def post(self, url: str, json: dict[str, object]) -> object:
        raise AssertionError("The local model should not be called for weak evidence")


def test_weak_retrieval_returns_not_found_without_model_call() -> None:
    service = QAService(FailingClient(), "qwen2.5:3b", "http://localhost:11434", max_evidence_distance=0.75)
    weak_result = SearchResult("weak-match", "faa_ac_43_204.pdf", 1, "Aircraft inspection guidance.", 0.95)
    result = service.answer_question("What is the speed of a rocket?", [weak_result])
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []
