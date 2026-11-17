"""Tests for the public FastAPI question-answering endpoint."""

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import AnswerResult, Citation
from app.retrieval.vector_store import SearchResult


class FakeRetriever:
    """Return a predictable retrieved chunk."""

    def retrieve(self, question: str, top_k: int) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id="faa-page-1",
                document_name="faa_ac_43_204.pdf",
                page_number=1,
                text="The circular provides technical information for inspections.",
                distance=0.1,
            )
        ]


class FakeQAService:
    """Return a predictable grounded answer."""

    def answer_question(
        self,
        question: str,
        chunks: list[SearchResult],
    ) -> AnswerResult:
        return AnswerResult(
            answer="The circular provides technical inspection information.",
            citations=[
                Citation(
                    document_name="faa_ac_43_204.pdf",
                    page_number=1,
                )
            ],
        )


def test_ask_endpoint_returns_answer_and_citation() -> None:
    """The endpoint should return validated answer, citations, and latency."""
    app.state.retriever = FakeRetriever()
    app.state.qa_service = FakeQAService()
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={"question": "What is the purpose of visual inspection?", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("The circular provides")
    assert body["citations"] == [
        {"document_name": "faa_ac_43_204.pdf", "page_number": 1}
    ]
    assert body["retrieval_count"] == 1
    assert body["latency_ms"] >= 0


def test_ask_endpoint_rejects_blank_question() -> None:
    """Pydantic should reject a blank request before RAG processing."""
    client = TestClient(app)

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422