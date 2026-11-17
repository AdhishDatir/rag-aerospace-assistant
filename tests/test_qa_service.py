"""Tests for grounded local Ollama answer generation."""

from app.retrieval.vector_store import SearchResult
from app.models.schemas import Citation
from app.services.qa_service import NOT_FOUND_MESSAGE, QAService


class FakeHTTPResponse:
    """Minimal stand-in for an Ollama HTTP response."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def raise_for_status(self) -> None:
        """Simulate a successful HTTP response."""

    def json(self) -> dict[str, str]:
        """Return the shape used by Ollama's generate endpoint."""
        return {"response": self.answer}


class FakeHTTPClient:
    """Captures requests and returns a predictable local-model response."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.request: dict[str, object] | None = None

    def post(self, url: str, json: dict[str, object]) -> FakeHTTPResponse:
        self.request = {"url": url, "json": json}
        return FakeHTTPResponse(self.answer)


def test_answer_question_returns_metadata_based_citations() -> None:
    """Citations must come from retrieved chunks, not model output."""
    client = FakeHTTPClient(
        "The circular provides technical information for aircraft visual inspections."
    )
    service = QAService(
        client=client,
        model="qwen2.5:3b",
        base_url="http://localhost:11434",
    )
    chunks = [
        SearchResult(
            chunk_id="faa-page-1",
            document_name="faa_ac_43_204.pdf",
            page_number=1,
            text="This advisory circular provides technical information to persons "
            "conducting a visual inspection of aircraft.",
            distance=0.05,
        )
    ]

    result = service.answer_question("What is the circular's purpose?", chunks)

    assert result.answer.startswith("The circular provides")
    assert result.citations[0].document_name == "faa_ac_43_204.pdf"
    assert result.citations[0].page_number == 1
    assert client.request is not None
    assert client.request["url"] == "http://localhost:11434/api/generate"


def test_answer_question_returns_fallback_without_evidence() -> None:
    """No retrieved evidence means no local-model request."""
    client = FakeHTTPClient("This response should never be used.")
    service = QAService(
        client=client,
        model="qwen2.5:3b",
        base_url="http://localhost:11434",
    )

    result = service.answer_question("What is the aircraft's maximum speed?", [])

    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []
    assert client.request is None


def test_answer_question_deduplicates_same_page_citations() -> None:
    """Two chunks from one page should produce a single source citation."""
    client = FakeHTTPClient("Visual inspection is a documented procedure.")
    service = QAService(client, "qwen2.5:3b", "http://localhost:11434")
    chunks = [
        SearchResult("one", "faa.pdf", 7, "Visual inspection evidence.", 0.1),
        SearchResult("two", "faa.pdf", 7, "More evidence from the same page.", 0.2),
    ]

    result = service.answer_question("What is visual inspection?", chunks)

    assert result.citations == [Citation(document_name="faa.pdf", page_number=7)]
