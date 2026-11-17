"""Generate answers strictly grounded in retrieved document chunks."""

import httpx

from app.models.schemas import AnswerResult, Citation
from app.retrieval.vector_store import SearchResult

NOT_FOUND_MESSAGE = "not found in the documents"

SYSTEM_PROMPT = f"""You are an aerospace technical-document assistant.

Answer using only the provided SOURCE EXCERPTS.
Do not use outside knowledge, assumptions, or facts not present in those excerpts.
If the excerpts do not clearly answer the question, reply with exactly:
{NOT_FOUND_MESSAGE}

Keep the answer concise and factual.
"""


class QAService:
    """Create evidence-grounded answers with a local Ollama model."""

    def __init__(
        self,
        client: httpx.Client,
        model: str,
        base_url: str,
    ) -> None:
        """Receive dependencies explicitly for easy testing."""
        self.client = client
        self.model = model
        self.generate_url = f"{base_url.rstrip('/')}/api/generate"

    def answer_question(
        self,
        question: str,
        retrieved_chunks: list[SearchResult],
    ) -> AnswerResult:
        """Answer using retrieved evidence, or return the fallback."""
        if not retrieved_chunks:
            return AnswerResult(answer=NOT_FOUND_MESSAGE, citations=[])

        source_excerpts = "\n\n".join(
            (
                f"[Source {index}: {chunk.document_name}, page {chunk.page_number}]\n"
                f"{chunk.text}"
            )
            for index, chunk in enumerate(retrieved_chunks, start=1)
        )

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"SOURCE EXCERPTS:\n{source_excerpts}"
        )

        response = self.client.post(
            self.generate_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
        )
        response.raise_for_status()

        answer = str(response.json().get("response", "")).strip()

        if not answer or answer.lower() == NOT_FOUND_MESSAGE:
            return AnswerResult(answer=NOT_FOUND_MESSAGE, citations=[])

        citations = [
            Citation(
                document_name=chunk.document_name,
                page_number=chunk.page_number,
            )
            for chunk in retrieved_chunks
        ]

        return AnswerResult(answer=answer, citations=citations)