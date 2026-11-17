"""FastAPI routes for grounded aerospace document questions."""

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from app.models.schemas import AskRequest, AskResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a grounded question about indexed documents",
)
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    """Retrieve evidence, generate a grounded answer, and return citations."""
    started_at = time.perf_counter()

    try:
        chunks = request.app.state.retriever.retrieve(
            payload.question,
            top_k=payload.top_k,
        )
        result = request.app.state.qa_service.answer_question(
            payload.question,
            chunks,
        )

        latency_ms = round((time.perf_counter() - started_at) * 1_000, 2)

        logger.info(
            "Question answered",
            extra={
                "event": "ask_completed",
                "latency_ms": latency_ms,
                "retrieval_count": len(chunks),
                "question_length": len(payload.question),
            },
        )

        return AskResponse(
            answer=result.answer,
            citations=result.citations,
            retrieval_count=len(chunks),
            latency_ms=latency_ms,
        )

    except httpx.HTTPError as error:
        logger.exception(
            "Local Ollama request failed",
            extra={
                "event": "ask_failed",
                "question_length": len(payload.question),
                "error_type": type(error).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local Ollama model is unavailable. Confirm Ollama is running.",
        ) from error

    except Exception as error:
        logger.exception(
            "Question processing failed",
            extra={
                "event": "ask_failed",
                "question_length": len(payload.question),
                "error_type": type(error).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process the question.",
        ) from error