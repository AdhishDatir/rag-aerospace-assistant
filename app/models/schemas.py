"""Pydantic models used by the API and answer-generation service."""

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A verifiable source location for an answer."""

    document_name: str
    page_number: int = Field(ge=1)


class AnswerResult(BaseModel):
    """A grounded answer and the source locations used to produce it."""

    answer: str
    citations: list[Citation]


class AskRequest(BaseModel):
    """Validated request body for the question-answering endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1_000)
    top_k: int = Field(default=4, ge=1, le=10)


class AskResponse(AnswerResult):
    """Answer payload returned by the API."""

    retrieval_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)