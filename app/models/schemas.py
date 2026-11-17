"""Pydantic models shared by the answer-generation layer and API."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A verifiable source location for an answer."""

    document_name: str
    page_number: int = Field(ge=1)


class AnswerResult(BaseModel):
    """A grounded answer and the source locations used to produce it."""

    answer: str
    citations: list[Citation]