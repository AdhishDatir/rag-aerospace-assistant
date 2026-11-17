"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared RAG resources once and release HTTP resources on shutdown."""
    configure_logging()

    logger.info("Loading RAG resources", extra={"event": "startup_started"})

    embedding_service = EmbeddingService()
    vector_store = ChromaVectorStore(Path(settings.chroma_path))

    app.state.retriever = Retriever(embedding_service, vector_store)
    app.state.ollama_client = httpx.Client(timeout=120.0)
    app.state.qa_service = QAService(
        client=app.state.ollama_client,
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        max_evidence_distance=settings.max_evidence_distance,
    )

    logger.info("RAG resources ready", extra={"event": "startup_completed"})

    yield

    app.state.ollama_client.close()
    logger.info("RAG resources released", extra={"event": "shutdown_completed"})


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A grounded question-answering assistant for aerospace documents.",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a lightweight status response used by tests and deployments."""
    return {"status": "ok"}
