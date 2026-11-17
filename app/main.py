"""FastAPI application entry point."""

from fastapi import FastAPI

from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A grounded question-answering assistant for aerospace documents.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a lightweight status response used by tests and deployments."""
    return {"status": "ok"}