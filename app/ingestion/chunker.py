"""Split page-aware PDF text into overlapping retrieval chunks."""

from dataclasses import dataclass

from app.ingestion.pdf_loader import PageText


@dataclass(frozen=True)
class TextChunk:
    """A searchable piece of page text with citation metadata."""

    chunk_id: str
    document_name: str
    page_number: int
    text: str


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be at least zero and smaller than chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Avoid cutting a word in half when a space is available.
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        # Move back slightly so neighboring chunks share useful context.
        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Convert extracted PDF pages into searchable chunks.

    Chunks never cross page boundaries, preserving accurate page citations.
    """
    chunks: list[TextChunk] = []

    for page in pages:
        page_chunks = split_text(
            text=page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, text in enumerate(page_chunks, start=1):
            chunks.append(
                TextChunk(
                    chunk_id=(
                        f"{page.document_name}-page-{page.page_number}-chunk-{chunk_index}"
                    ),
                    document_name=page.document_name,
                    page_number=page.page_number,
                    text=text,
                )
            )

    return chunks