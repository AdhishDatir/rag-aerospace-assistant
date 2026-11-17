"""Structured JSON logging for API requests and failures."""

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Format application logs as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize selected log fields into JSON."""
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field_name in (
            "event",
            "latency_ms",
            "retrieval_count",
            "question_length",
            "error_type",
        ):
            if hasattr(record, field_name):
                payload[field_name] = getattr(record, field_name)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging() -> None:
    """Configure one JSON console handler for the application."""
    root_logger = logging.getLogger()

    if any(
        getattr(handler, "is_rag_json_handler", False)
        for handler in root_logger.handlers
    ):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.is_rag_json_handler = True

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)