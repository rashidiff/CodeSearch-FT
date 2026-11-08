"""Structured standard-library logging with an allowlist of context fields."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

_SAFE_CONTEXT_FIELDS = ("event", "operation", "request_id", "duration_ms")


class JsonLogFormatter(logging.Formatter):
    """Render a stable JSON record without arbitrary LogRecord extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _SAFE_CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception_type"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else "Error"
            )
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure one JSON stream handler for application logs."""
    normalized_level = level.upper()
    numeric_level = logging.getLevelNamesMapping().get(normalized_level)
    if numeric_level is None:
        raise ValueError(
            f"Unsupported log level '{level}'. Use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    logger = logging.getLogger("codesearch_ft")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(numeric_level)
    logger.propagate = False
