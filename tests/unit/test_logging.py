"""Structured logging contract tests."""

import json
import logging

import pytest

from codesearch_ft.structured_logging import JsonLogFormatter, configure_logging


def test_json_formatter_emits_timestamp_level_message_and_allowlisted_context() -> None:
    record = logging.LogRecord(
        name="codesearch_ft.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="index operation started",
        args=(),
        exc_info=None,
    )
    record.event = "index.start"
    record.request_id = "request-123"
    record.secret = "must-not-be-serialized"

    rendered = JsonLogFormatter().format(record)
    payload = json.loads(rendered)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "codesearch_ft.test"
    assert payload["message"] == "index operation started"
    assert payload["event"] == "index.start"
    assert payload["request_id"] == "request-123"
    assert "secret" not in rendered


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unsupported log level"):
        configure_logging("TRACE")


def test_configure_logging_sets_json_stream_handler() -> None:
    configure_logging("WARNING")
    logger = logging.getLogger("codesearch_ft")

    assert logger.level == logging.WARNING
    assert logger.propagate is False
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0].formatter, JsonLogFormatter)

    logger.handlers.clear()
    logger.propagate = True
