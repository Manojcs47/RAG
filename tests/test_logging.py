from __future__ import annotations

from research_navigator.logging import configure_logging, get_logger


def test_logging_smoke() -> None:
    configure_logging("INFO", json_logs=True)
    log = get_logger("test")
    log.info("hello", foo="bar")  # must not raise
