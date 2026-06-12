from __future__ import annotations

import logging

from rag.observability.logging import get_logger, setup_logging


def test_setup_logging_does_not_raise():
    setup_logging("INFO")
    setup_logging("DEBUG", json_format=False)


def test_get_logger_returns_something():
    logger = get_logger("test.module")
    assert logger is not None


def test_get_logger_stdlib_fallback():
    logger = get_logger("fallback.test")
    assert hasattr(logger, "info") or hasattr(logger, "bind")


def test_setup_logging_invalid_level():
    setup_logging("INVALID_LEVEL")
