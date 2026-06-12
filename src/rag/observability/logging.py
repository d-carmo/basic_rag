from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(level: str = "INFO", *, json_format: bool = True) -> None:
    """Configure logging — structlog if available, stdlib fallback."""
    try:
        import structlog  # type: ignore[import-not-found]

        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, level.upper(), logging.INFO),
        )
    except ImportError:
        fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
        logging.basicConfig(
            format=fmt,
            stream=sys.stdout,
            level=getattr(logging, level.upper(), logging.INFO),
        )


def get_logger(name: str, **initial_values: Any) -> Any:
    """Return a structlog logger if available, else stdlib logger."""
    try:
        import structlog  # type: ignore[import-not-found]

        log = structlog.get_logger(name)
        if initial_values:
            log = log.bind(**initial_values)
        return log
    except ImportError:
        return logging.getLogger(name)
