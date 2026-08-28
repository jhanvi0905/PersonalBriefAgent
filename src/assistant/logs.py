"""Single logger for the DAG. CLI prints it to stderr; the web server also
fans it out to the browser over SSE."""

from __future__ import annotations

import logging

logger = logging.getLogger("assistant")


def configure_logging(level: int = logging.INFO) -> None:
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[dag] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
