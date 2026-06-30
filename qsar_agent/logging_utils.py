"""Structured logging helpers for QSAR Agent."""

from __future__ import annotations

import logging
from typing import Any


def get_logger(name: str = "qsar_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def append_log(logs: list[str], message: str, level: str = "INFO") -> None:
    logs.append(f"[{level}] {message}")


def append_warning(warnings: list[str], message: str) -> None:
    warnings.append(message)
