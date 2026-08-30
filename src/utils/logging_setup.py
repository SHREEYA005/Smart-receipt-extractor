"""Logging configuration.

Design note (Phase 16 / Phase 24): logs are meant to help a developer debug
*why* a receipt got a particular status, not to store the receipt content
itself. We deliberately log field statuses, confidence scores and timings,
but never the raw OCR text of a receipt, since receipts can contain names,
card-related account references and other personal data.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_file: str | Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("receipt_ai")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        # Avoid duplicate handlers if setup_logging is called more than once
        # (e.g. in tests that import the pipeline repeatedly).
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger
