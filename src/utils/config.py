"""Configuration loading.

Everything the pipeline can tune (OCR settings, confidence weights,
thresholds, paths) lives in a single YAML file so a reviewer can change
behaviour without touching code. If no config file is given, the defaults
below are used - they are the same values documented in the README and in
docs/technical_documentation.md.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "data_dir": "data/receipts",
        "output_dir": "outputs",
    },
    "ocr": {
        "engine": "tesseract",
        "lang": "eng",
        # Tesseract page segmentation mode. 6 = "assume a single uniform
        # block of text", which works better than the default (3) for
        # narrow receipt crops.
        "psm": 6,
        "oem": 3,
    },
    "preprocessing": {
        # Ordered list of strategies tried in Phase 8's multi-pass policy.
        # The pipeline stops as soon as one strategy clears the
        # confidence floor, or falls back to the best of all of them.
        "strategies": ["raw", "clahe_deskew", "full"],
        "min_acceptable_confidence": 55.0,
        "min_tokens": 5,
    },
    "confidence": {
        "thresholds": {"high": 0.85, "medium": 0.70},
        "low_confidence_flag": 0.70,
        "weights": {
            "store_name": {"ocr": 0.35, "position": 0.30, "pattern": 0.20, "length": 0.15},
            "date": {"ocr": 0.25, "pattern": 0.45, "keyword": 0.30, "ambiguity_penalty": 0.20},
            "total_amount": {"ocr": 0.25, "keyword": 0.30, "pattern": 0.15, "consistency": 0.30},
            "item": {"ocr": 0.45, "pattern": 0.35, "position": 0.20},
        },
    },
    "aggregation": {
        # Totals below this confidence are still reported, but excluded
        # from the headline "included" spend figures (Phase 11).
        "inclusion_confidence_threshold": 0.50,
    },
    "logging": {
        "level": "INFO",
        "log_file": "outputs/reports/pipeline.log",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration, merging a user-provided YAML file over defaults.

    Any key omitted from the file falls back to DEFAULT_CONFIG, so a
    reviewer only needs to override the handful of values they care about.
    """

    if config_path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    return _deep_merge(DEFAULT_CONFIG, user_config)
