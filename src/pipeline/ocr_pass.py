"""Multi-pass OCR / preprocessing fallback policy (Phase 8).

Runs OCR on the cheapest preprocessing first; only pays for heavier
preprocessing when the cheap pass looks unreliable. This keeps the
common case (already-clean scans, which is most of this dataset) fast,
while still giving noisy phone photos a chance at a better pass.

Decision rule, in order:
  1. Try each strategy in ``strategies`` (config-ordered).
  2. Stop as soon as one clears both a minimum mean confidence and a
     minimum token count (very high confidence on 2 tokens is not a
     reliable receipt read).
  3. If none clear the bar, use whichever pass scored highest anyway -
     we still want to return the best available answer, just flagged
     as such via ``meta.ocr_pass`` and low downstream field confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.ocr.engine import OCREngine, OCRResult
from src.preprocessing.pipeline import apply_strategy


@dataclass
class PassAttempt:
    strategy: str
    result: OCRResult
    preprocessing_meta: Dict


def run_multi_pass_ocr(
    image: np.ndarray,
    engine: OCREngine,
    strategies: List[str],
    min_confidence: float,
    min_tokens: int,
) -> PassAttempt:
    attempts: List[PassAttempt] = []

    for strategy_name in strategies:
        processed, prep_meta = apply_strategy(image, strategy_name)
        result = engine.run(processed)
        attempt = PassAttempt(strategy_name, result, prep_meta)
        attempts.append(attempt)

        if result.mean_confidence >= min_confidence and result.token_count >= min_tokens:
            return attempt

    # Nothing cleared the bar; return the best-scoring attempt so we still
    # produce output, rather than silently keeping only the last one tried.
    return max(attempts, key=lambda a: (a.result.mean_confidence, a.result.token_count))
