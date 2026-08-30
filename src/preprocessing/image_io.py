"""Image loading and validation (first stage of the pipeline diagram).

Handles the edge cases from Phase 9 that happen before OCR even runs:
a missing file, a zero-byte file, a file that isn't actually decodable
as an image, or an image so small it cannot plausibly contain readable
receipt text. These are reported as structured `ImageValidationError`s
rather than raw exceptions so the pipeline can turn them into a
low-confidence/"missing" record instead of crashing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MIN_PLAUSIBLE_DIMENSION = 50  # px; below this, "text" would be unrecognisable noise


class ImageValidationError(Exception):
    pass


def load_and_validate_image(path: str | Path) -> np.ndarray:
    path = Path(path)

    if not path.exists():
        raise ImageValidationError(f"file does not exist: {path}")

    if path.stat().st_size == 0:
        raise ImageValidationError(f"file is empty: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError(f"file could not be decoded as an image: {path}")

    h, w = image.shape[:2]
    if h < MIN_PLAUSIBLE_DIMENSION or w < MIN_PLAUSIBLE_DIMENSION:
        raise ImageValidationError(
            f"image too small to plausibly contain receipt text ({w}x{h}): {path}"
        )

    return image
