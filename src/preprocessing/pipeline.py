"""Image preprocessing (Phase 3).

Individual, testable operations, plus a small number of named
"strategies" that combine them. The pipeline does not assume any single
strategy is universally best - see src/pipeline/run.py for the multi-pass
policy that picks a strategy per image based on measured OCR confidence,
and experiments/run_preprocessing_experiment.py for the comparison across
a sample of the dataset.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import cv2
import numpy as np

MAX_DIMENSION = 2200  # cap very large phone-camera images before processing
MIN_DIMENSION = 900   # upscale very small images; small text hurts OCR


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_for_ocr(image: np.ndarray) -> np.ndarray:
    """Rescale so the longer side is in a range Tesseract works well with.

    Both directions matter: SROIE scans can be very tall/high-DPI (up to
    4032px), which slows OCR for no accuracy benefit, while some phone
    photos are small enough that text strokes become too thin to
    recognise reliably.
    """

    h, w = image.shape[:2]
    longest = max(h, w)

    if longest > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest
    elif longest < MIN_DIMENSION:
        scale = MIN_DIMENSION / longest
    else:
        scale = 1.0

    if scale == 1.0:
        return image

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=interp)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting (shadows, glare gradients).

    Estimates the background by heavily blurring the image, then divides
    it out. This targets the "lighting inconsistencies" failure mode
    (e.g. receipts photographed at an angle under a single light source)
    without touching genuinely dark ink strokes, which are narrow
    relative to the blur kernel.
    """

    background = cv2.medianBlur(gray, 31)
    background = np.where(background == 0, 1, background)  # avoid div-by-zero
    normalized = cv2.divide(gray, background, scale=255)
    return normalized


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate the dominant text skew angle in degrees.

    Uses the minimum-area bounding rectangle of dark ("text") pixels
    after Otsu thresholding. This is a standard, dependency-light
    approach that works well on single-block receipt images; it is not
    a full document-layout deskewer and is not expected to handle
    curved or multi-column text.
    """

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 20:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV's minAreaRect angle convention needs normalising into a
    # "how far off horizontal is this text" value in [-45, 45].
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Receipts are rarely skewed more than ~20 degrees in this dataset;
    # anything past that is far more likely a mis-estimate from a
    # non-text-dominated image than a genuinely rotated receipt.
    if abs(angle) > 20:
        return 0.0
    return float(angle)


def deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.3:
        return gray, 0.0

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


# ---------------------------------------------------------------------------
# Named strategies. Each takes a raw BGR/gray image and returns
# (processed_image, metadata). "raw" is included deliberately as a
# baseline: Phase 3 explicitly asks us not to assume preprocessing
# always helps, so the pipeline needs a true no-op to compare against.
# ---------------------------------------------------------------------------


def strategy_raw(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    gray = to_grayscale(image)
    gray = resize_for_ocr(gray)
    return gray, {"steps": ["grayscale", "resize"]}


def strategy_clahe_deskew(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    gray = to_grayscale(image)
    gray = resize_for_ocr(gray)
    gray = enhance_contrast(gray)
    gray, angle = deskew(gray)
    return gray, {"steps": ["grayscale", "resize", "clahe", "deskew"], "skew_angle": angle}


def strategy_full(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    gray = to_grayscale(image)
    gray = resize_for_ocr(gray)
    gray = normalize_illumination(gray)
    gray = denoise(gray)
    gray = enhance_contrast(gray)
    gray, angle = deskew(gray)
    gray = adaptive_threshold(gray)
    return gray, {
        "steps": ["grayscale", "resize", "illumination_norm", "denoise", "clahe", "deskew", "adaptive_threshold"],
        "skew_angle": angle,
    }


STRATEGIES: Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, Dict]]] = {
    "raw": strategy_raw,
    "clahe_deskew": strategy_clahe_deskew,
    "full": strategy_full,
}


def apply_strategy(image: np.ndarray, strategy_name: str) -> Tuple[np.ndarray, Dict]:
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown preprocessing strategy: {strategy_name}")
    return STRATEGIES[strategy_name](image)
