import cv2
import numpy as np

from src.preprocessing.pipeline import (
    adaptive_threshold,
    deskew,
    enhance_contrast,
    estimate_skew_angle,
    resize_for_ocr,
    to_grayscale,
)


def make_text_like_image(angle_deg: float = 0.0, size=(400, 800)) -> np.ndarray:
    """Synthetic image with a few dark rectangular 'text' blocks, optionally rotated.

    Real text detection is hard to fabricate deterministically, but skew
    estimation only needs a dark-pixel mass with a clear principal axis -
    a handful of horizontal bars stands in for a line of text well
    enough to test the deskew math itself.
    """
    img = np.full(size, 255, dtype=np.uint8)
    for y in range(50, 350, 40):
        cv2.rectangle(img, (50, y), (700, y + 15), 0, -1)

    if angle_deg != 0.0:
        h, w = img.shape
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=255)

    return img


def test_to_grayscale_converts_bgr():
    color = np.zeros((10, 10, 3), dtype=np.uint8)
    color[:, :, 0] = 100  # blue channel
    gray = to_grayscale(color)
    assert gray.ndim == 2
    assert gray.shape == (10, 10)


def test_to_grayscale_passthrough_for_already_gray():
    gray_in = np.zeros((10, 10), dtype=np.uint8)
    out = to_grayscale(gray_in)
    assert out is gray_in


def test_resize_for_ocr_upscales_small_images():
    small = np.zeros((300, 400), dtype=np.uint8)
    resized = resize_for_ocr(small)
    assert max(resized.shape) >= 900


def test_resize_for_ocr_downscales_huge_images():
    huge = np.zeros((3000, 5000), dtype=np.uint8)
    resized = resize_for_ocr(huge)
    assert max(resized.shape) <= 2200


def test_resize_for_ocr_no_op_for_reasonable_size():
    ok = np.zeros((1000, 1500), dtype=np.uint8)
    resized = resize_for_ocr(ok)
    assert resized.shape == ok.shape


def test_estimate_skew_angle_near_zero_for_upright_text():
    img = make_text_like_image(angle_deg=0.0)
    angle = estimate_skew_angle(img)
    assert abs(angle) < 2.0


def test_estimate_skew_angle_detects_rotation():
    img = make_text_like_image(angle_deg=10.0)
    angle = estimate_skew_angle(img)
    # Sign convention aside, a meaningful rotation should be detected,
    # not silently reported as ~0.
    assert abs(angle) > 3.0


def test_deskew_reduces_measured_skew():
    img = make_text_like_image(angle_deg=8.0)
    corrected, applied_angle = deskew(img)
    assert applied_angle != 0.0
    residual = estimate_skew_angle(corrected)
    assert abs(residual) < abs(applied_angle)


def test_deskew_is_near_noop_on_upright_image():
    img = make_text_like_image(angle_deg=0.0)
    corrected, applied_angle = deskew(img)
    assert applied_angle == 0.0
    assert corrected.shape == img.shape


def test_enhance_contrast_preserves_shape_and_dtype():
    img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    out = enhance_contrast(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_adaptive_threshold_output_is_binary():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    out = adaptive_threshold(img)
    unique_values = set(np.unique(out).tolist())
    assert unique_values.issubset({0, 255})
