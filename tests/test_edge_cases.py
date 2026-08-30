import numpy as np
import pytest

from src.preprocessing.image_io import ImageValidationError, load_and_validate_image
from src.utils.config import load_config
from src.utils.schema import validate_receipt_record


def test_missing_file_raises_validation_error_not_generic_exception(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ImageValidationError):
        load_and_validate_image(missing)


def test_empty_file_raises_validation_error(tmp_path):
    empty_file = tmp_path / "empty.jpg"
    empty_file.write_bytes(b"")
    with pytest.raises(ImageValidationError):
        load_and_validate_image(empty_file)


def test_non_image_file_raises_validation_error(tmp_path):
    fake = tmp_path / "not_an_image.jpg"
    fake.write_text("this is definitely not JPEG data")
    with pytest.raises(ImageValidationError):
        load_and_validate_image(fake)


def test_tiny_image_raises_validation_error(tmp_path):
    import cv2

    tiny = np.zeros((5, 5, 3), dtype=np.uint8)
    path = tmp_path / "tiny.jpg"
    cv2.imwrite(str(path), tiny)
    with pytest.raises(ImageValidationError):
        load_and_validate_image(path)


def test_process_receipt_handles_missing_image_gracefully(tmp_path):
    from src.pipeline.run import process_receipt

    config = load_config()
    record = process_receipt(tmp_path / "nonexistent.jpg", config)

    # Must return a schema-valid record, not raise.
    assert validate_receipt_record(record) == []
    assert record["store_name"]["status"] == "missing"
    assert record["total_amount"]["status"] == "missing"
    assert record["warnings"]


def test_process_receipt_handles_blank_image_gracefully(tmp_path):
    import cv2

    from src.pipeline.run import process_receipt

    blank = np.full((1000, 700, 3), 255, dtype=np.uint8)  # plain white, no text at all
    path = tmp_path / "blank.jpg"
    cv2.imwrite(str(path), blank)

    config = load_config()
    record = process_receipt(path, config)

    assert validate_receipt_record(record) == []
    # A blank image should yield low-confidence/missing fields, not a crash.
    assert record["total_amount"]["value"] is None or record["total_amount"]["confidence"] < 0.7


def test_config_defaults_load_without_a_file():
    config = load_config(None)
    assert config["ocr"]["engine"] == "tesseract"
    assert 0.0 < config["confidence"]["thresholds"]["medium"] < config["confidence"]["thresholds"]["high"] < 1.0


def test_config_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no_such_config.yaml")
