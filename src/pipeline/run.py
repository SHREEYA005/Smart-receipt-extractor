"""End-to-end per-receipt pipeline (ties together every phase).

    image -> validate -> preprocess (multi-pass) -> OCR -> normalize
    -> extract candidates -> validate/consistency -> confidence score
    -> structured JSON record

Every stage is allowed to fail gracefully: a bad image, an OCR pass that
returns nothing, or a field with no plausible candidate all produce a
valid schema record with null/low-confidence fields and a warning,
never an unhandled exception that stops the batch (Phase 9).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from src.confidence.scoring import score_date, score_items, score_store_name, score_total
from src.extraction.date_extraction import find_date_candidates
from src.extraction.items import find_item_candidates
from src.extraction.lines import build_lines
from src.extraction.store_name import select_store_name
from src.extraction.total_amount import find_total_candidates, select_total
from src.ocr.engine import get_engine
from src.pipeline.ocr_pass import run_multi_pass_ocr
from src.preprocessing.image_io import ImageValidationError, load_and_validate_image
from src.utils.schema import FieldValue, ItemRecord, build_receipt_record
from src.validation.validators import check_item_total_consistency

logger = logging.getLogger("receipt_ai")


def _empty_record(receipt_id: str, reason: str, warnings: List[str], meta: Dict[str, Any]) -> Dict[str, Any]:
    missing = FieldValue(None, 0.0, "missing")
    return build_receipt_record(receipt_id, missing, missing, [], missing, warnings, meta)


def process_receipt(image_path: str | Path, config: Dict[str, Any]) -> Dict[str, Any]:
    receipt_id = Path(image_path).stem
    warnings: List[str] = []
    start = time.time()

    try:
        image = load_and_validate_image(image_path)
    except ImageValidationError as e:
        logger.warning("receipt=%s status=rejected reason=%s", receipt_id, e)
        meta = {"processing_time_sec": round(time.time() - start, 3), "error": str(e)}
        return _empty_record(receipt_id, str(e), [f"image could not be processed: {e}"], meta)

    ocr_cfg = config["ocr"]
    engine = get_engine(ocr_cfg["engine"], lang=ocr_cfg["lang"], psm=ocr_cfg["psm"], oem=ocr_cfg["oem"])

    prep_cfg = config["preprocessing"]
    attempt = run_multi_pass_ocr(
        image,
        engine,
        strategies=prep_cfg["strategies"],
        min_confidence=prep_cfg["min_acceptable_confidence"],
        min_tokens=prep_cfg["min_tokens"],
    )
    ocr_result = attempt.result

    if ocr_result.token_count == 0:
        warnings.append("OCR returned no recognisable text across all preprocessing strategies")
        meta = {
            "processing_time_sec": round(time.time() - start, 3),
            "ocr_pass": attempt.strategy,
            "ocr_mean_confidence": ocr_result.mean_confidence,
        }
        return _empty_record(receipt_id, "no OCR text", warnings, meta)

    lines = build_lines(ocr_result.tokens)
    conf_weights = config["confidence"]["weights"]
    thresholds = config["confidence"]["thresholds"]

    # --- Store name -----------------------------------------------------
    store_candidate = select_store_name(lines)
    store_field, store_components = score_store_name(store_candidate, conf_weights["store_name"], thresholds)
    if store_field.status in ("missing", "low"):
        warnings.append("store name could not be extracted with confidence")

    # --- Date -------------------------------------------------------------
    date_candidates = find_date_candidates(ocr_result.full_text)
    date_field, date_components = score_date(date_candidates, conf_weights["date"], thresholds)
    if date_field.status == "ambiguous":
        warnings.append("date is ambiguous between DD/MM and MM/DD interpretations")
    elif date_field.status == "missing":
        warnings.append("no valid date found on receipt")

    # --- Total & items (extracted together: item search is bounded by
    #     where the total keyword appears) -------------------------------
    total_candidates = find_total_candidates(lines)
    best_total_line_idx = None
    if total_candidates:
        best_total_line_idx = max(
            total_candidates, key=lambda c: 0.7 * c.keyword_strength + 0.3 * (c.ocr_conf / 100.0)
        ).line_index

    item_candidates = find_item_candidates(lines, best_total_line_idx)
    item_prices = [c.price for c in item_candidates]

    provisional_total = select_total(total_candidates)
    consistency = check_item_total_consistency(
        item_prices, provisional_total.value if provisional_total else None
    )

    total_field, total_components = score_total(
        total_candidates, conf_weights["total_amount"], thresholds, consistency
    )
    if total_field.status == "missing":
        warnings.append("total amount could not be located")
    if consistency.signal == "mismatch":
        warnings.append(
            f"item prices (sum={consistency.item_sum}) do not reconcile with total "
            f"(total={consistency.total_value}, delta={consistency.delta}); "
            "could be tax/discount/rounding or an extraction error"
        )

    item_records: List[ItemRecord] = score_items(item_candidates, conf_weights["item"], thresholds)
    if not item_records:
        warnings.append("no line items could be reliably extracted")

    meta = {
        "processing_time_sec": round(time.time() - start, 3),
        "ocr_pass": attempt.strategy,
        "ocr_mean_confidence": round(ocr_result.mean_confidence, 2),
        "ocr_token_count": ocr_result.token_count,
        "preprocessing_steps": attempt.preprocessing_meta.get("steps", []),
        "consistency_signal": consistency.signal,
        "confidence_components": {
            "store_name": store_components,
            "date": date_components,
            "total_amount": total_components,
        },
    }

    record = build_receipt_record(receipt_id, store_field, date_field, item_records, total_field, warnings, meta)
    logger.info(
        "receipt=%s ocr_pass=%s store_status=%s date_status=%s total_status=%s items=%d",
        receipt_id, attempt.strategy, store_field.status, date_field.status, total_field.status, len(item_records),
    )
    return record
