"""Evaluation (Phase 12).

No ground-truth annotations ship with this dataset (checked directly -
see data/README.md). So this module does NOT compute or claim any
accuracy percentage. Instead it reports what can honestly be measured
without labels:

  - field coverage (how often each field produced a non-null value)
  - OCR confidence distribution
  - which preprocessing pass ended up being used, and how often
  - the item/total consistency-check pass rate
  - a sample-based manual review template, so a human can spot-check
    a subset and *establish* real accuracy numbers if they choose to

Phase 13 (confidence calibration) is deliberately not attempted for the
same reason: calibration analysis requires labelled correctness, which
this dataset does not provide.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

STATUS_ORDER = ["high", "medium", "low", "ambiguous", "missing"]


def _status_breakdown(records: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counter = Counter(r[field]["status"] for r in records)
    return {s: counter.get(s, 0) for s in STATUS_ORDER}


def build_evaluation_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"note": "no records to evaluate"}

    coverage = {
        "store_name": sum(1 for r in records if r["store_name"]["value"] is not None) / n,
        "date": sum(1 for r in records if r["date"]["value"] is not None) / n,
        "total_amount": sum(1 for r in records if r["total_amount"]["value"] is not None) / n,
        "at_least_one_item": sum(1 for r in records if len(r["items"]) > 0) / n,
    }

    ocr_confidences = [r["meta"]["ocr_mean_confidence"] for r in records if "ocr_mean_confidence" in r["meta"]]
    ocr_stats = {
        "mean": round(sum(ocr_confidences) / len(ocr_confidences), 2) if ocr_confidences else None,
        "min": round(min(ocr_confidences), 2) if ocr_confidences else None,
        "max": round(max(ocr_confidences), 2) if ocr_confidences else None,
    }

    pass_usage = Counter(r["meta"].get("ocr_pass", "unknown") for r in records)

    consistency_signals = Counter(r["meta"].get("consistency_signal", "unknown") for r in records)
    evaluable_consistency = consistency_signals.get("match", 0) + consistency_signals.get("close", 0) + consistency_signals.get("mismatch", 0)
    consistency_pass_rate = None
    if evaluable_consistency > 0:
        consistency_pass_rate = round(
            (consistency_signals.get("match", 0) + consistency_signals.get("close", 0)) / evaluable_consistency, 3
        )

    return {
        "measured_results": {
            "receipts_evaluated": n,
            "field_coverage": {k: round(v, 3) for k, v in coverage.items()},
            "field_status_breakdown": {
                field: _status_breakdown(records, field) for field in ("store_name", "date", "total_amount")
            },
            "ocr_confidence": ocr_stats,
            "preprocessing_pass_usage": dict(pass_usage),
            "item_total_consistency_signal_counts": dict(consistency_signals),
            "item_total_consistency_pass_rate": consistency_pass_rate,
        },
        "qualitative_observations": {
            "confidence_calibration": (
                "Not computed. No ground-truth labels are available for this dataset, so we "
                "cannot check whether 'confidence' correlates with actual correctness - only "
                "with the internal signals (OCR score, keyword match, consistency) that feed it. "
                "See docs/technical_documentation.md, Limitations."
            ),
            "extraction_accuracy": (
                "Not computed as a percentage against ground truth, because no ground truth "
                "exists for this dataset. Use the generated manual review sample "
                "(outputs/reports/manual_review_sample.csv) to establish real accuracy figures "
                "by hand-checking a sample of receipts against their source images."
            ),
        },
    }


def build_manual_review_sample(records: List[Dict[str, Any]], sample_size: int, seed: int = 42) -> List[Dict[str, Any]]:
    """A CSV template for a human reviewer to fill in ground truth by hand.

    This is the honest substitute for accuracy metrics we cannot compute
    automatically: pick a reproducible random sample, lay out what the
    pipeline extracted next to blank "correct?" columns.
    """

    rng = random.Random(seed)
    sample = rng.sample(records, min(sample_size, len(records)))

    rows = []
    for r in sample:
        rows.append(
            {
                "receipt_id": r["receipt_id"],
                "extracted_store_name": r["store_name"]["value"],
                "store_name_correct?": "",
                "extracted_date": r["date"]["value"],
                "date_correct?": "",
                "extracted_total": r["total_amount"]["value"],
                "total_correct?": "",
                "num_items_extracted": len(r["items"]),
                "items_look_reasonable?": "",
                "notes": "",
            }
        )
    return rows


def write_manual_review_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
