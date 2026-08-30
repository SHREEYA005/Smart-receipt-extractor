"""Financial summary aggregation (Phase 11).

Distinguishes *reported* totals (every total we extracted, regardless of
confidence) from *included* totals (only the ones confident enough to
trust in a spend figure). Blindly summing every extracted number - some
of which come from low-confidence OCR reads - would silently distort
the headline number; the assignment explicitly warns against that.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def build_financial_summary(records: List[Dict[str, Any]], inclusion_threshold: float) -> Dict[str, Any]:
    total_receipts = len(records)
    processed_receipts = sum(1 for r in records if r["total_amount"]["value"] is not None or r["store_name"]["value"] is not None or r["date"]["value"] is not None)

    reported_total_spend = 0.0
    included_total_spend = 0.0
    included_count = 0
    excluded_low_confidence_count = 0

    spend_per_store_reported: Dict[str, float] = defaultdict(float)
    spend_per_store_included: Dict[str, float] = defaultdict(float)
    transactions_per_store: Dict[str, int] = defaultdict(int)

    for r in records:
        total_field = r["total_amount"]
        value = total_field["value"]
        if value is None:
            continue

        value = float(value)
        store_name = r["store_name"]["value"] or "UNKNOWN"

        reported_total_spend += value
        spend_per_store_reported[store_name] += value

        if total_field["confidence"] >= inclusion_threshold and total_field["status"] != "ambiguous":
            included_total_spend += value
            included_count += 1
            spend_per_store_included[store_name] += value
            transactions_per_store[store_name] += 1
        else:
            excluded_low_confidence_count += 1

    return {
        "total_receipts_processed": total_receipts,
        "receipts_with_any_extracted_field": processed_receipts,
        "reported_total_spend": round(reported_total_spend, 2),
        "included_total_spend": round(included_total_spend, 2),
        "number_of_transactions_included": included_count,
        "number_of_totals_excluded_low_confidence": excluded_low_confidence_count,
        "spend_per_store_reported": {k: round(v, 2) for k, v in sorted(spend_per_store_reported.items())},
        "spend_per_store_included": {k: round(v, 2) for k, v in sorted(spend_per_store_included.items())},
        "transactions_per_store_included": dict(sorted(transactions_per_store.items())),
        "inclusion_confidence_threshold": inclusion_threshold,
        "policy_note": (
            "included_* figures only count total_amount fields with confidence >= "
            f"{inclusion_threshold} and status != 'ambiguous'. reported_* figures include "
            "every extracted total regardless of confidence, for transparency."
        ),
    }
