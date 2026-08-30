"""The receipt data contract (Phase 10).

A single dataclass, ``FieldValue``, represents every confidence-aware
field in the output (store name, date, total, and each line item's
price). It is intentionally simple: value, confidence, status, and -
only for fields where it matters - a list of alternatives when the
extractor found more than one plausible candidate.

``validate_receipt_record`` is a small, dependency-free structural
check. It exists so tests and the pipeline can assert "this dict is
shaped like the contract we promised" without pulling in a full
JSON Schema library for what is a genuinely simple structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

STATUS_VALUES = {"high", "medium", "low", "missing", "ambiguous"}


@dataclass
class FieldValue:
    value: Optional[Any]
    confidence: float
    status: str
    alternatives: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "value": self.value,
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
        }
        if self.alternatives:
            d["alternatives"] = self.alternatives
        return d


@dataclass
class ItemRecord:
    name: str
    price: Optional[str]
    confidence: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "price": self.price,
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
        }


def build_receipt_record(
    receipt_id: str,
    store_name: FieldValue,
    date: FieldValue,
    items: List[ItemRecord],
    total_amount: FieldValue,
    warnings: List[str],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the final per-receipt JSON record.

    Kept flat and close to the assignment's required shape
    (store_name / date / items / total_amount), with confidence and
    status attached to each field as specified in Phase 10, plus a
    ``warnings`` list and a small ``meta`` block for traceability
    (which OCR pass was used, timing, etc.) that does not interfere
    with downstream consumers that only care about the core fields.
    """

    return {
        "receipt_id": receipt_id,
        "store_name": store_name.to_dict(),
        "date": date.to_dict(),
        "items": [item.to_dict() for item in items],
        "total_amount": total_amount.to_dict(),
        "warnings": warnings,
        "meta": meta,
    }


def validate_receipt_record(record: Dict[str, Any]) -> List[str]:
    """Return a list of schema violations (empty list = valid).

    This is a structural check, not a correctness check: it confirms the
    record has the fields the contract promises, with the right types,
    not that the extracted values are accurate.
    """

    errors: List[str] = []

    required_top = ["receipt_id", "store_name", "date", "items", "total_amount", "warnings"]
    for key in required_top:
        if key not in record:
            errors.append(f"missing top-level key: {key}")

    for field_name in ("store_name", "date", "total_amount"):
        fv = record.get(field_name)
        if not isinstance(fv, dict):
            errors.append(f"{field_name} must be an object")
            continue
        if "value" not in fv:
            errors.append(f"{field_name} missing 'value'")
        if "confidence" not in fv or not isinstance(fv["confidence"], (int, float)):
            errors.append(f"{field_name} missing/invalid 'confidence'")
        elif not (0.0 <= float(fv["confidence"]) <= 1.0):
            errors.append(f"{field_name} confidence out of [0,1] range")
        if fv.get("status") not in STATUS_VALUES:
            errors.append(f"{field_name} has invalid status: {fv.get('status')}")

    items = record.get("items")
    if not isinstance(items, list):
        errors.append("items must be a list")
    else:
        for i, item in enumerate(items):
            if not isinstance(item, dict) or "name" not in item or "price" not in item:
                errors.append(f"items[{i}] missing 'name' or 'price'")

    if not isinstance(record.get("warnings"), list):
        errors.append("warnings must be a list")

    return errors
