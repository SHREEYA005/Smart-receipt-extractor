"""Line-item extraction (Phase 4).

This is the hardest and least reliable field the assignment asks for, and
it is treated that way: a line is only reported as an item when it has
a name-like prefix *and* a trailing currency-shaped amount, and it is
excluded outright if it matches a known non-item keyword (subtotal, tax,
cash tendered, change, discount, ...). When the evidence is weak the
correct behaviour is to return fewer items with lower confidence, not to
force a line into an item it probably isn't - per the assignment's
explicit instruction not to invent items when evidence is weak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.extraction.lines import Line
from src.extraction.total_amount import _parse_amount

NON_ITEM_KEYWORDS = [
    "subtotal", "sub total", "sub-total", "total", "cash", "change", "tender",
    "discount", "tax", "vat", "gst", "balance", "rounding", "card", "visa",
    "mastercard", "eftpos", "approval", "auth code", "member", "points",
    "qty description", "item description", "thank you", "items sold", "receipt no",
    "invoice no", "tel", "fax", "cashier", "till", "operator", "table", "guest",
]

TRAILING_PRICE_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[,.]\d{3})*[.,]\d{2})\s*[a-zA-Z]?\s*$")


@dataclass
class ItemCandidate:
    name: str
    price: float
    ocr_conf: float
    line_index: int


def _is_non_item_line(text_lower: str) -> bool:
    return any(kw in text_lower for kw in NON_ITEM_KEYWORDS)


def find_item_candidates(lines: List[Line], total_line_index: int | None) -> List[ItemCandidate]:
    """Look for item-shaped lines between the header block and the total.

    ``total_line_index`` (when known) bounds the search: item lines
    should not appear after the total keyword, which mostly rules out
    tax breakdowns and footer text that happen to contain numbers.
    """

    candidates: List[ItemCandidate] = []
    search_end = total_line_index if total_line_index is not None else len(lines)

    for idx, line in enumerate(lines[:search_end]):
        text = line.text.strip()
        text_lower = text.lower()

        if _is_non_item_line(text_lower):
            continue

        match = TRAILING_PRICE_RE.search(text)
        if not match:
            continue

        price = _parse_amount(text)
        if price is None:
            continue

        name = text[: match.start()].strip(" -:*")
        # Strip a leading quantity token like "2 x" or "1" if present,
        # keeping it out of the item name.
        name = re.sub(r"^\d+\s*[xX]?\s*", "", name).strip()

        if len(name) < 2:
            continue

        candidates.append(
            ItemCandidate(name=name, price=price, ocr_conf=line.mean_conf, line_index=idx)
        )

    return candidates
