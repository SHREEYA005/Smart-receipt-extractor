"""Total-amount extraction (Phase 4: "treated as a high-value field").

Strategy: scan every line for a total-like keyword, rank keywords by how
specific/high-value they are (an explicit "grand total" is stronger
evidence than a bare "total", which itself is stronger than a number
with no keyword at all), pull the nearest valid currency-shaped number
(same line, else the next line), and keep every candidate found so the
confidence/consistency stage can compare them instead of committing to
the first match blindly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.extraction.lines import Line

# Ordered strongest -> weakest. Checked as whole-word/phrase matches.
KEYWORD_TIERS = [
    (1.0, ["grand total", "amount due", "balance due", "total due", "net total"]),
    (0.8, ["total sale", "total amount", "total payable"]),
    (0.55, ["total"]),  # bare "total" is common but also collides with "subtotal"
]

EXCLUDE_KEYWORDS = ["subtotal", "sub total", "sub-total", "total qty", "total quantity", "total items"]

AMOUNT_RE = re.compile(
    r"(?:rm|myr|\$|usd|eur|r)?\s*(?<!\d)(\d{1,3}(?:[,.]\d{3})*[.,]\d{2})(?!\d)", re.IGNORECASE
)


@dataclass
class TotalCandidate:
    raw_line: str
    value: Optional[float]
    keyword_strength: float  # 0-1
    keyword: str
    ocr_conf: float
    line_index: int
    from_next_line: bool


def _parse_amount(text: str) -> Optional[float]:
    match = AMOUNT_RE.search(text)
    if not match:
        return None
    raw = match.group(1)
    # Normalise "1,234.56" and "1.234,56" style separators to a plain float.
    if raw.count(",") and raw.count("."):
        raw = raw.replace(",", "") if raw.rfind(".") > raw.rfind(",") else raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", ".") if raw.count(",") == 1 and len(raw.split(",")[-1]) == 2 else raw.replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    if 0 < value <= 1_000_000:
        return value
    return None


def _keyword_strength(text_lower: str) -> tuple[float, str]:
    if any(ex in text_lower for ex in EXCLUDE_KEYWORDS):
        return 0.0, ""
    for strength, keywords in KEYWORD_TIERS:
        for kw in keywords:
            if kw in text_lower:
                return strength, kw
    return 0.0, ""


def find_total_candidates(lines: List[Line]) -> List[TotalCandidate]:
    candidates: List[TotalCandidate] = []

    for idx, line in enumerate(lines):
        text_lower = line.text.lower()
        strength, keyword = _keyword_strength(text_lower)
        if strength == 0.0:
            continue

        value = _parse_amount(line.text)
        from_next_line = False
        source_line = line

        if value is None and idx + 1 < len(lines):
            # Some receipts print "TOTAL" and the amount on the next line.
            next_line = lines[idx + 1]
            value = _parse_amount(next_line.text)
            if value is not None:
                from_next_line = True
                source_line = next_line

        if value is None:
            continue

        candidates.append(
            TotalCandidate(
                raw_line=line.text,
                value=value,
                keyword_strength=strength,
                keyword=keyword,
                ocr_conf=source_line.mean_conf,
                line_index=idx,
                from_next_line=from_next_line,
            )
        )

    return candidates


def select_total(candidates: List[TotalCandidate]) -> Optional[TotalCandidate]:
    if not candidates:
        return None

    def score(c: TotalCandidate) -> float:
        return 0.7 * c.keyword_strength + 0.3 * (c.ocr_conf / 100.0)

    return max(candidates, key=score)
