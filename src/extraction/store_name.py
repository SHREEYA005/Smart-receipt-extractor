"""Store / vendor name extraction (Phase 4).

Heuristic, not fragile-regex: the store name is not identified by a
pattern, it's identified by *where* it sits on the receipt and what it
looks like relative to its neighbours. We score the first few lines of
the receipt and pick the best candidate rather than assuming line 0 is
always right (logos, "TAX INVOICE" headers, and stray OCR noise all
commonly occupy line 0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.extraction.lines import Line

HEADER_KEYWORDS = {
    "invoice", "tax invoice", "simplified tax invoice", "receipt", "receipt no",
    "official receipt", "cash bill", "cash invoice", "cash receipt", "bill", "order",
    "tel", "fax", "original receipt", "customer copy",
}

TOP_N_LINES = 8


@dataclass
class StoreCandidate:
    text: str
    line_index: int
    ocr_conf: float  # 0-100
    alpha_ratio: float
    length_score: float
    tagline_penalty: float  # 0-1, higher = looks more like a slogan than a name


def _looks_like_header_keyword(text: str) -> bool:
    lowered = re.sub(r"^[^a-z0-9]+", "", text.lower().strip())
    return any(lowered == kw or lowered.startswith(kw + " ") for kw in HEADER_KEYWORDS)


def _looks_like_ocr_noise(text: str) -> bool:
    """Reject lines that are mostly isolated 1-2 character fragments.

    Genuine store names read cleanly as a few real words; heavily
    garbled OCR on background clutter or receipt edges tends to come
    out as strings of short disconnected tokens (e.g. '. mn 5 . j ot',
    '2 ~ cin. / A noes >'). This is a coarse but effective filter for
    that failure mode.
    """
    tokens = text.split()
    if not tokens:
        return True
    short_tokens = sum(1 for t in tokens if len(t) <= 2)
    return (short_tokens / len(tokens)) > 0.5 and len(tokens) >= 3


def _looks_like_contact_line(text: str) -> bool:
    """Phone numbers, fax numbers, registration numbers, addresses with lots of digits."""
    digit_count = sum(c.isdigit() for c in text)
    return digit_count >= 5 or bool(re.search(r"\btel\b|\bfax\b|\bno\.?\s*\d", text.lower()))


def _looks_like_address_line(text: str) -> bool:
    """Street-address fragments that otherwise out-score the real store
    name on pure alpha-ratio/position (e.g. 'LOT 2811, JALAN ANGSA,',
    '3RD FLR, AEON TAMAN MALURI SC', 'SUITE C-3', 'NO 17-G, JALAN ...').
    """
    lowered = text.lower()
    address_markers = (
        "jalan", "lot ", "taman", " flr", "floor", "suite", "blk", "tingkat",
        "persiaran", "lorong", "kawasan", "seksyen", "sek.", "no.", "no:",
    )
    if any(marker in lowered for marker in address_markers):
        return True
    # Address fragments are also very often comma-separated with digits
    # ("2811, Jalan Angsa,") - a real store name rarely contains a comma.
    if "," in text and any(c.isdigit() for c in text):
        return True
    return False


def _looks_like_promotional_line(text: str) -> bool:
    lowered = text.lower()
    promo_markers = (
        "see back of receipt", "give us feedback", "your order number",
        "scan with", "download the app", "survey.", "win a", "chance to",
    )
    return any(marker in lowered for marker in promo_markers)


def _tagline_penalty(text: str) -> float:
    """Store names are almost never a full sentence.

    Marketing taglines ("Always Low Prices.", "Save money. Live
    better.") tend to (a) end in sentence punctuation and (b) contain
    several common lowercase function words. Both are cheap, reasonably
    reliable signals that a line is a slogan rather than the name
    itself, without needing a hand-maintained list of every retailer's
    tagline.
    """

    stripped = text.strip()
    penalty = 0.0
    if stripped.endswith((".", "!")):
        penalty += 0.6

    lowered = stripped.lower()
    function_words = {"low", "prices", "save", "money", "better", "every", "day", "always", "where", "everything"}
    word_hits = sum(1 for w in re.findall(r"[a-z]+", lowered) if w in function_words)
    if word_hits >= 2:
        penalty += 0.4

    return min(1.0, penalty)


def find_store_candidates(lines: List[Line]) -> List[StoreCandidate]:
    candidates: List[StoreCandidate] = []
    top_lines = lines[:TOP_N_LINES]

    for idx, line in enumerate(top_lines):
        text = line.text.strip()
        if len(text) < 2:
            continue
        if _looks_like_header_keyword(text):
            continue
        if _looks_like_contact_line(text):
            continue
        if _looks_like_address_line(text):
            continue
        if _looks_like_promotional_line(text):
            continue
        if _looks_like_ocr_noise(text):
            continue

        candidates.append(
            StoreCandidate(
                text=text,
                line_index=idx,
                ocr_conf=line.mean_conf,
                alpha_ratio=line.alpha_ratio,
                length_score=min(1.0, len(text) / 25.0),
                tagline_penalty=_tagline_penalty(text),
            )
        )

    return candidates


def select_store_name(lines: List[Line]) -> Optional[StoreCandidate]:
    candidates = find_store_candidates(lines)
    if not candidates:
        return None

    def score(c: StoreCandidate) -> float:
        position_score = 1.0 - (c.line_index / TOP_N_LINES)
        return (
            0.35 * (c.ocr_conf / 100.0)
            + 0.30 * position_score
            + 0.20 * c.alpha_ratio
            + 0.15 * c.length_score
            - 0.5 * c.tagline_penalty
        )

    return max(candidates, key=score)
