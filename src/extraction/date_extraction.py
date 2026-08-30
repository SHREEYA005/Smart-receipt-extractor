"""Date extraction and validation (Phase 4 / Phase 7).

Two things this module refuses to do:
  1. Guess a fake date when nothing plausible is present.
  2. Silently resolve a genuinely ambiguous numeric date (e.g. 03/04/2019)
     to one interpretation without recording that the other was possible.

A numeric date like DD/MM vs MM/DD is only ambiguous when both readings
are valid calendar dates (i.e. both day and month are <= 12). Month-name
dates ("20 Jan 2019") and 4-digit-first ISO dates are unambiguous by
construction and are preferred over pure-numeric matches when both are
present in the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

MONTHS = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    "january|february|march|april|june|july|august|september|october|november|december"
)

# Ordered: unambiguous formats first, ambiguous numeric formats last.
DATE_PATTERNS = [
    # ISO: 2019-06-15
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    # 15 Jun 2019 / 15-Jun-2019
    (re.compile(rf"\b(\d{{1,2}})[\s\-]({MONTHS})[\s\-,]+(\d{{2,4}})\b", re.IGNORECASE), "d_mon_y"),
    # Jun 15, 2019
    (re.compile(rf"\b({MONTHS})[\s\-]+(\d{{1,2}}),?\s+(\d{{2,4}})\b", re.IGNORECASE), "mon_d_y"),
    # Numeric, ambiguous: 15/06/2019, 15-06-19, 06/15/2019
    (re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b"), "numeric"),
]

MONTH_LOOKUP = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}
MONTH_LOOKUP["sept"] = 9
FULL_MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
for i, name in enumerate(FULL_MONTHS):
    MONTH_LOOKUP[name] = i + 1


@dataclass
class DateCandidate:
    raw_text: str
    iso_value: Optional[str]  # None if ambiguous / invalid
    ambiguous: bool
    alternatives: List[str]
    near_keyword: bool


def _to_full_year(y: int) -> int:
    if y >= 100:
        return y
    # Receipts in this dataset range 2010-2021; a 2-digit year in that
    # neighbourhood is virtually always 20xx.
    return 2000 + y


def _valid_date(y: int, m: int, d: int) -> bool:
    try:
        datetime(y, m, d)
        return True
    except ValueError:
        return False


def _parse_match(pattern_type: str, match: re.Match) -> Optional[DateCandidate]:
    raw = match.group(0)

    if pattern_type == "iso":
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if _valid_date(y, m, d):
            return DateCandidate(raw, f"{y:04d}-{m:02d}-{d:02d}", False, [], False)
        return None

    if pattern_type == "d_mon_y":
        d, mon_name, y = int(match.group(1)), match.group(2).lower(), int(match.group(3))
        m = MONTH_LOOKUP.get(mon_name)
        if m is None:
            return None
        y = _to_full_year(y)
        if _valid_date(y, m, d):
            return DateCandidate(raw, f"{y:04d}-{m:02d}-{d:02d}", False, [], False)
        return None

    if pattern_type == "mon_d_y":
        mon_name, d, y = match.group(1).lower(), int(match.group(2)), int(match.group(3))
        m = MONTH_LOOKUP.get(mon_name)
        if m is None:
            return None
        y = _to_full_year(y)
        if _valid_date(y, m, d):
            return DateCandidate(raw, f"{y:04d}-{m:02d}-{d:02d}", False, [], False)
        return None

    if pattern_type == "numeric":
        a, b, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        y = _to_full_year(y)

        a_as_day_valid = _valid_date(y, b, a)   # DD/MM
        b_as_day_valid = _valid_date(y, a, b)   # MM/DD

        if a_as_day_valid and b_as_day_valid and a != b:
            # Genuinely ambiguous: both readings are valid calendar dates.
            alt_dd_mm = f"{y:04d}-{b:02d}-{a:02d}"
            alt_mm_dd = f"{y:04d}-{a:02d}-{b:02d}"
            return DateCandidate(raw, None, True, [alt_dd_mm, alt_mm_dd], False)
        if a_as_day_valid:
            return DateCandidate(raw, f"{y:04d}-{b:02d}-{a:02d}", False, [], False)
        if b_as_day_valid:
            return DateCandidate(raw, f"{y:04d}-{a:02d}-{b:02d}", False, [], False)
        return None

    return None


DATE_KEYWORD_RE = re.compile(r"date|tarikh|invoice date|trans(?:action)?\s*date", re.IGNORECASE)


def find_date_candidates(full_text: str) -> List[DateCandidate]:
    candidates: List[DateCandidate] = []
    for line in full_text.splitlines():
        for pattern, ptype in DATE_PATTERNS:
            for match in pattern.finditer(line):
                candidate = _parse_match(ptype, match)
                if candidate is None:
                    continue
                candidate.near_keyword = bool(DATE_KEYWORD_RE.search(line))
                candidates.append(candidate)
    return candidates
