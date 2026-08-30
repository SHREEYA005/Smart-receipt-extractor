"""Cross-field consistency checks (Phase 5).

These are *signals*, not verdicts. A receipt where item prices don't sum
to the printed total is not necessarily wrong - tax, discounts, service
charges, tips and rounding are all legitimate reasons for a gap. So
`check_item_total_consistency` returns a graded signal (match / close /
mismatch / insufficient_data) with the actual delta, which the
confidence module folds in as one input among several rather than a
pass/fail gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# Anything within this fraction of the total is treated as consistent
# with taxes/rounding rather than a genuine extraction error.
RELATIVE_TOLERANCE = 0.15
ABSOLUTE_TOLERANCE = 2.00  # currency units, covers small fixed fees/rounding


@dataclass
class ConsistencyResult:
    signal: str  # "match" | "close" | "mismatch" | "insufficient_data"
    item_sum: Optional[float]
    total_value: Optional[float]
    delta: Optional[float]


def check_item_total_consistency(item_prices: List[float], total_value: Optional[float]) -> ConsistencyResult:
    if total_value is None or not item_prices:
        return ConsistencyResult("insufficient_data", None, total_value, None)

    item_sum = round(sum(item_prices), 2)
    delta = round(abs(item_sum - total_value), 2)

    tolerance = max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * total_value)

    if delta <= tolerance * 0.3:
        signal = "match"
    elif delta <= tolerance:
        signal = "close"
    else:
        signal = "mismatch"

    return ConsistencyResult(signal, item_sum, total_value, delta)


def is_plausible_amount(value: Optional[float]) -> bool:
    return value is not None and 0 < value <= 1_000_000


def detect_conflicting_totals(candidate_values: List[float], chosen_value: float) -> bool:
    """True if another candidate total materially disagrees with the chosen one."""
    for v in candidate_values:
        if v == chosen_value:
            continue
        if abs(v - chosen_value) > max(0.02, 0.02 * chosen_value):
            return True
    return False
