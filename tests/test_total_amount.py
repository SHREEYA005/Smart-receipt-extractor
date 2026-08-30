from src.extraction.lines import Line
from src.extraction.total_amount import find_total_candidates, select_total


def make_line(text: str, mean_conf: float = 90.0) -> Line:
    return Line(text=text, tokens=[], top=0, mean_conf=mean_conf)


def test_grand_total_beats_generic_total():
    lines = [
        make_line("SUBTOTAL 45.00"),
        make_line("TOTAL 48.00"),
        make_line("GRAND TOTAL 50.00"),
    ]
    candidates = find_total_candidates(lines)
    best = select_total(candidates)
    assert best.value == 50.00
    assert best.keyword == "grand total"


def test_subtotal_excluded_when_total_present():
    lines = [make_line("SUBTOTAL 45.00"), make_line("TOTAL 48.00")]
    candidates = find_total_candidates(lines)
    assert all(c.keyword != "subtotal" for c in candidates)
    best = select_total(candidates)
    assert best.value == 48.00


def test_amount_on_next_line_is_found():
    lines = [make_line("TOTAL"), make_line("48.00")]
    candidates = find_total_candidates(lines)
    assert candidates
    assert candidates[0].value == 48.00
    assert candidates[0].from_next_line is True


def test_no_keyword_yields_no_candidates():
    lines = [make_line("BANANAS 2.50"), make_line("APPLES 1.20")]
    candidates = find_total_candidates(lines)
    assert candidates == []


def test_currency_symbol_and_comma_thousands_parsed():
    lines = [make_line("AMOUNT DUE $1,234.56")]
    candidates = find_total_candidates(lines)
    assert candidates[0].value == 1234.56


def test_implausible_amount_rejected():
    # A "total" of 9,999,999.99 is not plausible for a retail receipt
    # and should not be returned as a valid amount.
    lines = [make_line("TOTAL 9999999.99")]
    candidates = find_total_candidates(lines)
    assert candidates == []


def test_select_total_returns_none_for_empty_candidates():
    assert select_total([]) is None
