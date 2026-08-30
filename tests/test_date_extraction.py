from src.extraction.date_extraction import find_date_candidates


def test_iso_date_extracted():
    candidates = find_date_candidates("Some text\n2019-06-15\nmore text")
    assert any(c.iso_value == "2019-06-15" for c in candidates)


def test_month_name_date_extracted_and_unambiguous():
    candidates = find_date_candidates("Receipt Date: 15 Jun 2019")
    matches = [c for c in candidates if c.iso_value == "2019-06-15"]
    assert matches
    assert matches[0].ambiguous is False


def test_mon_d_y_format_extracted():
    candidates = find_date_candidates("Jun 15, 2019")
    assert any(c.iso_value == "2019-06-15" for c in candidates)


def test_genuinely_ambiguous_numeric_date_flagged():
    # 03/04/2019: both "3 April" (DD/MM) and "March 4" (MM/DD) are valid
    # calendar dates, so this must be reported as ambiguous with both
    # alternatives, not silently resolved to one.
    candidates = find_date_candidates("03/04/2019")
    assert len(candidates) == 1
    c = candidates[0]
    assert c.ambiguous is True
    assert c.iso_value is None
    assert "2019-04-03" in c.alternatives  # DD/MM reading
    assert "2019-03-04" in c.alternatives  # MM/DD reading


def test_unambiguous_numeric_date_not_flagged():
    # Day=25 can only be a day, never a month, so this is unambiguous
    # even though it's a plain numeric format.
    candidates = find_date_candidates("25/12/2019")
    assert len(candidates) == 1
    c = candidates[0]
    assert c.ambiguous is False
    assert c.iso_value == "2019-12-25"


def test_invalid_date_is_rejected_not_hallucinated():
    # 32/13/2019 is not a valid date under either interpretation.
    candidates = find_date_candidates("32/13/2019")
    assert candidates == []


def test_two_digit_year_normalized():
    candidates = find_date_candidates("20-01-19")
    assert any(c.iso_value == "2019-01-20" for c in candidates)


def test_date_keyword_proximity_detected():
    candidates = find_date_candidates("Invoice Date: 2019-06-15")
    assert candidates[0].near_keyword is True

    candidates_no_kw = find_date_candidates("Random line 2019-06-15 with no label")
    assert candidates_no_kw[0].near_keyword is False


def test_no_date_in_text_returns_empty():
    assert find_date_candidates("No dates here at all, just words.") == []
