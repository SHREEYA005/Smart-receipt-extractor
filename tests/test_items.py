from src.extraction.items import find_item_candidates
from src.extraction.lines import Line


def make_line(text: str, mean_conf: float = 90.0) -> Line:
    return Line(text=text, tokens=[], top=0, mean_conf=mean_conf)


def test_simple_items_extracted():
    lines = [
        make_line("STORE NAME"),
        make_line("BREAD 3.50"),
        make_line("MILK 2.20"),
        make_line("TOTAL 5.70"),
    ]
    items = find_item_candidates(lines, total_line_index=3)
    names = {i.name for i in items}
    assert "BREAD" in names
    assert "MILK" in names
    assert len(items) == 2


def test_non_item_keywords_excluded():
    lines = [
        make_line("BREAD 3.50"),
        make_line("SUBTOTAL 3.50"),
        make_line("CASH TENDER 5.00"),
        make_line("CHANGE 1.50"),
        make_line("TOTAL 3.50"),
    ]
    items = find_item_candidates(lines, total_line_index=4)
    names = {i.name for i in items}
    assert names == {"BREAD"}


def test_lines_without_trailing_price_not_forced_into_items():
    lines = [
        make_line("BREAD 3.50"),
        make_line("Thank you for shopping"),
        make_line("TOTAL 3.50"),
    ]
    items = find_item_candidates(lines, total_line_index=2)
    assert len(items) == 1
    assert items[0].name == "BREAD"


def test_leading_quantity_stripped_from_name():
    lines = [make_line("2 x COFFEE 6.00"), make_line("TOTAL 6.00")]
    items = find_item_candidates(lines, total_line_index=1)
    assert items[0].name == "COFFEE"


def test_lines_after_total_are_not_scanned_for_items():
    lines = [
        make_line("BREAD 3.50"),
        make_line("TOTAL 3.50"),
        make_line("REF NUMBER 12.00"),  # a footer number that looks item-shaped
    ]
    items = find_item_candidates(lines, total_line_index=1)
    names = {i.name for i in items}
    assert "REF NUMBER" not in names


def test_no_candidates_when_nothing_matches():
    lines = [make_line("Welcome to our store"), make_line("Thank you")]
    assert find_item_candidates(lines, total_line_index=None) == []
