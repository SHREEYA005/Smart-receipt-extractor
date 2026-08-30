from src.utils.schema import FieldValue, ItemRecord, build_receipt_record, validate_receipt_record


def make_valid_record():
    return build_receipt_record(
        receipt_id="test_001",
        store_name=FieldValue("ACME STORE", 0.9, "high"),
        date=FieldValue("2019-06-15", 0.85, "high"),
        items=[ItemRecord("BREAD", "3.50", 0.8, "medium")],
        total_amount=FieldValue("3.50", 0.9, "high"),
        warnings=[],
        meta={"processing_time_sec": 0.5},
    )


def test_valid_record_has_no_errors():
    record = make_valid_record()
    assert validate_receipt_record(record) == []


def test_missing_top_level_key_detected():
    record = make_valid_record()
    del record["total_amount"]
    errors = validate_receipt_record(record)
    assert any("total_amount" in e for e in errors)


def test_confidence_out_of_range_detected():
    record = make_valid_record()
    record["store_name"]["confidence"] = 1.5
    errors = validate_receipt_record(record)
    assert any("confidence out of" in e for e in errors)


def test_invalid_status_detected():
    record = make_valid_record()
    record["date"]["status"] = "definitely_correct"
    errors = validate_receipt_record(record)
    assert any("invalid status" in e for e in errors)


def test_missing_field_values_are_schema_valid():
    # null value + 0 confidence + status="missing" is a first-class,
    # valid state (Phase 9 edge cases), not an error.
    record = build_receipt_record(
        receipt_id="empty_001",
        store_name=FieldValue(None, 0.0, "missing"),
        date=FieldValue(None, 0.0, "missing"),
        items=[],
        total_amount=FieldValue(None, 0.0, "missing"),
        warnings=["no OCR text found"],
        meta={},
    )
    assert validate_receipt_record(record) == []


def test_item_missing_price_key_detected():
    record = make_valid_record()
    record["items"] = [{"name": "BREAD"}]
    errors = validate_receipt_record(record)
    assert any("items[0]" in e for e in errors)


def test_field_value_to_dict_omits_alternatives_when_none():
    fv = FieldValue("X", 0.9, "high")
    d = fv.to_dict()
    assert "alternatives" not in d


def test_field_value_to_dict_includes_alternatives_when_present():
    fv = FieldValue(None, 0.5, "ambiguous", alternatives=[{"value": "A"}, {"value": "B"}])
    d = fv.to_dict()
    assert d["alternatives"] == [{"value": "A"}, {"value": "B"}]
