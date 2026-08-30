from src.aggregation.summary import build_financial_summary


def make_record(receipt_id, store, total_value, total_conf, total_status="high"):
    return {
        "receipt_id": receipt_id,
        "store_name": {"value": store, "confidence": 0.9, "status": "high"},
        "date": {"value": "2019-01-01", "confidence": 0.9, "status": "high"},
        "items": [],
        "total_amount": {"value": total_value, "confidence": total_conf, "status": total_status},
        "warnings": [],
        "meta": {},
    }


def test_included_spend_excludes_low_confidence():
    records = [
        make_record("r1", "STORE A", 10.0, 0.9),
        make_record("r2", "STORE A", 20.0, 0.3),  # below default inclusion threshold
    ]
    summary = build_financial_summary(records, inclusion_threshold=0.5)
    assert summary["reported_total_spend"] == 30.0
    assert summary["included_total_spend"] == 10.0
    assert summary["number_of_transactions_included"] == 1
    assert summary["number_of_totals_excluded_low_confidence"] == 1


def test_ambiguous_totals_excluded_even_if_confidence_high():
    records = [make_record("r1", "STORE A", 10.0, 0.95, total_status="ambiguous")]
    summary = build_financial_summary(records, inclusion_threshold=0.5)
    assert summary["included_total_spend"] == 0.0
    assert summary["number_of_totals_excluded_low_confidence"] == 1


def test_missing_total_not_counted_in_either_bucket():
    records = [make_record("r1", "STORE A", None, 0.0, total_status="missing")]
    summary = build_financial_summary(records, inclusion_threshold=0.5)
    assert summary["reported_total_spend"] == 0.0
    assert summary["number_of_transactions_included"] == 0
    assert summary["number_of_totals_excluded_low_confidence"] == 0


def test_spend_per_store_groups_correctly():
    records = [
        make_record("r1", "STORE A", 10.0, 0.9),
        make_record("r2", "STORE A", 15.0, 0.9),
        make_record("r3", "STORE B", 5.0, 0.9),
    ]
    summary = build_financial_summary(records, inclusion_threshold=0.5)
    assert summary["spend_per_store_included"]["STORE A"] == 25.0
    assert summary["spend_per_store_included"]["STORE B"] == 5.0
    assert summary["transactions_per_store_included"]["STORE A"] == 2


def test_unknown_store_bucketed_separately():
    records = [make_record("r1", None, 10.0, 0.9)]
    summary = build_financial_summary(records, inclusion_threshold=0.5)
    assert "UNKNOWN" in summary["spend_per_store_included"]


def test_empty_records_produce_zeroed_summary():
    summary = build_financial_summary([], inclusion_threshold=0.5)
    assert summary["total_receipts_processed"] == 0
    assert summary["reported_total_spend"] == 0.0
