from src.confidence.scoring import score_date, score_store_name, score_total
from src.extraction.date_extraction import DateCandidate
from src.extraction.store_name import StoreCandidate
from src.extraction.total_amount import TotalCandidate
from src.utils.config import DEFAULT_CONFIG
from src.validation.validators import ConsistencyResult

STORE_WEIGHTS = DEFAULT_CONFIG["confidence"]["weights"]["store_name"]
DATE_WEIGHTS = DEFAULT_CONFIG["confidence"]["weights"]["date"]
TOTAL_WEIGHTS = DEFAULT_CONFIG["confidence"]["weights"]["total_amount"]
THRESHOLDS = DEFAULT_CONFIG["confidence"]["thresholds"]


def test_missing_store_name_has_zero_confidence_and_missing_status():
    field, components = score_store_name(None, STORE_WEIGHTS, THRESHOLDS)
    assert field.value is None
    assert field.confidence == 0.0
    assert field.status == "missing"
    assert components == {}


def test_strong_store_candidate_reaches_high_status():
    candidate = StoreCandidate(
        text="WALMART", line_index=0, ocr_conf=95.0, alpha_ratio=1.0, length_score=0.6, tagline_penalty=0.0
    )
    field, _ = score_store_name(candidate, STORE_WEIGHTS, THRESHOLDS)
    assert field.status == "high"
    assert field.value == "WALMART"


def test_tagline_penalty_lowers_confidence():
    slogan = StoreCandidate(
        text="Always Low Prices.", line_index=1, ocr_conf=95.0, alpha_ratio=0.95, length_score=0.7, tagline_penalty=1.0
    )
    field_with_penalty, _ = score_store_name(slogan, STORE_WEIGHTS, THRESHOLDS)

    not_slogan = StoreCandidate(
        text="Always Low Prices.", line_index=1, ocr_conf=95.0, alpha_ratio=0.95, length_score=0.7, tagline_penalty=0.0
    )
    field_without_penalty, _ = score_store_name(not_slogan, STORE_WEIGHTS, THRESHOLDS)

    assert field_with_penalty.confidence < field_without_penalty.confidence


def test_confidence_never_exceeds_one_or_drops_below_zero():
    extreme = StoreCandidate(
        text="X", line_index=0, ocr_conf=100.0, alpha_ratio=1.0, length_score=1.0, tagline_penalty=1.0
    )
    field, _ = score_store_name(extreme, STORE_WEIGHTS, THRESHOLDS)
    assert 0.0 <= field.confidence <= 1.0


def test_ambiguous_date_reports_alternatives_and_null_value():
    candidate = DateCandidate(
        raw_text="03/04/2019", iso_value=None, ambiguous=True,
        alternatives=["2019-04-03", "2019-03-04"], near_keyword=False,
    )
    field, _ = score_date([candidate], DATE_WEIGHTS, THRESHOLDS)
    assert field.value is None
    assert field.status == "ambiguous"
    assert field.alternatives is not None
    assert len(field.alternatives) == 2


def test_unambiguous_keyword_matched_date_reaches_high_status():
    candidate = DateCandidate(
        raw_text="15 Jun 2019", iso_value="2019-06-15", ambiguous=False, alternatives=[], near_keyword=True,
    )
    field, _ = score_date([candidate], DATE_WEIGHTS, THRESHOLDS)
    assert field.status == "high"
    assert field.value == "2019-06-15"


def test_no_date_candidates_gives_missing_status():
    field, _ = score_date([], DATE_WEIGHTS, THRESHOLDS)
    assert field.status == "missing"
    assert field.value is None


def test_total_confidence_boosted_by_consistency_match():
    candidate = TotalCandidate(
        raw_line="GRAND TOTAL 50.00", value=50.0, keyword_strength=1.0, keyword="grand total",
        ocr_conf=90.0, line_index=0, from_next_line=False,
    )
    matching = ConsistencyResult(signal="match", item_sum=50.0, total_value=50.0, delta=0.0)
    mismatching = ConsistencyResult(signal="mismatch", item_sum=10.0, total_value=50.0, delta=40.0)

    field_match, _ = score_total([candidate], TOTAL_WEIGHTS, THRESHOLDS, matching)
    field_mismatch, _ = score_total([candidate], TOTAL_WEIGHTS, THRESHOLDS, mismatching)

    assert field_match.confidence > field_mismatch.confidence


def test_missing_total_candidates_gives_missing_status():
    field, _ = score_total([], TOTAL_WEIGHTS, THRESHOLDS, ConsistencyResult("insufficient_data", None, None, None))
    assert field.status == "missing"
    assert field.value is None


def test_low_confidence_flag_threshold_from_config():
    # Sanity check that the assignment's < 0.70 flagging boundary is
    # actually what MEDIUM_THRESHOLD is set to in the default config.
    assert DEFAULT_CONFIG["confidence"]["low_confidence_flag"] == 0.70
    assert THRESHOLDS["medium"] == 0.70
