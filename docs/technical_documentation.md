# Receipt Vision - Technical Documentation

## 1. Approach

The system follows a linear pipeline: validate the image, preprocess it,
run OCR, reconstruct lines from OCR tokens, extract candidate values for
each required field, cross-check the candidates against each other, score
confidence from named components, and emit a structured JSON record. The
same design runs across the full batch to produce a financial summary and
an evaluation report.

The central design decision was to treat every extraction as a *scored
candidate*, never a fact. Store name, date, total, and each line item all
go through: find candidates → score them → keep the best one *with its
score* → let downstream consistency checks and confidence scoring decide
how much to trust it. This is what makes "flag low-confidence fields"
possible without extra plumbing - the confidence was computed as part of
extraction, not bolted on afterward.

## 2. Tools Used

- **OCR:** Tesseract 5.3.4 via `pytesseract`, chosen because it exposes
  per-word confidence directly and needs no model download (EasyOCR and
  PaddleOCR both require fetching weights from hosts unreachable in this
  project's execution environment).
- **Image processing:** OpenCV (grayscale, CLAHE, adaptive threshold,
  denoising, deskew via `minAreaRect`).
- **Language/runtime:** Python 3.12, NumPy, PyYAML for config.
- **Testing:** pytest (65 tests).
- No LLM is used anywhere in the core pipeline. It is deterministic and
  reproducible - the same image produces the same output every run.

## 3. Pipeline Architecture

```
image -> validate -> preprocess (multi-pass) -> OCR -> line reconstruction
  -> {store, date, total, item} candidate extraction -> consistency checks
  -> confidence scoring -> structured JSON -> aggregation / evaluation
```

Preprocessing is not applied unconditionally. Three named strategies
(`raw`, `clahe_deskew`, `full`) are tried cheapest-first; a pass is
accepted once its OCR mean confidence and token count clear a configured
bar, otherwise the pipeline escalates. Measuring this on a 30-image
sample showed heavier preprocessing does **not** universally help: `full`
found more OCR tokens (148.6 avg vs 129.9 for `raw`) but at *lower* mean
confidence (67.8 vs 73.2) - see `experiments/preprocessing_comparison.csv`.
This is the direct justification for trying `raw` first rather than
always running the heaviest pipeline.

## 4. Confidence Scoring

Each field's confidence is a weighted sum of named components (weights in
`configs/default.yaml`):

- **store_name:** OCR confidence, position on the receipt, alphabetic
  ratio, length, minus a penalty if the line reads like a marketing
  slogan rather than a name.
- **date:** OCR confidence, whether the format parses as a valid
  calendar date, proximity to a "date" keyword, minus an ambiguity
  penalty when a numeric date is genuinely readable two ways (e.g.
  `03/04/2019` - both 3 April and March 4 are valid dates, so both are
  reported as alternatives instead of guessing).
- **total_amount:** OCR confidence, strength of the matched keyword
  (`grand total` > `total` > unlabelled number), currency-format
  validity, and a consistency score from comparing the total against
  the sum of extracted item prices.
- **items:** OCR confidence, price-pattern validity, position.

Every component is stored per-record under `meta.confidence_components`
so a reviewer can see *why* a field scored the way it did, not just the
final number.

## 5. Challenges Faced

- **The dataset has no ground truth.** This shaped almost every other
  decision: the evaluation module reports coverage/consistency statistics
  instead of an accuracy percentage, and generates a manual-review CSV
  rather than fabricating a number.
- **Keyword collisions on `total`.** Some receipts print several lines
  containing the substring "total" (subtotal, tax total, grand total),
  and the keyword-tier ranking occasionally locks onto the wrong one. On
  receipt `X51005705727`, this produced a genuinely wrong total (2.16
  instead of 38.15) - but the system caught its own error: the item/total
  mismatch check flagged it, confidence dropped to 0.45, status became
  `ambiguous`, and the correct value surfaced as an alternative. This is
  the intended failure mode: wrong-but-flagged, not wrong-and-confident.
- **A real regex bug caught by testing.** The amount-parsing regex was
  not anchored against surrounding digits, so an implausible number like
  `9999999.99` could be matched as a substring (`999.99`). A test written
  specifically to check implausible-amount rejection caught this; the fix
  was a negative lookbehind/lookahead around the digit pattern. No receipt
  in the actual dataset happened to trigger this bug, but it would have
  on a malformed OCR read.
- **Multi-line items.** Some receipts print an item's description and its
  quantity/price on separate lines. The current line-based extractor
  cannot merge these, so such items are either missed or captured as two
  partial/lower-confidence fragments. This is the single biggest driver
  of the 53.8% item-coverage figure and is documented as a known
  limitation rather than worked around with brittle heuristics.

## 6. Edge Cases Handled

Missing/corrupt/empty/too-small image files are caught in
`ImageValidationError` and produce a valid schema record with `missing`
status and a warning - never an unhandled exception. A blank image (no
recognisable text at all) similarly degrades to `missing`/low-confidence
fields rather than crashing. All of this is covered by
`tests/test_edge_cases.py`, and confirmed across the full 370-image batch:
0 crashes, 0 schema validation failures.

## 7. Evaluation

Measured on the full 370-image dataset (see `outputs/reports/evaluation_report.json`
after running the pipeline):

| Metric | Value |
|---|---|
| Store name found | 98.9% |
| Date found | 59.5% |
| Total amount found | 90.8% |
| At least one item found | 53.8% |
| Mean OCR confidence | 74.1 |
| Item/total consistency pass rate | 40.1% |

No accuracy percentage is reported because no ground truth exists for
this dataset. `outputs/reports/manual_review_sample.csv` is a
reproducible random sample (seeded) with blank columns for a human to
fill in real correctness.

## 8. Improvements / Future Work

1. Multi-line item merging - the highest-ROI single change for item
   coverage.
2. A small hand-labelled sample (30-40 receipts) to replace reasoned
   confidence weights with weights fit against real correctness, and to
   enable an actual calibration analysis.
3. A store-name classifier (dictionary or small learned model) for the
   stylised-logo failure case.
4. Layout/x-position-based item-price association instead of
   same-OCR-line trailing-number matching, for multi-column receipts.
