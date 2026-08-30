#!/usr/bin/env python3
"""Run the full receipt extraction pipeline over a directory of images.

Usage:
    python run_pipeline.py --data-dir data/receipts --output-dir outputs
    python run_pipeline.py --data-dir data/receipts --limit 20
    python run_pipeline.py --config configs/default.yaml

Every image in --data-dir (jpg/jpeg/png) is processed independently; a
failure on one image is logged and recorded as a low-confidence result,
it does not stop the batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.aggregation.summary import build_financial_summary
from src.evaluation.evaluate import build_evaluation_report, build_manual_review_sample, write_manual_review_csv
from src.pipeline.run import process_receipt
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.utils.schema import validate_receipt_record

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carbon Crunch receipt extraction pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory of receipt images")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write outputs")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images (useful for smoke tests)")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N images before applying --limit (for chunked runs)")
    parser.add_argument("--append", action="store_true", help="Do not overwrite existing per-receipt JSON files; still rebuilds summary/eval from everything in outputs/json")
    parser.add_argument("--review-sample-size", type=int, default=15, help="Number of receipts in the manual review CSV")
    return parser.parse_args()


def find_images(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    data_dir = Path(args.data_dir or config["paths"]["data_dir"])
    output_dir = Path(args.output_dir or config["paths"]["output_dir"])
    json_dir = output_dir / "json"
    summary_dir = output_dir / "summary"
    reports_dir = output_dir / "reports"
    for d in (json_dir, summary_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(config["logging"]["log_file"], config["logging"]["level"])

    if not data_dir.exists():
        logger.error("data directory does not exist: %s", data_dir)
        print(f"ERROR: data directory does not exist: {data_dir}")
        print("Set --data-dir, or place images under data/receipts/ (see data/README.md).")
        return 1

    images = find_images(data_dir)
    if args.start_index:
        images = images[args.start_index :]
    if args.limit:
        images = images[: args.limit]

    if not images:
        logger.warning("no images found in %s", data_dir)
        print(f"No images found in {data_dir}. Nothing to do.")
        return 0

    logger.info("starting pipeline run: %d images from %s", len(images), data_dir)

    records = []
    schema_errors_found = 0

    for i, image_path in enumerate(images, start=1):
        out_path = json_dir / f"{image_path.stem}.json"
        if args.append and out_path.exists():
            continue

        record = process_receipt(image_path, config)

        errors = validate_receipt_record(record)
        if errors:
            schema_errors_found += 1
            logger.error("receipt=%s schema validation failed: %s", record["receipt_id"], errors)

        out_path = json_dir / f"{record['receipt_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        records.append(record)
        if i % 25 == 0 or i == len(images):
            print(f"  processed {i}/{len(images)}")

    logger.info("finished OCR/extraction for %d receipts (%d schema errors)", len(records), schema_errors_found)

    # Rebuild summary/evaluation from *everything* currently in outputs/json,
    # not just this invocation's batch. This lets the pipeline be run in
    # chunks (--start-index/--limit) across multiple invocations and still
    # produce a correct whole-dataset summary at the end, without needing
    # to keep every record in memory across processes.
    all_records = []
    for p in sorted(json_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            all_records.append(json.load(f))

    summary = build_financial_summary(all_records, config["aggregation"]["inclusion_confidence_threshold"])
    with open(summary_dir / "financial_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    evaluation = build_evaluation_report(all_records)
    with open(reports_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    review_rows = build_manual_review_sample(all_records, args.review_sample_size)
    write_manual_review_csv(review_rows, reports_dir / "manual_review_sample.csv")

    print(f"\nProcessed {len(records)} receipts this run ({len(all_records)} total in {json_dir}).")
    print(f"  JSON outputs:        {json_dir}")
    print(f"  Financial summary:   {summary_dir / 'financial_summary.json'}")
    print(f"  Evaluation report:   {reports_dir / 'evaluation_report.json'}")
    print(f"  Manual review sample:{reports_dir / 'manual_review_sample.csv'}")
    if schema_errors_found:
        print(f"  WARNING: {schema_errors_found} records failed schema validation - see log.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
