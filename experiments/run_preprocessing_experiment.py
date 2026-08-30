"""Preprocessing strategy comparison (Phase 3 / Phase 14).

Runs every named preprocessing strategy against a sample of the dataset
and records OCR mean confidence and token count for each. This is the
experiment referenced in docs/technical_documentation.md - it is what
justified making "raw" the first strategy tried in the multi-pass policy
(src/pipeline/ocr_pass.py) rather than always preprocessing.

No ground truth exists for this dataset, so this script measures OCR
confidence and token yield, not extraction accuracy. That distinction is
stated explicitly in the output.

Usage:
    python experiments/run_preprocessing_experiment.py --sample-size 40
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.engine import get_engine
from src.preprocessing.image_io import ImageValidationError, load_and_validate_image
from src.preprocessing.pipeline import STRATEGIES, apply_strategy

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare preprocessing strategies")
    parser.add_argument("--data-dir", type=str, default="data/receipts")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments/preprocessing_comparison.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images found in {data_dir}")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(images, min(args.sample_size, len(images)))

    engine = get_engine("tesseract", lang="eng", psm=6, oem=3)
    rows = []

    for image_path in sample:
        try:
            image = load_and_validate_image(image_path)
        except ImageValidationError as e:
            print(f"skipping {image_path.name}: {e}")
            continue

        for strategy_name in STRATEGIES:
            t0 = time.time()
            processed, _meta = apply_strategy(image, strategy_name)
            result = engine.run(processed)
            elapsed = time.time() - t0

            rows.append(
                {
                    "receipt_id": image_path.stem,
                    "strategy": strategy_name,
                    "ocr_mean_confidence": round(result.mean_confidence, 2),
                    "token_count": result.token_count,
                    "runtime_sec": round(elapsed, 3),
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["receipt_id", "strategy", "ocr_mean_confidence", "token_count", "runtime_sec"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

    # Per-strategy aggregate, printed directly (not just left in the CSV)
    # so the comparison is visible without opening another file.
    by_strategy: dict = {}
    for r in rows:
        by_strategy.setdefault(r["strategy"], []).append(r)

    print("\nPer-strategy averages (measured on this sample, not the full dataset):")
    print(f"{'strategy':<15}{'mean_ocr_conf':<16}{'mean_tokens':<14}{'mean_runtime_s':<15}")
    for strategy_name, strategy_rows in by_strategy.items():
        n = len(strategy_rows)
        mean_conf = sum(r["ocr_mean_confidence"] for r in strategy_rows) / n
        mean_tokens = sum(r["token_count"] for r in strategy_rows) / n
        mean_runtime = sum(r["runtime_sec"] for r in strategy_rows) / n
        print(f"{strategy_name:<15}{mean_conf:<16.2f}{mean_tokens:<14.1f}{mean_runtime:<15.3f}")


if __name__ == "__main__":
    main()
