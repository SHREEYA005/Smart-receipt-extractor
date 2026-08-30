# Dataset

This repository does not ship receipt images (they are not this project's
IP to redistribute, and keeping the repo image-free keeps it small and
avoids committing anything that might be considered personal data).

## Setup

1. Download the dataset from the link provided in the assignment.
2. Place all receipt images directly under `data/receipts/` (create the
   folder if it does not exist):

   ```
   data/receipts/0.jpg
   data/receipts/1.jpg
   data/receipts/X51005268200.jpg
   ...
   ```

3. Run `python run_pipeline.py`. The data directory is also configurable
   via `--data-dir` or `configs/default.yaml`, so this layout is not
   hardcoded anywhere in the code.

## What the provided dataset actually looks like

(Recorded here after inspecting the real dataset used during development -
this is a factual description, not an assumption.)

- **371 images**, all `.jpg` except one `.png`.
- **Two distinct sub-populations**, distinguishable by filename:
  - 349 files named like `X51005268200.jpg` - clean, top-down scanned
    receipts (this naming convention matches the public SROIE / ICDAR2019
    receipt dataset). Mostly Malaysian retail receipts (MYR amounts,
    GST/SST breakdowns), consistent lighting, minimal skew, dot-matrix
    and thermal-printer fonts.
  - 22 files named numerically (`0.jpg`-`21.jpg`) - real phone photos of
    receipts held in a hand, resting on a lap/car seat, etc. US and South
    African retailers (Walmart, Dollar Tree, Spar). These contribute the
    "real-world noise" the assignment describes: glare, uneven lighting,
    background clutter, slight rotation, fingers in frame, and at least
    one receipt with a handwritten circle/annotation on top of the print.
- Image dimensions vary widely: width 433-2481px, height 543-4032px.
- **No ground-truth labels/annotations ship with the dataset** - no
  `.txt`/`.json`/`.csv` files were found alongside the images. This is why
  `src/evaluation/evaluate.py` reports coverage/consistency statistics
  instead of an accuracy percentage, and why it generates a manual review
  CSV rather than fabricating one.
