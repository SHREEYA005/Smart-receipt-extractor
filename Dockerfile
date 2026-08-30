FROM python:3.12-slim

# Tesseract is a system package, not a pip package - this is the only
# non-Python dependency the pipeline needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Place receipt images in ./data/receipts on the host and mount it, e.g.:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs carbon-crunch-receipt-ai
ENTRYPOINT ["python", "run_pipeline.py"]
