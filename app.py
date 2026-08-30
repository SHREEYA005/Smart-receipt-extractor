"""Receipt Vision - interactive web demo.

Upload a receipt image, see the extracted fields, confidence scores, and
warnings live. This wraps the exact same pipeline used by run_pipeline.py -
no separate extraction logic, so what you see here matches the CLI output
exactly.

Run locally:
    streamlit run app.py

Deployed on Hugging Face Spaces via the Dockerfile in this repo.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline.run import process_receipt
from src.utils.config import load_config

st.set_page_config(page_title="Receipt Vision", page_icon="🧾", layout="centered")

STATUS_COLORS = {
    "high": "🟢",
    "medium": "🟡",
    "low": "🟠",
    "ambiguous": "🟣",
    "missing": "⚪",
}


def render_field(label: str, field: dict) -> None:
    icon = STATUS_COLORS.get(field["status"], "⚪")
    value = field["value"] if field["value"] is not None else "—"
    st.markdown(f"**{label}:** {value}  {icon} `{field['status']}` (confidence: {field['confidence']:.2f})")
    if field.get("alternatives"):
        alts = ", ".join(str(a.get("value", a)) for a in field["alternatives"])
        st.caption(f"Alternatives considered: {alts}")


st.title("🧾 Receipt Vision")
st.caption(
    "Confidence-aware receipt extraction. Upload a receipt image to see exactly what the "
    "pipeline extracts, how confident it is in each field, and why."
)

with st.expander("How to read this"):
    st.markdown(
        "- 🟢 **high** confidence (≥0.85) · 🟡 **medium** (≥0.70) · 🟠 **low** (<0.70, flagged) "
        "· 🟣 **ambiguous** (conflicting candidates found, both shown) · ⚪ **missing** (nothing found)\n"
        "- Nothing here is guessed to look good. A wrong or uncertain field is reported as such, "
        "with a confidence score and warning, rather than silently smoothed over."
    )

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="Uploaded receipt", use_container_width=True)

    with st.spinner("Running OCR and extraction..."):
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        config = load_config()
        record = process_receipt(tmp_path, config)
        Path(tmp_path).unlink(missing_ok=True)

    with col2:
        st.subheader("Extracted fields")
        render_field("Store name", record["store_name"])
        render_field("Date", record["date"])
        render_field("Total amount", record["total_amount"])

        st.markdown(f"**Items found:** {len(record['items'])}")
        for item in record["items"]:
            icon = STATUS_COLORS.get(item["status"], "⚪")
            st.markdown(f"- {item['name']} — {item['price']}  {icon} (confidence: {item['confidence']:.2f})")

        if record["warnings"]:
            st.subheader("Warnings")
            for w in record["warnings"]:
                st.warning(w)

        with st.expander("Confidence breakdown (why these scores?)"):
            st.json(record["meta"]["confidence_components"])

        with st.expander("Full JSON output"):
            st.json(record)

        st.download_button(
            "Download JSON",
            data=json.dumps(record, indent=2),
            file_name=f"{record['receipt_id']}.json",
            mime="application/json",
        )
else:
    st.info("Upload a receipt image above to see it processed live.")

st.divider()
st.caption(
    "This demo runs the same pipeline as the command-line tool in this repository — "
    "Tesseract OCR, layout-aware field extraction, and component-based confidence scoring. "
    "No ground-truth labels exist for the dataset this was built and tested on, so confidence "
    "scores reflect internal signal strength, not a guarantee of correctness."
)
