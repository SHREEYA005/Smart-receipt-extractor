"""OCR engine abstraction (Phase 2).

Why Tesseract:

The dataset (see docs/technical_documentation.md and data/README.md) mixes
clean scanned receipts with real phone photos. None of it requires a
learned detector to find text regions - the images are already single
tight receipt crops - so the main job is *recognition* plus per-word
confidence, which Tesseract exposes directly via ``image_to_data``.
EasyOCR and PaddleOCR were considered (both listed in the assignment) but
both need to download multi-hundred-MB model weights from hosts that are
not reachable from this project's execution environment, which would
break reproducibility for anyone re-running the pipeline in the same
kind of sandboxed/offline setting. Tesseract ships as a plain system
package with no model download step, which made it the pragmatic choice
here. The engine is hidden behind ``OCREngine`` specifically so a future
maintainer with network/GPU access can drop in EasyOCR or PaddleOCR
without touching anything downstream of OCR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pytesseract
from pytesseract import Output


@dataclass
class OCRToken:
    text: str
    conf: float  # 0-100, as reported by the engine
    left: int
    top: int
    width: int
    height: int
    line_num: int
    block_num: int
    par_num: int


@dataclass
class OCRResult:
    full_text: str
    tokens: List[OCRToken]
    mean_confidence: float
    engine_name: str
    image_shape: tuple
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.tokens)


class OCREngine(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, image: np.ndarray) -> OCRResult:
        ...


def _group_tokens_into_lines(tokens: List[OCRToken]) -> str:
    """Rebuild readable multi-line text from token boxes.

    Tesseract's own newline-joined output (``image_to_string``) and its
    per-word table (``image_to_data``) can disagree slightly on
    whitespace; we rebuild text directly from the token table so the
    text we display/search is guaranteed consistent with the token
    confidences and bounding boxes we score against.
    """

    if not tokens:
        return ""

    lines: Dict[tuple, List[OCRToken]] = {}
    for t in tokens:
        key = (t.block_num, t.par_num, t.line_num)
        lines.setdefault(key, []).append(t)

    ordered_keys = sorted(lines.keys())
    out_lines = []
    for key in ordered_keys:
        line_tokens = sorted(lines[key], key=lambda t: t.left)
        out_lines.append(" ".join(t.text for t in line_tokens if t.text.strip()))
    return "\n".join(line for line in out_lines if line.strip())


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 6, oem: int = 3):
        self.lang = lang
        self.psm = psm
        self.oem = oem

    def _config_str(self) -> str:
        return f"--oem {self.oem} --psm {self.psm}"

    def run(self, image: np.ndarray) -> OCRResult:
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=self._config_str(),
            output_type=Output.DICT,
        )

        tokens: List[OCRToken] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i]
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = -1.0

            if not text.strip() or conf < 0:
                # conf == -1 marks structural (non-text) boxes in
                # Tesseract's output; skip those, they are not tokens.
                continue

            tokens.append(
                OCRToken(
                    text=text,
                    conf=conf,
                    left=int(data["left"][i]),
                    top=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                    line_num=int(data["line_num"][i]),
                    block_num=int(data["block_num"][i]),
                    par_num=int(data["par_num"][i]),
                )
            )

        mean_conf = float(np.mean([t.conf for t in tokens])) if tokens else 0.0
        full_text = _group_tokens_into_lines(tokens)

        return OCRResult(
            full_text=full_text,
            tokens=tokens,
            mean_confidence=mean_conf,
            engine_name=self.name,
            image_shape=tuple(image.shape),
            metadata={"psm": self.psm, "oem": self.oem, "lang": self.lang},
        )


def get_engine(name: str = "tesseract", **kwargs) -> OCREngine:
    if name == "tesseract":
        return TesseractEngine(**kwargs)
    raise ValueError(f"Unknown OCR engine: {name}")
