"""Turn a flat OCR token list back into ordered, positioned lines.

Field extraction depends heavily on layout (Phase 2/4): "is this the
first line on the receipt", "is there a number at the end of this
line" and so on all require tokens grouped into lines and sorted top to
bottom, left to right - not just the flat token list Tesseract returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.ocr.engine import OCRToken


@dataclass
class Line:
    text: str
    tokens: List[OCRToken]
    top: int
    mean_conf: float

    @property
    def alpha_ratio(self) -> float:
        letters = sum(c.isalpha() for c in self.text)
        return letters / max(1, len(self.text.replace(" ", "")))

    @property
    def digit_ratio(self) -> float:
        digits = sum(c.isdigit() for c in self.text)
        return digits / max(1, len(self.text.replace(" ", "")))


def build_lines(tokens: List[OCRToken]) -> List[Line]:
    groups: dict = {}
    for t in tokens:
        key = (t.block_num, t.par_num, t.line_num)
        groups.setdefault(key, []).append(t)

    lines: List[Line] = []
    for key in sorted(groups.keys()):
        line_tokens = sorted(groups[key], key=lambda t: t.left)
        text = " ".join(t.text for t in line_tokens).strip()
        if not text:
            continue
        top = min(t.top for t in line_tokens)
        mean_conf = sum(t.conf for t in line_tokens) / len(line_tokens)
        lines.append(Line(text=text, tokens=line_tokens, top=top, mean_conf=mean_conf))

    lines.sort(key=lambda ln: ln.top)
    return lines
