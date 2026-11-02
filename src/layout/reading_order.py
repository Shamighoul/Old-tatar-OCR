"""Right-to-left reading order for detector outputs (box-level OCR engines).

Detectors such as PaddleOCR/EasyOCR return boxes sorted top-to-bottom, left-to-right
and may split one printed line into several boxes. For Arabic script we group boxes
into rows by vertical overlap and read each row right-to-left.

Limitation (documented in docs/architecture.md): multi-column pages are read row by
row across columns; proper column handling needs a region-level layout model.
"""

from __future__ import annotations

from ocr.base import OCRLine


def _overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    top, bottom = max(a[1], b[1]), min(a[3], b[3])
    inter = max(0, bottom - top)
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def group_rows(lines: list[OCRLine], min_overlap: float = 0.5) -> list[list[OCRLine]]:
    boxed = [l for l in lines if l.bbox is not None]
    boxed.sort(key=lambda l: (l.bbox[1] + l.bbox[3]) / 2)
    rows: list[list[OCRLine]] = []
    for line in boxed:
        for row in rows:
            if any(_overlap_ratio(line.bbox, other.bbox) >= min_overlap for other in row):
                row.append(line)
                break
        else:
            rows.append([line])
    for row in rows:
        row.sort(key=lambda l: -l.bbox[2])  # right edge, descending -> RTL
    rows.sort(key=lambda r: min(l.bbox[1] for l in r))
    return rows


def merge_rows_rtl(lines: list[OCRLine], min_overlap: float = 0.5) -> list[OCRLine]:
    """Merge boxes of the same visual row into one RTL line (text joined with spaces)."""
    merged = []
    for row in group_rows(lines, min_overlap):
        confs = [(l.confidence, max(1, len(l.text))) for l in row if l.confidence is not None]
        merged.append(OCRLine(
            text=" ".join(l.text for l in row if l.text),
            confidence=sum(c * w for c, w in confs) / sum(w for _, w in confs) if confs else None,
            bbox=(min(l.bbox[0] for l in row), min(l.bbox[1] for l in row),
                  max(l.bbox[2] for l in row), max(l.bbox[3] for l in row)),
        ))
    return merged
