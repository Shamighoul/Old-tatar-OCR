"""End-to-end page processing with independent stages.

    image -> preprocessing -> layout/recognition (OCR backend) -> [transcription] -> [translation]

Only the OCR stage exists in Phase 1. The result schema already carries the other
representations so that later stages add fields instead of changing the contract:

* ``raw_ocr``       literal machine transcription in Arabic script (never LM-corrected)
* ``corrected_ocr`` reserved for an explicit, separately stored post-correction
* ``modern_tatar``  reserved (Phase 4: Old Tatar -> modern Tatar Cyrillic)
* ``translation``   reserved (Phase 5: modern Tatar -> ru / ba)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from ocr.base import OCRBackend
from preprocessing import PreprocessConfig, preprocess


def confidence_band(confidence: float | None) -> str | None:
    """Coarse UI band. Engine confidences are uncalibrated; thresholds are provisional."""
    if confidence is None:
        return None
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.7:
        return "medium"
    return "low"


def run_ocr(image_path: str | Path, backend: OCRBackend, pre_cfg: PreprocessConfig,
            include_lines: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    original = Image.open(image_path)
    pre = preprocess(original, pre_cfg)
    result = backend.recognize(pre.image)
    out: dict[str, Any] = {
        "raw_ocr": result.text,
        "corrected_ocr": None,
        "modern_tatar": None,
        "translation": {},
        "confidence": result.confidence,
        "confidence_band": confidence_band(result.confidence),
        "model": backend.name,
        "model_version": result.model_version,
        "processing_time": round(time.perf_counter() - t0, 3),
        "ocr_time": round(result.processing_time, 3),
        "model_load_time": round(backend.load_time or 0.0, 3),
        "image": {"path": str(image_path), "size": list(original.size), "processed_size": list(pre.image.size)},
        "preprocessing": {"config": pre_cfg.name, "steps": pre.steps},
        "stages": {"transcription": "not implemented (phase 4)", "translation": "not implemented (phase 5)"},
    }
    if include_lines:
        out["lines"] = [{"text": l.text, "confidence": l.confidence,
                         "confidence_band": confidence_band(l.confidence), "bbox": l.bbox}
                        for l in result.lines]
    return out
