"""Preprocessing pipeline driven by a YAML config.

Order: EXIF orientation -> document boundary -> perspective -> optional crop -> deskew
-> resize -> contrast -> (optional, off by default) binarization.

The input image object is never modified; callers keep the original. Every applied
step is logged so OCR(original) vs OCR(preprocessed) runs are traceable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from preprocessing import steps

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "preprocess"


@dataclass
class PreprocessConfig:
    name: str = "original"
    exif_orientation: bool = True
    document_detection: bool = False
    perspective_correction: bool = False  # needs document_detection
    crop: tuple[float, float, float, float] | None = None  # fractions (x0, y0, x1, y1)
    deskew: bool = False
    deskew_max_angle: float = 5.0
    deskew_min_angle: float = 0.3  # smaller estimates are treated as noise
    max_side: int | None = None
    min_side: int | None = None
    contrast: str | None = None  # None | "clahe" | "autocontrast"
    binarize: bool = False  # deliberately off by default (destroys thin strokes/dots)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreprocessConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown preprocessing keys: {sorted(unknown)}")
        if d.get("crop") is not None:
            d = {**d, "crop": tuple(d["crop"])}
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["crop"] is not None:
            d["crop"] = list(d["crop"])
        return d


@dataclass
class PreprocessResult:
    image: Image.Image
    config: PreprocessConfig
    steps: list[dict[str, Any]] = field(default_factory=list)


def load_preprocess_config(name_or_path: str | None) -> PreprocessConfig:
    """``None``/"original" -> EXIF only; a bare name -> configs/preprocess/<name>.yaml."""
    if name_or_path in (None, "", "original"):
        return PreprocessConfig()
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("name", path.stem)
    return PreprocessConfig.from_dict(data)


def preprocess(image: Image.Image, config: PreprocessConfig) -> PreprocessResult:
    log: list[dict[str, Any]] = []
    img = image.convert("RGB") if image.mode != "RGB" else image.copy()

    if config.exif_orientation:
        # EXIF lives on the original object; convert() drops it.
        oriented, info = steps.exif_orientation(image)
        img = oriented.convert("RGB")
        log.append({"step": "exif_orientation", **info})

    if config.document_detection:
        quad = steps.detect_document(img)
        log.append({"step": "document_detection", "found": quad is not None,
                    "quad": quad.round(1).tolist() if quad is not None else None})
        if quad is not None and config.perspective_correction:
            img = steps.perspective_correct(img, quad)
            log.append({"step": "perspective_correction", "size": list(img.size)})

    if config.crop is not None:
        img = steps.crop_fraction(img, config.crop)
        log.append({"step": "crop", "box": list(config.crop), "size": list(img.size)})

    if config.deskew:
        angle = steps.estimate_skew(img, max_angle=config.deskew_max_angle)
        applied = abs(angle) >= config.deskew_min_angle
        if applied:
            img = steps.rotate(img, angle)
        log.append({"step": "deskew", "angle": round(angle, 2), "applied": applied})

    if config.max_side or config.min_side:
        img, scale = steps.resize(img, config.max_side, config.min_side)
        log.append({"step": "resize", "scale": round(scale, 4), "size": list(img.size)})

    if config.contrast == "clahe":
        img = steps.clahe(img)
        log.append({"step": "contrast", "method": "clahe"})
    elif config.contrast == "autocontrast":
        img = steps.autocontrast(img)
        log.append({"step": "contrast", "method": "autocontrast"})
    elif config.contrast is not None:
        raise ValueError(f"unknown contrast method {config.contrast!r}")

    if config.binarize:
        img = steps.binarize(img)
        log.append({"step": "binarize", "method": "adaptive_gaussian"})

    return PreprocessResult(image=img, config=config, steps=log)
