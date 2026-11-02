"""Line segmentation with kraken's BLLA baseline model (page -> ordered text lines).

Used by line-level recognizers that have no detector of their own (e.g. line VLMs).
Lines come in kraken's reading order for ``horizontal-rl`` (right-to-left pages);
crops are produced with kraken's polygon extractor (dewarped, background masked),
which is how eScriptorium/kraken training lines are cut.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class LineRegion:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]] | None
    baseline: list[tuple[int, int]] | None


class KrakenLineSegmenter:
    def __init__(self, model_path: str | None = None, text_direction: str = "horizontal-rl") -> None:
        from kraken.configs import SegmentationInferenceConfig
        from kraken.tasks import SegmentationTaskModel

        self._model = SegmentationTaskModel.load_model(model_path)
        self._config = SegmentationInferenceConfig(text_direction=text_direction)
        path = Path(str(model_path or resources.files("kraken").joinpath("blla.mlmodel")))
        self.version = f"{path.name}@{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"

    def segment(self, image: Image.Image) -> Any:
        """Raw kraken ``Segmentation`` (needed by kraken recognizers).

        If ``OLD_TATAR_OCR_SEG_CACHE`` points to a directory, segmentations are cached
        there by (image pixels, segmentation model, direction). All kraken recognizers
        and line VLMs then see *identical* lines, which isolates recognizer quality in
        comparisons and avoids recomputing the slowest step.
        """
        cache_dir = os.environ.get("OLD_TATAR_OCR_SEG_CACHE")
        if not cache_dir:
            return self._model.predict(image, self._config)
        digest = hashlib.sha256(image.tobytes() + repr((image.size, image.mode, self.version,
                                                         self._config.text_direction)).encode()).hexdigest()
        path = Path(cache_dir).expanduser() / f"{digest[:32]}.pkl"
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        seg = self._model.predict(image, self._config)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(seg, f)
        return seg

    def lines(self, image: Image.Image, segmentation: Any | None = None) -> list[LineRegion]:
        from kraken.lib.segmentation import extract_polygons

        seg = segmentation if segmentation is not None else self.segment(image)
        out = []
        for crop, line in extract_polygons(image, seg):
            boundary = [tuple(map(int, p)) for p in (getattr(line, "boundary", None) or [])]
            if boundary:
                xs, ys = zip(*boundary)
                bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                bbox = tuple(map(int, line.bbox))
            out.append(LineRegion(image=crop.convert("RGB"), bbox=bbox, polygon=boundary or None,
                                  baseline=[tuple(map(int, p)) for p in (getattr(line, "baseline", None) or [])] or None))
        return out
