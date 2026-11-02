"""EasyOCR adapter (CRAFT detector + Arabic-script CRNN), a cheap extra baseline."""

from __future__ import annotations

import importlib.util

import numpy as np
from PIL import Image

from layout.reading_order import merge_rows_rtl
from ocr.base import OCRBackend, OCRLine, OCRResult


class EasyOCRBackend(OCRBackend):
    name = "easyocr"

    @classmethod
    def check_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("easyocr") is None:
            return False, "package 'easyocr' not installed (pip install -e '.[easyocr]')"
        return True, "ok"

    def _load(self) -> None:
        import easyocr

        self._langs = list(self.config.get("langs", ["ar", "fa"]))
        self._reader = easyocr.Reader(self._langs, gpu=False, verbose=False)
        self.model_version = f"easyocr {easyocr.__version__} langs={'+'.join(self._langs)}"

    def _recognize(self, image: Image.Image) -> OCRResult:
        w, h = image.size
        max_side = self.config.get("max_side", 2400)
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        detections = self._reader.readtext(np.asarray(image), detail=1, paragraph=False)
        raw = []
        for box, text, conf in detections:
            xs = [p[0] / scale for p in box]
            ys = [p[1] / scale for p in box]
            raw.append(OCRLine(text=text, confidence=float(conf),
                               bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))))
        lines = merge_rows_rtl(raw)
        return OCRResult(text="\n".join(l.text for l in lines), model_name=self.name,
                         model_version=self.model_version, lines=lines,
                         metadata={"n_boxes": len(raw), "work_scale": scale})
