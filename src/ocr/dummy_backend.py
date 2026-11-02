"""Model-free backend used by unit/integration tests."""

from __future__ import annotations

from PIL import Image

from ocr.base import OCRBackend, OCRLine, OCRResult


class DummyBackend(OCRBackend):
    name = "dummy"
    model_version = "dummy-1"

    def _recognize(self, image: Image.Image) -> OCRResult:
        text = self.config.get("text", "")
        w, h = image.size
        lines = [OCRLine(text=t, confidence=self.config.get("confidence", 0.5), bbox=(0, 0, w, h))
                 for t in text.splitlines() if t]
        return OCRResult(text=text, model_name=self.name, model_version=self.model_version,
                         lines=lines, metadata={"image_size": [w, h]})
