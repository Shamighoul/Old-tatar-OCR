"""OCR/HTR backends behind a common interface (see ``ocr.base`` and ``ocr.registry``)."""

from ocr.base import BackendUnavailable, OCRBackend, OCRLine, OCRResult

__all__ = ["BackendUnavailable", "OCRBackend", "OCRLine", "OCRResult"]
