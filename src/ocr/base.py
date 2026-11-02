"""Backend-agnostic OCR interface.

The rest of the system (CLI, evaluation, future API) only sees ``OCRBackend`` and
``OCRResult``; concrete engines live in adapter modules and are created through
``ocr.registry`` so that heavy dependencies are imported lazily.
"""

from __future__ import annotations

import abc
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from PIL import Image

from text.arabic_normalization import source_preserving_text


class BackendUnavailable(RuntimeError):
    """Backend cannot run here (missing package/binary/credentials/licence/hardware)."""


@dataclass
class OCRLine:
    text: str
    confidence: float | None = None
    # Axis-aligned box (x0, y0, x1, y1) in pixels of the image passed to recognize().
    bbox: tuple[int, int, int, int] | None = None
    # Optional polygon / baseline in the same coordinate frame.
    polygon: list[tuple[int, int]] | None = None
    baseline: list[tuple[int, int]] | None = None


@dataclass
class OCRResult:
    """Raw recognition output of one page/image.

    ``text`` is the *raw* model transcription in Arabic script (only
    ``source_preserving_text`` cleanup). No language-model correction is ever applied
    here; a future post-correction stage must store its output separately.
    """

    text: str
    model_name: str
    model_version: str
    confidence: float | None = None
    processing_time: float = 0.0
    lines: list[OCRLine] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OCRResult":
        lines = [
            OCRLine(
                text=l["text"],
                confidence=l.get("confidence"),
                bbox=tuple(l["bbox"]) if l.get("bbox") is not None else None,
                polygon=[tuple(p) for p in l["polygon"]] if l.get("polygon") else None,
                baseline=[tuple(p) for p in l["baseline"]] if l.get("baseline") else None,
            )
            for l in d.get("lines", [])
        ]
        return cls(
            text=d["text"], model_name=d["model_name"], model_version=d["model_version"],
            confidence=d.get("confidence"), processing_time=d.get("processing_time", 0.0),
            lines=lines, metadata=d.get("metadata", {}),
        )


def weighted_confidence(lines: list[OCRLine]) -> float | None:
    """Page confidence = character-length-weighted mean of line confidences.

    Engine confidences are *not calibrated* and not comparable across engines.
    """
    scored = [(l.confidence, max(1, len(l.text))) for l in lines if l.confidence is not None]
    if not scored:
        return None
    return sum(c * w for c, w in scored) / sum(w for _, w in scored)


class OCRBackend(abc.ABC):
    """Base class for OCR/HTR adapters.

    Subclasses set ``name``/``model_version`` and implement ``_load`` and
    ``_recognize``; ``recognize`` adds lazy loading, timing and text hygiene.
    """

    #: registry key, e.g. "paddle", "kraken-ottoman"
    name: str = "base"
    #: human-readable model identity incl. revision/hash once loaded
    model_version: str = "unknown"

    def __init__(self, **config: Any) -> None:
        self.config: dict[str, Any] = config
        self._loaded = False
        self.load_time: float | None = None

    # -- availability -------------------------------------------------------------
    @classmethod
    def check_available(cls) -> tuple[bool, str]:
        """Cheap check (imports/binaries/credentials) without downloading weights."""
        return True, "ok"

    # -- lifecycle ----------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        ok, reason = self.check_available()
        if not ok:
            raise BackendUnavailable(f"{self.name}: {reason}")
        t0 = time.perf_counter()
        self._load()
        self.load_time = time.perf_counter() - t0
        self._loaded = True

    def _load(self) -> None:  # pragma: no cover - trivial default
        pass

    # -- inference ----------------------------------------------------------------
    def recognize(self, image: Image.Image) -> OCRResult:
        self.load()
        t0 = time.perf_counter()
        result = self._recognize(image.convert("RGB"))
        result.processing_time = time.perf_counter() - t0
        result.text = source_preserving_text(result.text)
        if result.confidence is None:
            result.confidence = weighted_confidence(result.lines)
        return result

    @abc.abstractmethod
    def _recognize(self, image: Image.Image) -> OCRResult:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "model_version": self.model_version, "config": self.config}
