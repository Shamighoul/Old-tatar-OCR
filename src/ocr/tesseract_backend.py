"""Tesseract 5 adapter via the system binary (no Python binding needed)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from ocr.base import OCRBackend, OCRResult


class TesseractBackend(OCRBackend):
    name = "tesseract"

    @classmethod
    def check_available(cls) -> tuple[bool, str]:
        exe = shutil.which("tesseract")
        if exe is None:
            return False, ("binary 'tesseract' not found (Ubuntu: sudo apt install tesseract-ocr "
                           "tesseract-ocr-ara tesseract-ocr-fas)")
        langs = subprocess.run([exe, "--list-langs"], capture_output=True, text=True).stdout.split()
        missing = [l for l in ("ara", "fas") if l not in langs]
        if missing:
            return False, f"tesseract traineddata missing: {missing}"
        return True, "ok"

    def _load(self) -> None:
        out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout
        self.model_version = f"{out.splitlines()[0]} lang={self.config.get('lang', 'ara+fas')}"

    def _recognize(self, image: Image.Image) -> OCRResult:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            image.save(path)
            proc = subprocess.run(
                ["tesseract", str(path), "stdout", "-l", self.config.get("lang", "ara+fas"),
                 "--psm", str(self.config.get("psm", 3))],
                capture_output=True, text=True, check=True)
        return OCRResult(text=proc.stdout, model_name=self.name, model_version=self.model_version)
