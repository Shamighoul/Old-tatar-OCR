"""Download-and-verify cache for model weight files (Zenodo / Hugging Face URLs).

Files are pinned by checksum so that every experiment records exactly which weights
were used. Nothing is downloaded at import time.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class ModelFile:
    key: str
    url: str
    filename: str
    checksum: str  # "md5:<hex>" or "sha256:<hex>"
    size_bytes: int | None = None
    license: str = "unknown"
    source: str = ""


def models_dir() -> Path:
    root = os.environ.get("OLD_TATAR_OCR_MODELS", "~/.cache/old_tatar_ocr/models")
    path = Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, checksum: str) -> bool:
    algo, _, expected = checksum.partition(":")
    return _digest(path, algo) == expected


def fetch(model: ModelFile, timeout: int = 60, retries: int = 50) -> Path:
    """Return a local path to ``model``, downloading and verifying it if needed.

    Downloads resume from a ``.part`` file (HTTP Range) and retry on stalls, so large
    weights survive slow or flaky connections.
    """
    target = models_dir() / model.key / model.filename
    if target.exists() and verify(target, model.checksum):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries):
        done = tmp.stat().st_size if tmp.exists() else 0
        if model.size_bytes and done >= model.size_bytes:
            break
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(model.url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if done and r.status_code == 206 else "wb"
                with tmp.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            if not model.size_bytes or tmp.stat().st_size >= model.size_bytes:
                break
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            time.sleep(min(60, 2 ** min(attempt, 6)))
    if not verify(tmp, model.checksum):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {model.url} (expected {model.checksum})")
    tmp.replace(target)
    return target
