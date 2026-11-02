"""Prefetch model weights (resumable, checksum-verified) so benchmark runs never wait on the network.

    python scripts/download_models.py                 # kraken models, CHURRO GGUF, Azra
    python scripts/download_models.py churro azra
    python scripts/download_models.py --check churro  # exit 0 only if already complete
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr.churro_backend import GGUF_FILES  # noqa: E402
from ocr.kraken_backend import MODEL_CATALOG  # noqa: E402
from ocr.model_store import fetch, models_dir, verify  # noqa: E402
from ocr.ottoman_backend import AZRA_MODEL  # noqa: E402

AZRA_FULL_REVISION = "7c1c429e9f3d3bad1d59b6ef802264f694b171be"
CHURRO_DEFAULT = ("Q4_K_M", "mmproj-Q8_0")


def churro_files():
    return [GGUF_FILES[k] for k in CHURRO_DEFAULT]


def is_complete(group: str) -> bool:
    if group == "kraken":
        files = list(MODEL_CATALOG.values())
    elif group == "churro":
        files = churro_files()
    elif group == "azra":
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(AZRA_MODEL, "model.safetensors", revision=AZRA_FULL_REVISION)
        return isinstance(path, str) and Path(path).exists()
    else:
        raise ValueError(group)
    for f in files:
        p = models_dir() / f.key / f.filename
        if not (p.exists() and verify(p, f.checksum)):
            return False
    return True


def download(group: str) -> None:
    if group == "kraken":
        for f in MODEL_CATALOG.values():
            fetch(f)
    elif group == "churro":
        for f in churro_files():
            fetch(f)
    elif group == "azra":
        from huggingface_hub import snapshot_download
        for attempt in range(200):
            try:
                snapshot_download(AZRA_MODEL, revision=AZRA_FULL_REVISION)
                return
            except Exception as exc:  # network hiccups: retry with resume
                print(f"azra retry {attempt}: {exc!r}"[:200], flush=True)
                time.sleep(15)
        raise RuntimeError("Azra download failed")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="*", default=["kraken", "churro", "azra"])
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.check:
        return 0 if all(is_complete(g) for g in args.groups) else 1
    for g in args.groups:
        t = time.time()
        download(g)
        print(f"{g} ok ({time.time() - t:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
