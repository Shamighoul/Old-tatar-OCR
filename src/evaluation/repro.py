"""Reproducibility record: git state, hardware, library versions, dataset hash, seed."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import random
import resource
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRACKED_PACKAGES = [
    "numpy", "pillow", "opencv-python-headless", "rapidfuzz", "torch", "torchvision", "transformers",
    "paddlepaddle", "paddleocr", "paddlex", "kraken", "llama_cpp_python", "easyocr", "qwen-vl-utils",
    "huggingface-hub",
]


def git_info() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def hardware_info() -> dict:
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_gb = None
    try:
        mem_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError):
        pass
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {"platform": platform.platform(), "python": platform.python_version(), "cpu": cpu,
            "cpu_count": os.cpu_count(), "ram_gb": mem_gb, "gpu": gpu or "none (CPU only)",
            "threads": int(os.environ.get("OLD_TATAR_OCR_THREADS", os.cpu_count() or 1))}


def library_versions() -> dict:
    out = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def dataset_version(manifest_path: str | Path, gt_paths: list[Path]) -> str:
    h = hashlib.sha256(Path(manifest_path).read_bytes())
    for p in sorted(gt_paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def gpu_peak_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    return None
