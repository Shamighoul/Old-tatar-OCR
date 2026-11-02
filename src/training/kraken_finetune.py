"""Strategy A: fine-tune a kraken line recognizer (PP-OCRv6 or OpenITI VGSL) with ``ketos train``.

The run directory records everything needed to reproduce it (config.yaml with base model hash,
dataset version, hyper-parameters, seed, git state, hardware, library versions) plus the ketos
log; the best checkpoint is evaluated with the same benchmark runner as the Phase 1 baselines
(``kraken-custom`` backend), so fine-tuned and baseline numbers are directly comparable.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from evaluation import repro
from ocr.kraken_backend import MODEL_CATALOG
from ocr.model_store import fetch


@dataclass
class FinetuneConfig:
    base: str = "ppocrv6-medium"  # MODEL_CATALOG key or path to a model file
    train_list: str = ""
    val_list: str = ""
    out_dir: str = "models/finetuned/run"
    max_epochs: int = 50
    min_epochs: int = 5
    lag: int = 5  # early stopping patience (validations)
    lrate: float = 1e-4  # low LR: adapt a pretrained model, do not retrain it
    # The ppocrv6 family defaults to AdamW+Muon with a cosine schedule, which needs a known number
    # of steps and crashes with early stopping in kraken 7.1.1 (int(inf) in configure_optimizers).
    # Early stopping therefore uses plain AdamW with a constant LR; use quit="fixed" for cosine.
    optimizer: str = "AdamW"
    schedule: str = "constant"
    quit: str = "early"
    warmup: int = 0  # steps of linear LR warm-up
    batch_size: int = 1  # kraken pads batches; keep 1 like at inference (see docs/experiments.md)
    augment: bool = True
    normalization: str = "NFD"  # OpenITI and PP-OCRv6 models were trained on NFD text
    resize: str = "union"  # add Tatar-specific code points (ڭ ۋ ࢭ ...) to the base codec
    seed: int = 42
    threads: int = field(default_factory=lambda: int(os.environ.get("OLD_TATAR_OCR_THREADS", os.cpu_count() or 1)))
    device: str = "cpu"
    extra_args: list[str] = field(default_factory=list)


def base_model_path(base: str) -> Path:
    return fetch(MODEL_CATALOG[base]) if base in MODEL_CATALOG else Path(base).expanduser()


def build_command(cfg: FinetuneConfig, ketos: str = "ketos") -> list[str]:
    cmd = [ketos, "-d", cfg.device, "--threads", str(cfg.threads), "-s", str(cfg.seed), "--deterministic",
           "train", "-f", "page", "-i", str(base_model_path(cfg.base)), "--resize", cfg.resize,
           "-t", cfg.train_list, "-o", str(Path(cfg.out_dir) / "checkpoints"),
           "-u", cfg.normalization, "-n", "-q", cfg.quit, "-N", str(cfg.max_epochs),
           "--min-epochs", str(cfg.min_epochs), "--lag", str(cfg.lag), "-r", str(cfg.lrate),
           "--optimizer", cfg.optimizer, "--schedule", cfg.schedule, "--warmup", str(cfg.warmup),
           "-B", str(cfg.batch_size), "--augment" if cfg.augment else "--no-augment"]
    if cfg.schedule == "cosine":
        if cfg.quit != "fixed":
            raise ValueError("cosine schedule needs quit='fixed' (a known number of epochs)")
        cmd += ["--cos-max", str(cfg.max_epochs)]
    if cfg.val_list:
        cmd += ["-e", cfg.val_list]
    return cmd + list(cfg.extra_args)


def find_best_model(checkpoint_dir: Path, ketos: str = "ketos") -> Path | None:
    """Best weights written by ketos; if training was interrupted, convert the best Lightning
    checkpoint (``checkpoint_<epoch>-<val_metric>.ckpt``, highest metric wins) with ``ketos convert``."""
    weights = [p for p in checkpoint_dir.rglob("*") if p.suffix in (".safetensors", ".mlmodel")]
    best = [p for p in weights if "best" in p.name]
    if best or weights:
        return max(best or weights, key=lambda p: p.stat().st_mtime)
    ckpts = list(checkpoint_dir.rglob("*.ckpt"))
    if not ckpts:
        return None

    def score(p: Path) -> float:
        try:
            return float(p.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return float("-inf")

    best_ckpt = max(ckpts, key=score)
    target = checkpoint_dir / f"{best_ckpt.stem}.safetensors"
    subprocess.run([ketos, "convert", "-o", str(target), str(best_ckpt)], check=True, capture_output=True)
    return target


def run_finetune(cfg: FinetuneConfig, dataset_version: str = "", note: str = "") -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ketos = shutil.which("ketos") or str(Path(sys.executable).with_name("ketos"))
    cmd = build_command(cfg, ketos)
    base_path = base_model_path(cfg.base)
    record = {
        "config": asdict(cfg), "command": cmd, "note": note, "dataset_version": dataset_version,
        "base_model": {"key_or_path": cfg.base, "file": str(base_path),
                       "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest()},
        "git": repro.git_info(), "hardware": repro.hardware_info(), "libraries": repro.library_versions(),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(record, allow_unicode=True, sort_keys=False), encoding="utf-8")
    t0 = time.perf_counter()
    with (out / "ketos.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    record["train_seconds"] = round(time.perf_counter() - t0, 1)
    record["returncode"] = proc.returncode
    best = find_best_model(out / "checkpoints", ketos)
    if best is not None:
        final = out / f"model{best.suffix}"
        shutil.copyfile(best, final)
        record["model"] = str(final)
        record["model_sha256"] = hashlib.sha256(final.read_bytes()).hexdigest()
        record["model_size_mb"] = round(final.stat().st_size / 1e6, 1)
    (out / "config.yaml").write_text(yaml.safe_dump(record, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return record
