"""Assemble kraken training sets (PAGE XML) from the versioned ground truth.

Only the *current* version of a page is used, only if it is human-verified (accepted statuses)
and has line-level PAGE XML. Test pages are never written — the split comes from the manifest,
which follows source/family-level splits (``benchmark.sources``).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from benchmark.gt_store import GTStore
from benchmark.manifest import read_manifest, resolve

DEFAULT_STATUSES = ("gold_verified", "silver_proofread")
TRAIN_SPLITS = ("train", "val")


@dataclass
class TrainingSet:
    root: Path
    files: dict[str, list[Path]] = field(default_factory=dict)  # split -> PAGE XML files
    pages: list[dict] = field(default_factory=list)  # provenance per page
    version: str = ""

    def list_file(self, split: str) -> Path:
        return self.root / f"{split}.txt"


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copyfile(src, dst)


def eligible_pages(data: Path, statuses: tuple[str, ...] = DEFAULT_STATUSES) -> list[dict]:
    store = GTStore(data)
    out = []
    for row in read_manifest(data / "benchmark_manifest.csv"):
        if row["split"] not in TRAIN_SPLITS:
            continue
        cur = store.current(row["id"])
        if cur and cur["status"] in statuses and cur.get("has_page_xml"):
            out.append({**row, "gt_version": cur["version"], "gt_status": cur["status"], "gt_sha256": cur["sha256"]})
    return out


def sample_pages(pages: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Nested, source-stratified subset of train pages (for learning curves).

    Pages are ordered round-robin over sources after a seeded shuffle, so the first n pages
    cover as many sources as possible and subsets for growing n are nested.
    """
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for p in sorted(pages, key=lambda p: p["id"]):
        by_source.setdefault(p["source_id"], []).append(p)
    for lst in by_source.values():
        rng.shuffle(lst)
    sources = sorted(by_source)
    rng.shuffle(sources)
    ordered = []
    while any(by_source.values()):
        for s in sources:
            if by_source[s]:
                ordered.append(by_source[s].pop())
    return ordered[:n]


def build_training_set(data: Path, out: Path, statuses: tuple[str, ...] = DEFAULT_STATUSES,
                       train_pages: list[dict] | None = None) -> TrainingSet:
    """Write <out>/<split>/<page>.xml (+ image link) and <out>/<split>.txt lists."""
    data, out = Path(data), Path(out)
    store = GTStore(data)
    manifest = data / "benchmark_manifest.csv"
    pages = eligible_pages(data, statuses)
    if train_pages is not None:
        keep = {p["id"] for p in train_pages}
        pages = [p for p in pages if p["split"] != "train" or p["id"] in keep]
    if out.exists():
        shutil.rmtree(out)
    ts = TrainingSet(root=out)
    digest = hashlib.sha256()
    for p in sorted(pages, key=lambda p: p["id"]):
        assert p["split"] in TRAIN_SPLITS, p["id"]  # test pages must never reach training
        split_dir = out / p["split"]
        split_dir.mkdir(parents=True, exist_ok=True)
        image = resolve(manifest, p["image_path"])
        image_name = f"{p['id']}{image.suffix}"
        _link(image, split_dir / image_name)
        xml = store.page_xml(p["id"], p["gt_version"])
        xml = re.sub(r'imageFilename="[^"]*"', f'imageFilename="{image_name}"', xml, count=1)
        xml_path = split_dir / f"{p['id']}.xml"
        xml_path.write_text(xml, encoding="utf-8")
        ts.files.setdefault(p["split"], []).append(xml_path)
        ts.pages.append({k: p[k] for k in ("id", "source_id", "split", "gt_version", "gt_status", "gt_sha256")})
        digest.update(f"{p['id']}:{p['gt_sha256']}".encode())
    for split, files in ts.files.items():
        ts.list_file(split).write_text("\n".join(str(f.resolve()) for f in files) + "\n", encoding="utf-8")
    ts.version = digest.hexdigest()[:16]
    (out / "dataset.json").write_text(json.dumps({"version": ts.version, "statuses": list(statuses),
                                                  "pages": ts.pages}, ensure_ascii=False, indent=2), encoding="utf-8")
    return ts
