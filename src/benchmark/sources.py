"""data/sources.yaml: source metadata contract and split policy.

Every source (book / manuscript / newspaper issue) is the unit of splitting. Sources that share
a series, press or scribe form a ``family``; a family must never be split across train/val/test
either — otherwise near-identical typography (or handwriting) leaks into the test set.

    python -m benchmark.sources validate
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from benchmark.manifest import TYPES

COMMON = ["source_id", "kind", "title", "approx_year", "type", "script_style", "language", "image_license"]
BY_KIND = {
    "wikisource": ["commons_file", "wikisource_index", "gt_license", "split"],
    "commons_pages": ["commons_file", "pages", "planned_split", "gt_license"],
    "commons_image": ["commons_file", "split"],
}
SPLITS = {"train", "val", "test", "unlabeled"}
RECOMMENDED = ["author", "publisher", "place", "font"]


def load_sources(path: str | Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["sources"]


def eval_split(src: dict) -> str:
    """Split the source belongs to (planned split for sources still being annotated)."""
    return src.get("split") if src.get("split") not in (None, "unlabeled") else src.get("planned_split", "unlabeled")


def validate_sources(sources: list[dict]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    seen = set()
    families: dict[str, set[str]] = defaultdict(set)
    for s in sources:
        sid = s.get("source_id", "?")
        if sid in seen:
            errors.append(f"{sid}: duplicate source_id")
        seen.add(sid)
        kind = s.get("kind")
        if kind not in BY_KIND:
            errors.append(f"{sid}: unknown kind {kind!r}")
            continue
        for f in COMMON + BY_KIND[kind]:
            if s.get(f) in (None, ""):
                errors.append(f"{sid}: missing required field {f!r}")
        if s.get("type") not in TYPES:
            errors.append(f"{sid}: type must be one of {sorted(TYPES)}")
        if s.get("type") == "handwritten" and not s.get("writer"):
            errors.append(f"{sid}: handwritten sources need 'writer' (use 'unknown' if not known)")
        for key in ("split", "planned_split"):
            if s.get(key) is not None and s[key] not in SPLITS:
                errors.append(f"{sid}: {key} must be one of {sorted(SPLITS)}")
        if kind == "commons_pages" and not all(isinstance(p, int) and p > 0 for p in s.get("pages") or []):
            errors.append(f"{sid}: pages must be a list of positive page numbers")
        missing = [f for f in RECOMMENDED if not s.get(f)]
        if missing:
            warnings.append(f"{sid}: recommended metadata missing: {missing}")
        split = eval_split(s)
        if split in ("train", "val", "test"):
            families[s.get("family") or sid].add(split)
    for family, splits in sorted(families.items()):
        if len(splits) > 1:
            errors.append(f"family {family!r} spans several splits {sorted(splits)}")
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("path", nargs="?", default="data/sources.yaml")
    args = p.parse_args(argv)
    sources = load_sources(args.path)
    errors, warnings = validate_sources(sources)
    by_split: dict[str, list[str]] = defaultdict(list)
    for s in sources:
        by_split[eval_split(s)].append(s["source_id"])
    for split, ids in sorted(by_split.items()):
        print(f"{split}: {len(ids)} sources — {', '.join(ids)}")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
