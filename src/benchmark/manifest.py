"""benchmark_manifest.csv: reading, writing and validation.

Paths inside the manifest are relative to the manifest's directory (``data/``).

Validation enforces the data contract, most importantly that splits are made at the
level of ``source_id`` (book / manuscript / newspaper issue): pages of one source may
never appear in two of train/val/test.

    python -m benchmark.manifest validate data/benchmark_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED_COLUMNS = [
    "id", "image_path", "source_id", "page", "type", "script_style",
    "approx_year", "ground_truth_path", "split",
]
EXTRA_COLUMNS = ["subset", "language", "gt_status", "license", "notes"]
COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS

TYPES = {"printed", "handwritten"}
SPLITS = {"train", "val", "test", "unlabeled"}
EVAL_SPLITS = {"train", "val", "test"}
GT_STATUSES = {"", "none", "draft_unverified", "silver_unproofread", "silver_proofread", "gold_verified"}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_manifest(rows: list[dict[str, str]], path: str | Path) -> None:
    rows = sorted(rows, key=lambda r: (r["source_id"], int(r["page"]) if str(r["page"]).isdigit() else 0, r["id"]))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})


def resolve(manifest_path: str | Path, relative: str) -> Path:
    return (Path(manifest_path).parent / relative).resolve()


def source_leakage(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    """source_id -> set of evaluation splits, for sources found in more than one."""
    splits_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] in EVAL_SPLITS:
            splits_by_source[row["source_id"]].add(row["split"])
    return {s: sp for s, sp in splits_by_source.items() if len(sp) > 1}


def validate(rows: list[dict[str, str]], manifest_path: str | Path | None = None,
             check_files: bool = True) -> list[str]:
    errors: list[str] = []
    if rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            return [f"missing columns: {missing}"]
    seen: set[str] = set()
    for row in rows:
        rid = row["id"]
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)
        if row["type"] not in TYPES:
            errors.append(f"{rid}: type must be one of {sorted(TYPES)}, got {row['type']!r}")
        if row["split"] not in SPLITS:
            errors.append(f"{rid}: split must be one of {sorted(SPLITS)}, got {row['split']!r}")
        if row.get("gt_status", "") not in GT_STATUSES:
            errors.append(f"{rid}: unknown gt_status {row.get('gt_status')!r}")
        if row["split"] in EVAL_SPLITS and not row["ground_truth_path"]:
            errors.append(f"{rid}: split {row['split']} requires ground_truth_path (use split=unlabeled)")
        if check_files and manifest_path is not None:
            for col in ("image_path", "ground_truth_path"):
                if row[col] and not resolve(manifest_path, row[col]).exists():
                    errors.append(f"{rid}: {col} not found: {row[col]}")
    for source, splits in sorted(source_leakage(rows).items()):
        errors.append(f"source {source!r} appears in several splits {sorted(splits)} (split by source_id!)")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("manifest")
    v.add_argument("--no-files", action="store_true", help="skip file existence checks")
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    errors = validate(rows, args.manifest, check_files=not args.no_files)
    by_split: dict[str, int] = defaultdict(int)
    for r in rows:
        by_split[r["split"]] += 1
    print(f"{len(rows)} rows, {len({r['source_id'] for r in rows})} sources, splits: {dict(by_split)}")
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
