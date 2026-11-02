"""Prepare human annotation tasks (benchmark preparation tool). Never creates labels.

* ``--run results/<model>/<run>`` -> ``verify_tasks.csv``: pages with *silver* GT that an
  annotator should check against the image (target: gold_verified), sorted by how much
  the model disagrees with GT (likely GT errors or hard pages first).
* ``--model NAME`` -> pre-annotation for pages without GT (split=unlabeled):
  ``<id>.draft.txt`` marked as UNVERIFIED machine output, and ``lines_to_check.tsv``
  with lines sorted by ascending confidence (active-learning seed).

Drafts are written to data/annotation/, never into data/ground_truth.

    python scripts/make_annotation_tasks.py --run results/kraken-ppocrv6/20261102-155320-original
    python scripts/make_annotation_tasks.py --model kraken-ppocrv6 --preprocess mobile
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.manifest import read_manifest, resolve  # noqa: E402

HEADER = ("# DRAFT — UNVERIFIED MACHINE OUTPUT ({model}).\n"
          "# Correct against the image, delete these two lines, and only then add it as ground truth.\n")


def verify_tasks(run: Path, rows: dict[str, dict], out: Path) -> int:
    metrics = json.loads((run / "metrics.json").read_text())
    tasks = []
    for page in metrics.get("pages", []):
        row = rows.get(page["id"])
        if row and row.get("gt_status", "").startswith("silver"):
            tasks.append({"id": page["id"], "image": row["image_path"], "ground_truth": row["ground_truth_path"],
                          "gt_status": row["gt_status"], "model": metrics["model"],
                          "model_cer_normalized": round(page["cer_normalized"], 4), "notes": row.get("notes", "")})
    tasks.sort(key=lambda r: -r["model_cer_normalized"])
    if tasks:
        with (out / "verify_tasks.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(tasks[0]))
            w.writeheader()
            w.writerows(tasks)
    return len(tasks)


def draft_tasks(model: str, preprocess_name: str, manifest: Path, rows: dict[str, dict], out: Path) -> int:
    from PIL import Image

    from ocr.registry import create_backend
    from preprocessing import load_preprocess_config, preprocess

    backend = create_backend(model)
    cfg = load_preprocess_config(preprocess_name)
    lines = []
    for rid, row in rows.items():
        if row["split"] != "unlabeled":
            continue
        result = backend.recognize(preprocess(Image.open(resolve(manifest, row["image_path"])), cfg).image)
        (out / f"{rid}.draft.txt").write_text(HEADER.format(model=result.model_version) + result.text + "\n",
                                              encoding="utf-8")
        lines += [{"id": rid, "line": i, "confidence": l.confidence, "bbox": l.bbox, "text": l.text}
                  for i, l in enumerate(result.lines)]
    lines.sort(key=lambda r: (r["confidence"] is None, r["confidence"] or 0.0))
    with (out / "lines_to_check.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "line", "confidence", "bbox", "text"], delimiter="\t")
        w.writeheader()
        w.writerows(lines)
    return len(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="benchmark run directory (for verify tasks)")
    p.add_argument("--model", help="backend for drafting unlabeled pages")
    p.add_argument("--preprocess", default="original")
    p.add_argument("--manifest", default=str(ROOT / "data" / "benchmark_manifest.csv"))
    p.add_argument("--out", default=str(ROOT / "data" / "annotation"))
    args = p.parse_args()
    if not (args.run or args.model):
        p.error("give --run and/or --model")

    manifest = Path(args.manifest)
    rows = {r["id"]: r for r in read_manifest(manifest)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.run:
        print(f"verify tasks: {verify_tasks(Path(args.run), rows, out)} -> {out / 'verify_tasks.csv'}")
    if args.model:
        print(f"draft lines: {draft_tasks(args.model, args.preprocess, manifest, rows, out)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
