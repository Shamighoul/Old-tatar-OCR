"""Export line-level training data from verified PAGE XML ground truth.

    python scripts/export_lines.py --out data/processed/lines              # gold + silver_proofread
    python scripts/export_lines.py --statuses gold_verified --pad 4

For every page whose *current* GT version has a PAGE XML (data/gt_history) and an accepted
status, each line is cropped (polygon bounding box + padding) from the page image and saved as
<out>/<split>/<page_id>_l###.png; <out>/lines.jsonl lists image, text, page, source and split.
Splits come from the manifest (which follows source-level planned splits), so no page — and no
source family — is shared between splits. kraken can also train directly from the PAGE XML.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from benchmark.gt_store import GTStore  # noqa: E402
from benchmark.manifest import read_manifest, resolve  # noqa: E402
from benchmark.pagexml import parse_page_xml  # noqa: E402

DATA = ROOT / "data"


def export(data: Path, out: Path, statuses: set[str], pad: int) -> dict[str, int]:
    manifest = data / "benchmark_manifest.csv"
    store = GTStore(data)
    counts: dict[str, int] = {}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "lines.jsonl").open("w", encoding="utf-8") as jl:
        for row in read_manifest(manifest):
            cur = store.current(row["id"])
            if not cur or cur["status"] not in statuses or not cur.get("has_page_xml"):
                continue
            doc = parse_page_xml(store.page_xml(row["id"], cur["version"]))
            split = row["split"]
            (out / split).mkdir(exist_ok=True)
            with Image.open(resolve(manifest, row["image_path"])) as page:
                page = page.convert("RGB")
                for n, line in enumerate(doc.lines, 1):
                    if not line.text.strip() or not line.polygon:
                        continue
                    xs, ys = zip(*line.polygon)
                    box = (max(0, min(xs) - pad), max(0, min(ys) - pad),
                           min(page.width, max(xs) + pad), min(page.height, max(ys) + pad))
                    name = f"{row['id']}_l{n:03d}.png"
                    page.crop(box).save(out / split / name)
                    jl.write(json.dumps({"image": f"{split}/{name}", "text": line.text, "page_id": row["id"],
                                         "source_id": row["source_id"], "split": split, "bbox": box,
                                         "gt_version": cur["version"], "gt_status": cur["status"]},
                                        ensure_ascii=False) + "\n")
                    counts[split] = counts.get(split, 0) + 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--out", default=str(DATA / "processed" / "lines"))
    p.add_argument("--statuses", default="gold_verified,silver_proofread")
    p.add_argument("--pad", type=int, default=4)
    args = p.parse_args()
    counts = export(Path(args.data), Path(args.out), set(args.statuses.split(",")), args.pad)
    print(f"lines per split: {counts or 'none (no verified line-level GT yet)'} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
