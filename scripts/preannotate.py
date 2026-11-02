"""Machine pre-annotation for human correction (Phase 2, human-in-the-loop).

For each selected page the OCR backend produces line polygons, baselines, text and confidence;
the result is written as PAGE XML that eScriptorium/Transkribus can import, and recorded in the
versioned GT store as a *draft* (status draft_unverified — never used as ground truth).

    python scripts/preannotate.py                                  # all phase2_pilot pages without GT
    python scripts/preannotate.py --ids commons_salima_1898_p0010 --model kraken-ppocrv6

Output (data/annotation/<batch>/):
    images/<page_id>.jpg       page images named exactly as referenced by the XML
    <page_id>.xml              PAGE XML draft (one region, lines in reading order)
    tasks.csv                  pages sorted by mean line confidence (hardest first)
    lines_to_check.tsv         all lines sorted by confidence (active-learning seed)
    README.md                  correction workflow
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from benchmark.gt_store import GTStore  # noqa: E402
from benchmark.manifest import read_manifest, resolve  # noqa: E402
from benchmark.pagexml import to_page_xml  # noqa: E402
from ocr.registry import create_backend  # noqa: E402

DATA = ROOT / "data"
LOW_CONFIDENCE = 0.8
WEB_FORMATS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

WORKFLOW = """# Correction batch {batch}

Machine drafts by `{model}` — **unverified**. Correct them, then import:

1. eScriptorium: create a document, upload everything from `images/`, then *Import → Transcription (XML)*
   with the `*.xml` files of this folder (files are matched by image name).
2. Correct line polygons/order and the text following `docs/annotation_guidelines.md`
   (letter by letter as printed; no modernisation; ZWNJ where letters are visibly disconnected).
3. Export *PAGE XML*, then:
   `python scripts/import_annotations.py <exported>/*.xml --status gold_verified --author "<name>"`

Start with the pages at the top of `tasks.csv` and the lines at the top of `lines_to_check.tsv`
(lowest model confidence).
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--subset", default="phase2_pilot")
    p.add_argument("--model", default="kraken-ppocrv6")
    p.add_argument("--batch", default=None)
    p.add_argument("--no-store", action="store_true", help="do not record drafts in data/gt_history")
    args = p.parse_args()

    manifest = DATA / "benchmark_manifest.csv"
    rows = read_manifest(manifest)
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    else:
        rows = [r for r in rows if r.get("subset") == args.subset and not r["ground_truth_path"]]
    if not rows:
        print("nothing to pre-annotate", file=sys.stderr)
        return 1
    batch = args.batch or f"{datetime.now():%Y%m%d}-{args.model}"
    out = DATA / "annotation" / batch
    (out / "images").mkdir(parents=True, exist_ok=True)
    backend = create_backend(args.model)
    store = GTStore(DATA)

    tasks, lines_out = [], []
    for i, row in enumerate(rows, 1):
        src_img = resolve(manifest, row["image_path"])
        if src_img.suffix.lower() in WEB_FORMATS:
            image_name = f"{row['id']}{src_img.suffix.lower()}"
            shutil.copyfile(src_img, out / "images" / image_name)
        else:  # e.g. JPEG 2000 scans: annotation tools need JPEG/PNG/TIFF -> lossless PNG copy
            image_name = f"{row['id']}.png"
            with Image.open(src_img) as im:
                im.save(out / "images" / image_name)
        with Image.open(src_img) as im:
            result = backend.recognize(im)
            width, height = im.size
        xml = to_page_xml(image_name, width, height, result.lines, creator=f"old-tatar-ocr preannotate ({result.model_version})",
                          comments=f"DRAFT machine pre-annotation, status=draft_unverified, model={args.model}")
        (out / f"{row['id']}.xml").write_text(xml, encoding="utf-8")
        if not args.no_store:
            store.commit(row["id"], result.text or "", "draft_unverified", author=result.model_version,
                         source=f"machine:{args.model}", note=f"pre-annotation batch {batch}", page_xml=xml)
        confs = [l.confidence for l in result.lines if l.confidence is not None]
        tasks.append({"id": row["id"], "source_id": row["source_id"], "lines": len(result.lines),
                      "mean_confidence": round(sum(confs) / len(confs), 4) if confs else "",
                      "low_confidence_lines": sum(c < LOW_CONFIDENCE for c in confs),
                      "seconds": round(result.processing_time, 1)})
        lines_out += [{"id": row["id"], "line": n, "confidence": round(l.confidence, 4) if l.confidence is not None else "",
                       "bbox": l.bbox, "text": l.text} for n, l in enumerate(result.lines, 1)]
        print(f"[{i}/{len(rows)}] {row['id']}: {len(result.lines)} lines, mean conf {tasks[-1]['mean_confidence']}, "
              f"{result.processing_time:.0f}s", flush=True)

    tasks.sort(key=lambda t: (t["mean_confidence"] == "", t["mean_confidence"] or 0))
    with (out / "tasks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tasks[0]))
        w.writeheader()
        w.writerows(tasks)
    lines_out.sort(key=lambda r: (r["confidence"] == "", r["confidence"] or 0))
    with (out / "lines_to_check.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "line", "confidence", "bbox", "text"], delimiter="\t")
        w.writeheader()
        w.writerows(lines_out)
    (out / "README.md").write_text(WORKFLOW.format(batch=batch, model=args.model), encoding="utf-8")
    print(f"batch -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
