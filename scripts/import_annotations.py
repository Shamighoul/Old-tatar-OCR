"""Import human-corrected PAGE XML (eScriptorium / Transkribus export) as a new GT version.

    python scripts/import_annotations.py exported/*.xml --status gold_verified --author "Name" \
        --note "proofread by philologist"

Per file: page id = image file name without extension (as produced by scripts/preannotate.py),
text = lines in reading order. A new immutable version is committed to data/gt_history (with the
PAGE XML for line-level training), data/ground_truth/<id>.txt becomes that version, and the
manifest row gets ground_truth_path, gt_status and its split (= planned_split of the source).
Reports the correction effort: CER between the latest machine draft and the corrected text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.gt_store import GTStore  # noqa: E402
from benchmark.manifest import read_manifest, validate, write_manifest  # noqa: E402
from benchmark.pagexml import parse_page_xml  # noqa: E402
from benchmark.sources import load_sources  # noqa: E402
from evaluation.metrics import cer  # noqa: E402
from text.arabic_normalization import normalize_for_evaluation  # noqa: E402

DATA = ROOT / "data"
HUMAN_STATUSES = ["gold_verified", "silver_proofread", "silver_unproofread", "silver_partial"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xml", nargs="+")
    p.add_argument("--status", required=True, choices=HUMAN_STATUSES)
    p.add_argument("--author", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--data", default=str(DATA))
    args = p.parse_args(argv)

    data = Path(args.data)
    manifest = data / "benchmark_manifest.csv"
    rows = {r["id"]: r for r in read_manifest(manifest)}
    planned = {s["source_id"]: s.get("planned_split") or s.get("split") for s in load_sources(data / "sources.yaml")}
    store = GTStore(data)
    for path in map(Path, args.xml):
        xml = path.read_text(encoding="utf-8")
        doc = parse_page_xml(xml)
        page_id = Path(doc.image_filename).stem or path.stem
        if page_id not in rows:
            print(f"SKIP {path}: page id {page_id!r} not in manifest", file=sys.stderr)
            continue
        if not doc.text.strip():
            print(f"SKIP {path}: no text", file=sys.stderr)
            continue
        drafts = [v for v in store.history(page_id) if v["status"] == "draft_unverified"]
        entry = store.commit(page_id, doc.text, args.status, author=args.author, source=f"annotation:{path.name}",
                             note=args.note, page_xml=xml)
        row = rows[page_id]
        row["ground_truth_path"] = f"ground_truth/{page_id}.txt"
        row["gt_status"] = args.status
        if row["split"] == "unlabeled" and planned.get(row["source_id"]) in ("train", "val", "test"):
            row["split"] = planned[row["source_id"]]
        effort = ""
        if drafts:
            draft_text = store.text(page_id, drafts[-1]["version"])
            effort = f", draft->final CER {cer(normalize_for_evaluation(doc.text, 'nfc'), normalize_for_evaluation(draft_text, 'nfc')):.3f}"
        print(f"{page_id}: {entry['version']} {args.status}, {entry['n_lines']} lines, split={row['split']}{effort}")

    errors = validate(list(rows.values()), manifest, check_files=False)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    write_manifest(list(rows.values()), manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
