"""Download the selected pages of `kind: commons_pages` sources (Phase 2 annotation candidates).

Pages are taken at the **native scan resolution**: the original PDF is downloaded once
(sha1 checked against Commons) to data/raw/<source_id>/source.pdf and the embedded page image
is extracted losslessly with ``pdfimages`` (largest image on the page). Pages without an
embedded raster fall back to ``pdftoppm`` at 300 dpi. (Commons thumbnails snap to fixed widths —
e.g. 500 px for these books — which is too small for annotation.)

Per page: image -> data/raw/<source_id>/pNNNN.<ext>, provenance -> data/page_meta/<id>.json,
manifest row with split=unlabeled, gt_status=none (it becomes train/val/test only after verified
ground truth is imported with scripts/import_annotations.py).

    python scripts/fetch_source_pages.py                 # all commons_pages sources
    python scripts/fetch_source_pages.py --source commons_salima_1898
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from benchmark.commons import COMMONS_API, api, download  # noqa: E402
from benchmark.manifest import read_manifest, write_manifest  # noqa: E402
from benchmark.sources import load_sources, validate_sources  # noqa: E402

DATA = ROOT / "data"


def source_pdf(src: dict) -> tuple[Path, dict]:
    info = api(COMMONS_API, action="query", titles="File:" + src["commons_file"], prop="imageinfo",
               iiprop="url|size|sha1")["query"]["pages"][0]["imageinfo"][0]
    path = DATA / "raw" / src["source_id"] / "source.pdf"
    if not (path.exists() and hashlib.sha1(path.read_bytes()).hexdigest() == info["sha1"]):
        print(f"  downloading {src['commons_file']} ({info['size'] / 1e6:.1f} MB)", flush=True)
        download(info["url"].split("?")[0], path)
        if hashlib.sha1(path.read_bytes()).hexdigest() != info["sha1"]:
            raise RuntimeError(f"sha1 mismatch for {src['commons_file']}")
    return path, info


def extract_page(pdf: Path, page: int, target_stem: Path) -> tuple[Path, str]:
    """Largest embedded image of ``page`` (native format), else a 300 dpi render."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdfimages", "-f", str(page), "-l", str(page), "-all", str(pdf), f"{tmp}/img"],
                       check=True, capture_output=True)
        images = []
        for f in Path(tmp).iterdir():
            try:
                with Image.open(f) as im:
                    images.append((im.width * im.height, f))
            except OSError:
                continue
        if images:
            best = max(images)[1]
            target = target_stem.with_suffix(best.suffix.lower().replace(".jpeg", ".jpg"))
            shutil.copyfile(best, target)
            return target, "pdfimages (embedded scan, lossless)"
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", "300", "-png", "-singlefile",
                        str(pdf), str(target_stem)], check=True, capture_output=True)
        return target_stem.with_suffix(".png"), "pdftoppm 300 dpi render"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=None)
    args = p.parse_args()

    sources = load_sources(DATA / "sources.yaml")
    errors, _ = validate_sources(sources)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    manifest = DATA / "benchmark_manifest.csv"
    rows = {r["id"]: r for r in read_manifest(manifest)}
    for src in sources:
        if src["kind"] != "commons_pages" or (args.source and src["source_id"] not in args.source):
            continue
        pdf, info = source_pdf(src)
        for n in src["pages"]:
            rid = f"{src['source_id']}_p{n:04d}"
            stem = DATA / "raw" / src["source_id"] / f"p{n:04d}"
            existing = sorted(stem.parent.glob(stem.name + ".*"))
            if existing:
                image, method = existing[0], json.loads((DATA / "page_meta" / f"{rid}.json").read_text())["method"]
            else:
                image, method = extract_page(pdf, n, stem)
            with Image.open(image) as im:
                size = im.size
            meta = {"id": rid, "source_id": src["source_id"], "page": n, "commons_file": src["commons_file"],
                    "commons_file_sha1": info["sha1"], "method": method, "image_size": list(size),
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
            (DATA / "page_meta").mkdir(exist_ok=True)
            (DATA / "page_meta" / f"{rid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                           encoding="utf-8")
            if rid in rows and rows[rid].get("ground_truth_path"):
                continue  # already annotated: keep its status/split
            rows[rid] = {
                "id": rid, "image_path": str(image.relative_to(DATA)), "source_id": src["source_id"], "page": str(n),
                "type": src["type"], "script_style": src["script_style"], "approx_year": str(src["approx_year"]),
                "ground_truth_path": "", "split": "unlabeled", "subset": "phase2_pilot",
                "language": src["language"], "gt_status": "none",
                "license": f"image: {src['image_license']}; GT: {src['gt_license']}",
                "notes": f"planned_split={src['planned_split']}; family={src.get('family', src['source_id'])}",
            }
            print(f"  {rid}: {image.relative_to(DATA)} {size[0]}x{size[1]} ({method})", flush=True)
    write_manifest(list(rows.values()), manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
