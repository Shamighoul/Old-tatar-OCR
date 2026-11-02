"""Side-by-side examples for the baseline report: GT vs every model on chosen pages.

    python scripts/make_examples.py --pages ws_tatar_maktabe_1915_p0004 ws_rus_maktaplare_1925_p0006 \
        --lines 6 --out docs/examples

Uses the latest run of each model in results/ (same selection as evaluation.report) and writes
<out>/examples.md plus a downscaled page preview (<out>/<id>.jpg) for each page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from benchmark.manifest import read_manifest, resolve  # noqa: E402
from evaluation.evaluate import load_predictions  # noqa: E402
from evaluation.metrics import cer  # noqa: E402
from evaluation.report import latest_runs  # noqa: E402
from text.arabic_normalization import normalize_for_evaluation  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pages", nargs="+", required=True)
    p.add_argument("--lines", type=int, default=6)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--manifest", default=str(ROOT / "data" / "benchmark_manifest.csv"))
    p.add_argument("--out", default=str(ROOT / "docs" / "examples"))
    p.add_argument("--preprocess", default="original")
    args = p.parse_args()

    manifest = Path(args.manifest)
    rows = {r["id"]: r for r in read_manifest(manifest)}
    runs = [r for r in latest_runs(Path(args.results)) if r.name.endswith(args.preprocess) or args.preprocess in r.name]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Примеры распознавания (первые строки страницы)", "",
          "Текст показан как есть (`raw_ocr`), CER страницы — по профилю `standard`. "
          "Направление строк — справа налево.", ""]
    for pid in args.pages:
        row = rows[pid]
        img = Image.open(resolve(manifest, row["image_path"])).convert("RGB")
        img.thumbnail((900, 1300))
        img.save(out / f"{pid}.jpg", quality=80)
        gt = resolve(manifest, row["ground_truth_path"]).read_text(encoding="utf-8")
        md += [f"## {pid} ({row['script_style']}, {row['approx_year']})", "", f"![{pid}]({pid}.jpg)", "",
               "**GT (Wikisource, " + row["gt_status"] + "):**", "", "```text",
               *[l for l in gt.splitlines() if l.strip()][: args.lines], "```", ""]
        scored = []
        for run in runs:
            pred = load_predictions(run / "predictions.jsonl").get(pid)
            if not pred or pred.get("error"):
                continue
            score = cer(normalize_for_evaluation(gt), normalize_for_evaluation(pred["text"]))
            scored.append((score, run.parent.name, pred["text"]))
        for score, model, text in sorted(scored):
            md += [f"**{model}** — normalized CER {100 * score:.1f}%", "", "```text",
                   *[l for l in text.splitlines() if l.strip()][: args.lines], "```", ""]
    (out / "examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out / 'examples.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
