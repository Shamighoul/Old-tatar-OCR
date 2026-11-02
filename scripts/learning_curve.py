"""Learning curve: how much verified Old Tatar annotation is needed? (baseline question 5)

Fine-tunes the same base model on nested, source-stratified subsets of the verified train pages
(e.g. 5, 10, 25, 50, all), evaluates each on the same test pages and tabulates CER vs. data size.
The zero-data point is the untouched base model (Phase 1 baseline).

    python scripts/learning_curve.py --base ppocrv6-medium --sizes 5 10 25 50 --evaluate core
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.dataset import eligible_pages  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="ppocrv6-medium")
    p.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 25, 50])
    p.add_argument("--evaluate", choices=["core", "all"], default="core")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train = [x for x in eligible_pages(ROOT / "data") if x["split"] == "train"]
    sizes = sorted({min(n, len(train)) for n in args.sizes if n > 0} | ({len(train)} if train else set()))
    if not train:
        print("No verified train pages yet — nothing to learn from.", file=sys.stderr)
        return 2
    rows = []
    for n in sizes:
        name = f"lc-{args.base}-{n:04d}p-s{args.seed}"
        subprocess.run([sys.executable, "scripts/finetune_kraken.py", "--name", name, "--base", args.base,
                        "--train-pages", str(n), "--seed", str(args.seed), "--evaluate", args.evaluate],
                       cwd=ROOT, check=False)
        runs = sorted((ROOT / "results" / "kraken-custom").glob(f"*-{name}-original-{args.evaluate}/metrics.json"))
        if runs:
            o = json.loads(runs[-1].read_text())["overall"]
            rows.append((n, o["cer"], o["cer_normalized"], o["wer_normalized"]))
    print("| train pages | CER % | norm. CER % | norm. WER % |\n|---|---|---|---|")
    for n, c, cn, wn in rows:
        print(f"| {n} | {100 * c:.1f} | {100 * cn:.1f} | {100 * wn:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
