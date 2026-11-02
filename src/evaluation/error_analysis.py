"""Error analysis across benchmark runs.

    python -m evaluation.error_analysis --group core     # -> results/summary/error_analysis_core.md

Per model (latest run per model/tag, same pages for everyone):
* per-letter recall for letters that matter for Old Tatar (ڭ ۋ گ چ پ ژ ق غ ە ۇ …):
  share of occurrences in GT that are aligned to the identical letter in the output;
* error structure: substitutions / insertions / deletions (standard profile);
* most frequent confusion pairs.
Alignment = Levenshtein opcodes on ``standard``-normalized page text (``nfc`` for the
letter table, so that variants folded by ``standard`` are still visible).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from benchmark.manifest import read_manifest, resolve
from evaluation.evaluate import load_predictions
from evaluation.metrics import char_confusions, char_counts
from evaluation.report import GROUPS, latest_runs, pct
from text.arabic_normalization import normalize_for_evaluation

# Letters whose recognition matters for Tatar (Persian/Turkic additions + Tatar-specific).
KEY_LETTERS = ["ڭ", "گ", "ۋ", "چ", "پ", "ژ", "ق", "غ", "ك", "ک", "ی", "ي", "ى", "ە", "ھ", "ۇ", "ئ", "ء", "ة", "ث", "ذ", "ظ"]


def letter_recall(ref: str, hyp: str) -> tuple[Counter, Counter]:
    total, correct = Counter(ref), Counter()
    for tag, i1, i2, _j1, _j2 in Levenshtein.opcodes(ref, hyp):
        if tag == "equal":
            correct.update(ref[i1:i2])
    return total, correct


def build(results: Path, manifest: Path, group: str, preprocess: str) -> str:
    rows = [r for r in read_manifest(manifest) if r["ground_truth_path"] and r["split"] == "test" and GROUPS[group](r)]
    ids = {r["id"] for r in rows}
    gts = {r["id"]: resolve(manifest, r["ground_truth_path"]).read_text(encoding="utf-8") for r in rows}
    models = []
    for run in latest_runs(results):
        preds = load_predictions(run / "predictions.jsonl")
        if not ids <= set(preds) or not run.name.endswith(preprocess):
            continue
        total, correct, conf, counts = Counter(), Counter(), Counter(), None
        for rid in sorted(ids):
            hyp = "" if preds[rid].get("error") else preds[rid].get("text", "")
            ref_n, hyp_n = normalize_for_evaluation(gts[rid], "nfc"), normalize_for_evaluation(hyp, "nfc")
            t, c = letter_recall(ref_n, hyp_n)
            total.update(t)
            correct.update(c)
            ref_s, hyp_s = normalize_for_evaluation(gts[rid], "standard"), normalize_for_evaluation(hyp, "standard")
            conf.update(char_confusions(ref_s, hyp_s))
            cc = char_counts(ref_s, hyp_s)
            counts = cc if counts is None else counts + cc
        models.append({"model": run.parent.name, "total": total, "correct": correct, "conf": conf, "counts": counts})
    models.sort(key=lambda m: m["counts"].rate)

    md = [f"# Error analysis — pages `{group}` ({len(rows)} pages), preprocessing `{preprocess}`", "",
          "## Error structure (standard profile)", "",
          "| model | norm. CER % | substitutions | insertions | deletions | ref chars |", "|---|---|---|---|---|---|"]
    for m in models:
        c = m["counts"]
        md.append(f"| {m['model']} | {pct(c.rate)} | {c.substitutions} | {c.insertions} | {c.deletions} | {c.ref_len} |")

    letters = [l for l in KEY_LETTERS if any(m["total"][l] for m in models)]
    md += ["", "## Per-letter recall % (nfc profile; n = occurrences in GT)", "",
           "| model | " + " | ".join(f"{l} (U+{ord(l):04X})" for l in letters) + " |",
           "|---|" + "---|" * len(letters),
           "| n | " + " | ".join(str(models[0]["total"][l]) if models else "" for l in letters) + " |"]
    for m in models:
        md.append(f"| {m['model']} | " + " | ".join(
            pct(m["correct"][l] / m["total"][l]) if m["total"][l] else "—" for l in letters) + " |")

    for m in models:
        md += ["", f"### {m['model']}: top confusions (standard profile)", "",
               "| GT | output | count |", "|---|---|---|"]
        for (r, h), n in m["conf"].most_common(12):
            fmt = lambda ch: f"`{ch}` U+{ord(ch):04X}" if ch else "∅"  # noqa: E731
            md.append(f"| {fmt(r)} | {fmt(h)} | {n} |")
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--manifest", default="data/benchmark_manifest.csv")
    p.add_argument("--group", default="core", choices=list(GROUPS))
    p.add_argument("--preprocess", default="original")
    p.add_argument("--out", default="results/summary")
    args = p.parse_args(argv)
    md = build(Path(args.results), Path(args.manifest), args.group, args.preprocess)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"error_analysis_{args.group}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
