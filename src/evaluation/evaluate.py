"""Run one OCR backend on a benchmark split and score it.

    python -m evaluation.evaluate --model paddle --manifest data/benchmark_manifest.csv --split test
    python -m evaluation.evaluate --model churro --subset core --preprocess scan
    python -m evaluation.evaluate --resume results/churro/20261102-153000      # continue a run
    python -m evaluation.evaluate --rescore results/paddle/20261102-153000     # re-score saved predictions

Writes results/<model>/<timestamp>[-<tag>]/:
    predictions.jsonl  one JSON line per page (raw OCR text, lines, confidence, timing)
    metrics.json       overall / per type / per source_id / per script_style / per subset,
                       runtime, memory, top character confusions, per-page scores
    config.yaml        exact command, backend config, preprocessing, git, hardware,
                       library versions, dataset hash, seed
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

from benchmark.manifest import read_manifest, resolve
from evaluation import repro
from evaluation.scoring import score_run, summary_view
from ocr.base import BackendUnavailable
from ocr.registry import SPECS, availability, create_backend
from preprocessing import load_preprocess_config, preprocess


def select_rows(rows: list[dict], split: str, subset: str | None, doc_type: str | None,
                ids: list[str] | None, limit: int | None) -> list[dict]:
    out = [r for r in rows if r["ground_truth_path"] and (split == "all" or r["split"] == split)]
    if subset:
        out = [r for r in out if r.get("subset") == subset]
    if doc_type:
        out = [r for r in out if r["type"] == doc_type]
    if ids:
        out = [r for r in out if r["id"] in set(ids)]
    return out[:limit] if limit else out


def load_predictions(path: Path) -> dict[str, dict]:
    preds = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                preds[rec["id"]] = rec
    return preds


def format_summary(model: str, metrics: dict) -> str:
    s = summary_view(metrics)
    rt, mem = metrics.get("runtime", {}), metrics.get("memory", {})

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.2f}%"

    def sec(x):
        return "n/a" if x is None else f"{x:.1f}s"

    lines = [
        f"model: {model}",
        f"documents: {s['documents']}",
        f"pages: {s['pages']}",
        f"CER: {pct(s['cer'])}",
        f"WER: {pct(s['wer'])}",
        f"normalized CER: {pct(s['cer_normalized'])}   normalized WER: {pct(s['wer_normalized'])}",
        f"loose CER: {pct(s['cer_loose'])}   arabic-only CER: {pct(s['cer_arabic_only'])}   BoW-F1: {pct(s['bow_f1'])}",
        f"runtime/page: {sec(rt.get('per_page_mean_s'))} (median {sec(rt.get('per_page_median_s'))}), "
        f"model load: {sec(rt.get('load_s'))}",
        f"GPU memory: {'n/a (CPU run)' if mem.get('gpu_peak_mb') is None else str(mem['gpu_peak_mb']) + ' MB'}"
        f"   peak RSS: {mem.get('peak_rss_mb')} MB",
    ]
    for key in ("by_type", "by_source_id"):
        for group, g in metrics.get(key, {}).items():
            lines.append(f"  {key[3:]}={group}: CER {pct(g['cer'])}, normalized CER {pct(g['cer_normalized'])}, "
                         f"arabic-only CER {pct(g['cer_arabic_only'])} ({g['pages']} pages)")
    if metrics.get("failures"):
        lines.append(f"failed pages: {len(metrics['failures'])}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    out_dir: Path
    if args.resume or args.rescore:
        out_dir = Path(args.resume or args.rescore)
        cfg = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
        for k, v in cfg["args"].items():
            if k not in ("resume", "rescore"):
                setattr(args, k, v)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.output_dir) / args.model / (stamp + (f"-{args.tag}" if args.tag else ""))

    manifest = Path(args.manifest)
    rows = select_rows(read_manifest(manifest), args.split, args.subset, args.type, args.ids, args.limit)
    if not rows:
        print("no labelled rows selected", file=sys.stderr)
        return 1
    gt_paths = {r["id"]: resolve(manifest, r["ground_truth_path"]) for r in rows}
    gts = {rid: p.read_text(encoding="utf-8") for rid, p in gt_paths.items()}
    pred_path = out_dir / "predictions.jsonl"
    backend_config = yaml.safe_load(Path(args.backend_config).read_text()) if args.backend_config else {}

    load_s = None
    model_version = None
    if not args.rescore:
        ok, reason = availability(args.model)
        out_dir.mkdir(parents=True, exist_ok=True)
        pre_cfg = load_preprocess_config(args.preprocess)
        config_record = {
            "args": {k: v for k, v in vars(args).items()},
            "command": " ".join(sys.argv),
            "backend": {"name": args.model, "spec": SPECS[args.model].__dict__, "config": backend_config},
            "preprocessing": pre_cfg.to_dict(),
            "dataset": {"manifest": str(manifest), "version": repro.dataset_version(manifest, list(gt_paths.values())),
                        "n_pages": len(rows), "ids": [r["id"] for r in rows]},
            "git": repro.git_info(), "hardware": repro.hardware_info(), "libraries": repro.library_versions(),
            "seed": args.seed,
        }
        if not ok:
            (out_dir / "config.yaml").write_text(yaml.safe_dump(config_record, allow_unicode=True, sort_keys=False))
            metrics = {"model": args.model, "status": "unavailable", "reason": reason}
            (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
            print(f"model: {args.model}\nstatus: UNAVAILABLE — {reason}")
            return 2

        repro.set_seed(args.seed)
        backend = create_backend(args.model, **backend_config)
        try:
            backend.load()
        except BackendUnavailable as exc:
            print(f"model: {args.model}\nstatus: UNAVAILABLE — {exc}")
            return 2
        load_s = backend.load_time
        model_version = backend.model_version
        config_record["backend"]["model_version"] = model_version
        config_record["backend"]["load_time_s"] = load_s
        (out_dir / "config.yaml").write_text(yaml.safe_dump(config_record, allow_unicode=True, sort_keys=False))

        done = load_predictions(pred_path)
        with pred_path.open("a", encoding="utf-8") as f:
            for i, row in enumerate(rows, 1):
                if row["id"] in done and not done[row["id"]].get("error"):
                    continue
                record = {"id": row["id"], "model": args.model, "model_version": model_version}
                try:
                    original = Image.open(resolve(manifest, row["image_path"]))
                    pre = preprocess(original, pre_cfg)
                    result = backend.recognize(pre.image)
                    record.update({"text": result.text, "confidence": result.confidence,
                                   "processing_time": result.processing_time,
                                   "lines": [l.__dict__ for l in result.lines] if args.save_lines else None,
                                   "preprocessing": pre.steps, "metadata": result.metadata})
                except Exception as exc:  # keep going; failures are scored as empty output
                    record.update({"text": "", "error": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc(limit=5)})
                f.write(json.dumps(record, ensure_ascii=False, default=list) + "\n")
                f.flush()
                status = record.get("error") or f"{record['processing_time']:.1f}s"
                print(f"[{i}/{len(rows)}] {row['id']}: {status}", file=sys.stderr, flush=True)
    else:
        cfg = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
        load_s = cfg["backend"].get("load_time_s")
        model_version = cfg["backend"].get("model_version")

    preds = load_predictions(pred_path)
    metrics = score_run(rows, gts, preds)
    metrics = {
        "model": args.model, "model_version": model_version,
        "status": "ok" if not metrics["failures"] else "partial",
        "run": {"split": args.split, "subset": args.subset, "type": args.type, "preprocess": args.preprocess or "original",
                "timestamp": datetime.now().isoformat(timespec="seconds"), "output_dir": str(out_dir)},
        **metrics,
    }
    metrics["runtime"]["load_s"] = load_s
    if not args.rescore:
        metrics["memory"] = {"peak_rss_mb": repro.peak_rss_mb(), "gpu_peak_mb": repro.gpu_peak_mb()}
    else:
        old = json.loads((out_dir / "metrics.json").read_text()) if (out_dir / "metrics.json").exists() else {}
        metrics["memory"] = old.get("memory", {})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(format_summary(args.model, metrics))
    print(f"results: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help=f"backend name: {', '.join(n for n, s in SPECS.items() if not s.hidden)}")
    p.add_argument("--manifest", default="data/benchmark_manifest.csv")
    p.add_argument("--split", default="test", help="train|val|test|all")
    p.add_argument("--subset", default=None, help="manifest 'subset' filter, e.g. core|mixed")
    p.add_argument("--type", default=None, choices=["printed", "handwritten"])
    p.add_argument("--ids", nargs="*", default=None, help="only these page ids")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--preprocess", default="original", help="original | scan | mobile | path/to.yaml")
    p.add_argument("--backend-config", default=None, help="YAML with backend overrides")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--tag", default="", help="suffix for the run directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-lines", action="store_true", help="store line boxes/text in predictions")
    p.add_argument("--resume", default=None, help="continue an interrupted run directory")
    p.add_argument("--rescore", default=None, help="only recompute metrics for a run directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.model or args.resume or args.rescore):
        build_parser().error("--model is required (or --resume/--rescore)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
