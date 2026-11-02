"""Phase 3, strategy A: fine-tune a kraken recognizer on verified Old Tatar line data, then evaluate.

    # real run (needs human-verified train/val pages with PAGE XML in data/gt_history)
    python scripts/finetune_kraken.py --base ppocrv6-medium --name ppocrv6-oldtatar-v1

    # pipeline smoke test on synthetic pages (model is NOT a result; never evaluate it as one)
    python scripts/finetune_kraken.py --synthetic-smoke --max-epochs 2 --name smoke

Steps: assemble PAGE XML training set from data/gt_history (test pages excluded by construction)
-> ketos train (early stopping on val) -> copy best model to models/finetuned/<name>/model.*
-> optional benchmark evaluation with the `kraken-custom` backend (--evaluate core|all).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from benchmark.gt_store import GTStore  # noqa: E402
from training.dataset import build_training_set, eligible_pages, sample_pages  # noqa: E402
from training.kraken_finetune import FinetuneConfig, run_finetune  # noqa: E402
from training.synthetic import make_synthetic_set  # noqa: E402

DATA = ROOT / "data"


def synthetic_lists(out: Path, n_train_lines: int, n_val_lines: int) -> tuple[str, str, str]:
    """Smoke data rendered from *train-family* draft text only (never test text)."""
    store = GTStore(DATA)
    train_sources = ("commons_salima", "commons_manbit", "commons_bigiev")
    texts = []
    for pid in sorted(p.name for p in (DATA / "gt_history").iterdir() if p.name.startswith(train_sources)):
        last = store.history(pid)[-1]
        texts += [l for l in store.text(pid, last["version"]).splitlines() if len(l.strip()) > 10]
    if len(texts) < n_train_lines + n_val_lines:
        raise SystemExit(f"only {len(texts)} text lines available for synthetic smoke data")
    train = make_synthetic_set(texts[:n_train_lines], out / "train", seed=1)
    val = make_synthetic_set(texts[n_train_lines:n_train_lines + n_val_lines], out / "val", seed=2)
    (out / "train.txt").write_text("\n".join(str(p.resolve()) for p in train) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(str(p.resolve()) for p in val) + "\n", encoding="utf-8")
    return str(out / "train.txt"), str(out / "val.txt"), "synthetic-smoke"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="run name -> models/finetuned/<name>")
    p.add_argument("--base", default="ppocrv6-medium")
    p.add_argument("--train-pages", type=int, default=None, help="use a nested subset of N train pages")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--min-epochs", type=int, default=5)
    p.add_argument("--lag", type=int, default=5)
    p.add_argument("--lrate", type=float, default=1e-4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", default="AdamW", choices=["AdamW", "AdamW+Muon", "Adam", "SGD"])
    p.add_argument("--schedule", default="constant", choices=["constant", "cosine"])
    p.add_argument("--quit", default="early", choices=["early", "fixed"])
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--smoke-lines", type=int, nargs=2, default=[120, 24], metavar=("TRAIN", "VAL"))
    p.add_argument("--evaluate", choices=["none", "core", "all"], default="core")
    args = p.parse_args()

    out = ROOT / "models" / "finetuned" / args.name
    if args.synthetic_smoke:
        train_list, val_list, version = synthetic_lists(out / "data", *args.smoke_lines)
        evaluate = "none"
    else:
        pages = eligible_pages(DATA)
        train = [x for x in pages if x["split"] == "train"]
        if not train or not any(x["split"] == "val" for x in pages):
            print("No human-verified train/val pages with line-level PAGE XML yet "
                  "(see docs/dataset.md: preannotate -> correct -> import_annotations). Nothing to train on.",
                  file=sys.stderr)
            return 2
        subset = sample_pages(train, args.train_pages, args.seed) if args.train_pages else None
        ts = build_training_set(DATA, out / "data", train_pages=subset)
        train_list, val_list, version = str(ts.list_file("train")), str(ts.list_file("val")), ts.version
        evaluate = args.evaluate

    cfg = FinetuneConfig(base=args.base, train_list=train_list, val_list=val_list, out_dir=str(out),
                         max_epochs=args.max_epochs, min_epochs=min(args.min_epochs, args.max_epochs),
                         lag=args.lag, lrate=args.lrate, augment=not args.no_augment, seed=args.seed,
                         optimizer=args.optimizer, schedule=args.schedule, quit=args.quit, warmup=args.warmup)
    record = run_finetune(cfg, dataset_version=version,
                          note="SYNTHETIC SMOKE TEST — not a result" if args.synthetic_smoke else "")
    print(f"ketos exit={record['returncode']}, {record['train_seconds']} s, model={record.get('model')}")
    if record["returncode"] != 0 or "model" not in record:
        print(f"training failed, see {out / 'ketos.log'}", file=sys.stderr)
        return 1

    if evaluate != "none":
        backend_cfg = out / "backend.yaml"
        backend_cfg.write_text(yaml.safe_dump({"model_path": record["model"]}), encoding="utf-8")
        cmd = [sys.executable, "-m", "evaluation.evaluate", "--model", "kraken-custom", "--split", "test",
               "--backend-config", str(backend_cfg), "--tag", f"{args.name}-original-{evaluate}", "--save-lines"]
        if evaluate == "core":
            cmd += ["--subset", "core"]
        subprocess.run(cmd, cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
