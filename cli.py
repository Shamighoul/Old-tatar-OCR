"""Command-line entry point.

    python cli.py models                                   # backends + availability + licence
    python cli.py ocr page.jpg --model churro              # OCR one image -> JSON
    python cli.py ocr page.jpg --model paddle --preprocess mobile --lines --out result.json
    python cli.py benchmark --model paddle --split test    # = python -m evaluation.evaluate ...
    python cli.py report                                   # compare all runs in results/

Privacy: images are processed in memory and never copied or stored by this tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))


def cmd_models(args: argparse.Namespace) -> int:
    from ocr.registry import list_backends

    rows = list_backends(include_hidden=args.all)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for r in rows:
        mark = "OK " if r["available"] else "-- "
        print(f"{mark}{r['name']:<22} commercial={r['commercial_use']:<5} {r['license']}")
        print(f"     {r['description']}")
        if not r["available"]:
            print(f"     unavailable: {r['reason']}")
    return 0


def cmd_ocr(args: argparse.Namespace) -> int:
    import yaml

    from ocr.registry import availability, create_backend
    from pipeline import run_ocr
    from preprocessing import load_preprocess_config

    ok, reason = availability(args.model)
    if not ok:
        print(json.dumps({"model": args.model, "error": f"backend unavailable: {reason}"}, ensure_ascii=False))
        return 2
    overrides = yaml.safe_load(Path(args.backend_config).read_text()) if args.backend_config else {}
    backend = create_backend(args.model, **overrides)
    result = run_ocr(args.image, backend, load_preprocess_config(args.preprocess), include_lines=args.lines)
    text = json.dumps(result, ensure_ascii=False, indent=2, default=list)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("models", help="list OCR backends")
    m.add_argument("--json", action="store_true")
    m.add_argument("--all", action="store_true", help="include test-only backends")

    o = sub.add_parser("ocr", help="recognise one image")
    o.add_argument("image")
    o.add_argument("--model", default="kraken-ppocrv6")
    o.add_argument("--preprocess", default="original", help="original | scan | mobile | path/to.yaml")
    o.add_argument("--backend-config", default=None)
    o.add_argument("--lines", action="store_true", help="include per-line text/boxes/confidence")
    o.add_argument("--out", default=None, help="also write the JSON here")

    sub.add_parser("benchmark", help="run evaluation (args passed to evaluation.evaluate)", add_help=False)
    sub.add_parser("report", help="aggregate results/ (args passed to evaluation.report)", add_help=False)

    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "benchmark":
        from evaluation.evaluate import main as evaluate_main
        return evaluate_main(argv[1:])
    if argv and argv[0] == "report":
        from evaluation.report import main as report_main
        return report_main(argv[1:])

    args = parser.parse_args(argv)
    return {"models": cmd_models, "ocr": cmd_ocr}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
