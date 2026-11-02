"""Versioned ground truth: every change is a new immutable version, nothing is overwritten.

Layout::

    data/gt_history/<page_id>/history.jsonl   one JSON line per version (append-only)
    data/gt_history/<page_id>/v001.txt        page text of version 1 (lines as on the page)
    data/gt_history/<page_id>/v002.txt        ...
    data/gt_history/<page_id>/v002.page.xml   optional line-level annotation (PAGE XML)
    data/ground_truth/<page_id>.txt           copy of the *current* version (what evaluation reads)

A version is *promoted* to current unless it is a machine draft (``draft_unverified``):
drafts are kept for provenance (e.g. to measure correction effort draft -> gold) but never
become ground truth. Every history entry records parent version, status, author, source,
timestamp, sha256 and CER against the parent.

    python -m benchmark.gt_store init                      # import existing GT as v001 (idempotent)
    python -m benchmark.gt_store log <page_id>
    python -m benchmark.gt_store diff <page_id> v001 v002
    python -m benchmark.gt_store commit <page_id> --text-file corrected.txt --status gold_verified \
        --author "Name" --note "proofread against scan"
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark.manifest import GT_STATUSES, read_manifest, write_manifest
from evaluation.metrics import cer
from text.arabic_normalization import normalize_for_evaluation

DRAFT_STATUSES = {"draft_unverified"}


class GTStore:
    def __init__(self, data_dir: str | Path):
        self.data = Path(data_dir)
        self.history_dir = self.data / "gt_history"
        self.current_dir = self.data / "ground_truth"

    # ----------------------------------------------------------------- read
    def history(self, page_id: str) -> list[dict]:
        path = self.history_dir / page_id / "history.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def current(self, page_id: str) -> dict | None:
        promoted = [v for v in self.history(page_id) if v["promoted"]]
        return promoted[-1] if promoted else None

    def text(self, page_id: str, version: str) -> str:
        return (self.history_dir / page_id / f"{version}.txt").read_text(encoding="utf-8")

    def page_xml(self, page_id: str, version: str) -> str | None:
        path = self.history_dir / page_id / f"{version}.page.xml"
        return path.read_text(encoding="utf-8") if path.exists() else None

    # ---------------------------------------------------------------- write
    def commit(self, page_id: str, text: str, status: str, author: str, source: str, note: str = "",
               page_xml: str | None = None, promote: bool | None = None) -> dict:
        if status not in GT_STATUSES or status in ("", "none"):
            raise ValueError(f"invalid GT status {status!r}")
        promote = status not in DRAFT_STATUSES if promote is None else promote
        if promote and status in DRAFT_STATUSES:
            raise ValueError("machine drafts cannot become ground truth")
        text = text.rstrip("\n") + "\n"
        history = self.history(page_id)
        version = f"v{len(history) + 1:03d}"
        folder = self.history_dir / page_id
        folder.mkdir(parents=True, exist_ok=True)
        txt_path = folder / f"{version}.txt"
        if txt_path.exists():  # never overwrite, even if history.jsonl was damaged
            raise FileExistsError(txt_path)
        txt_path.write_text(text, encoding="utf-8")
        if page_xml is not None:
            (folder / f"{version}.page.xml").write_text(page_xml, encoding="utf-8")
        parent = history[-1] if history else None
        entry = {
            "version": version,
            "parent": parent["version"] if parent else None,
            "status": status,
            "promoted": promote,
            "author": author,
            "source": source,
            "note": note,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "n_lines": sum(1 for line in text.splitlines() if line.strip()),
            "has_page_xml": page_xml is not None,
            "cer_vs_parent": (cer(normalize_for_evaluation(self.text(page_id, parent["version"]), "nfc"),
                                  normalize_for_evaluation(text, "nfc")) if parent else None),
        }
        with (folder / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if promote:
            self.current_dir.mkdir(parents=True, exist_ok=True)
            (self.current_dir / f"{page_id}.txt").write_text(text, encoding="utf-8")
        return entry

    def diff(self, page_id: str, a: str, b: str) -> str:
        ta, tb = self.text(page_id, a), self.text(page_id, b)
        head = f"CER({a} -> {b}, nfc) = {cer(normalize_for_evaluation(ta, 'nfc'), normalize_for_evaluation(tb, 'nfc')):.4f}\n"
        return head + "".join(difflib.unified_diff(ta.splitlines(True), tb.splitlines(True), a, b))


def set_manifest_status(manifest: Path, page_id: str, status: str, **fields: str) -> None:
    rows = read_manifest(manifest)
    for row in rows:
        if row["id"] == page_id:
            row["gt_status"] = status
            row.update(fields)
            break
    else:
        raise KeyError(f"{page_id} not in {manifest}")
    write_manifest(rows, manifest)


def init_from_manifest(store: GTStore, manifest: Path) -> int:
    """Import existing GT files as v001 for pages that have no history yet."""
    added = 0
    for row in read_manifest(manifest):
        if not row["ground_truth_path"] or store.history(row["id"]):
            continue
        text = (manifest.parent / row["ground_truth_path"]).read_text(encoding="utf-8")
        meta_path = (manifest.parent / row["ground_truth_path"]).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        source = (f"wikisource:{meta['wikisource_title']}@rev{meta['revid']}" if "revid" in meta
                  else row.get("license", "unknown"))
        status = row["gt_status"] if row["gt_status"] not in ("", "none", "draft_unverified") else "silver_unproofread"
        store.commit(row["id"], text, status, author=meta.get("revision_user", "unknown"), source=source,
                     note="initial import of benchmark GT", promote=True)
        added += 1
    return added


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    lg = sub.add_parser("log")
    lg.add_argument("page_id")
    df = sub.add_parser("diff")
    df.add_argument("page_id")
    df.add_argument("a")
    df.add_argument("b")
    cm = sub.add_parser("commit")
    cm.add_argument("page_id")
    cm.add_argument("--text-file", required=True)
    cm.add_argument("--status", required=True, choices=sorted(GT_STATUSES - {"", "none"}))
    cm.add_argument("--author", required=True)
    cm.add_argument("--source", default="manual")
    cm.add_argument("--note", default="")
    cm.add_argument("--page-xml", default=None)
    args = p.parse_args(argv)

    store = GTStore(args.data)
    manifest = Path(args.data) / "benchmark_manifest.csv"
    if args.cmd == "init":
        print(f"imported {init_from_manifest(store, manifest)} pages as v001")
    elif args.cmd == "log":
        for v in store.history(args.page_id):
            cur = " (current)" if v == store.current(args.page_id) else ""
            print(f"{v['version']}{cur} {v['created']} {v['status']} by {v['author']} — {v['note']} "
                  f"[cer_vs_parent={v['cer_vs_parent']}]")
    elif args.cmd == "diff":
        print(store.diff(args.page_id, args.a, args.b))
    elif args.cmd == "commit":
        xml = Path(args.page_xml).read_text(encoding="utf-8") if args.page_xml else None
        entry = store.commit(args.page_id, Path(args.text_file).read_text(encoding="utf-8"), args.status,
                             args.author, args.source, args.note, page_xml=xml)
        if entry["promoted"]:
            set_manifest_status(manifest, args.page_id, args.status)
        print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
