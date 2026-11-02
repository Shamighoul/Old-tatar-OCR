"""Build the real-document Old Tatar benchmark from Wikimedia Commons + Wikisource.

For every source in data/sources.yaml (kind: wikisource) this script
  1. lists the transcribed ProofreadPage pages (Arabic script, and the parallel
     modern Cyrillic ``/tt-Cyrl/N`` pages when present),
  2. pins each page to a revision id in data/wikisource_lock.json (reproducible GT),
  3. converts wikitext -> plain text (benchmark.wikisource, rules in data/README.md),
  4. downloads the page image rendered by Commons from the public-domain scan,
  5. commits GT through the versioned store (data/gt_history/, current copy in
     data/ground_truth/<id>.txt; human-corrected versions are never overwritten), writes
     .modern_tatar.txt / .json metadata and upserts rows into data/benchmark_manifest.csv.

No labels are invented: GT status is taken from the Wikisource quality level
(1 = not proofread -> silver_unproofread, 3 = proofread -> silver_proofread,
4 = validated -> gold_verified).

    python scripts/fetch_wikisource_benchmark.py            # use pinned revisions
    python scripts/fetch_wikisource_benchmark.py --update-lock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.commons import WS_API, api, commons_page_image_url, download  # noqa: E402
from benchmark.gt_store import GTStore  # noqa: E402
from benchmark.manifest import read_manifest, write_manifest  # noqa: E402
from benchmark.wikisource import script_profile, wikitext_to_text  # noqa: E402

DATA = ROOT / "data"
PAGE_NS = 104
QUALITY_TO_STATUS = {1: "silver_unproofread", 2: "silver_unproofread", 3: "silver_proofread", 4: "gold_verified"}
CORE_ARABIC_SHARE = 0.8


def list_pages(index: str) -> list[dict]:
    pages, cont = [], {}
    while True:
        d = api(WS_API, action="query", generator="allpages", gapnamespace=PAGE_NS,
                gapprefix=index + "/", gaplimit=500, prop="info|proofread", **cont)
        pages += d.get("query", {}).get("pages", [])
        if "continue" not in d:
            return pages
        cont = d["continue"]


def fetch_revisions(revids: list[int]) -> dict[int, dict]:
    out = {}
    for i in range(0, len(revids), 50):
        d = api(WS_API, action="query", prop="revisions|proofread", revids="|".join(map(str, revids[i:i + 50])),
                rvprop="ids|content|user|timestamp", rvslots="main")
        for p in d["query"]["pages"]:
            for rev in p.get("revisions", []):
                out[rev["revid"]] = {"title": p["title"], "content": rev["slots"]["main"]["content"],
                                     "user": rev["user"], "timestamp": rev["timestamp"],
                                     "quality": p.get("proofread", {}).get("quality")}
    return out


def page_number(title: str) -> tuple[int | None, bool]:
    tail = title.rsplit("/", 2)
    is_cyrl = len(tail) == 3 and tail[1] == "tt-Cyrl"
    try:
        return int(tail[-1]), is_cyrl
    except ValueError:
        return None, is_cyrl


def build_source(src: dict, lock: dict, update_lock: bool, width: int) -> list[dict]:
    store = GTStore(DATA)
    index = src["wikisource_index"]
    prefix = f"Page:{index}/"
    if update_lock or not any(t.startswith(prefix) for t in lock):
        for p in list_pages(index):
            lock[p["title"]] = p["lastrevid"]
    titles = [t for t in lock if t.startswith(prefix)]
    revs = fetch_revisions([lock[t] for t in titles])
    by_page: dict[int, dict] = {}
    for rev in revs.values():
        n, is_cyrl = page_number(rev["title"])
        if n is not None:
            by_page.setdefault(n, {})["cyrl" if is_cyrl else "arab"] = rev

    rows = []
    for n, variants in sorted(by_page.items()):
        rev = variants.get("arab")
        if rev is None:
            continue
        text, conv = wikitext_to_text(rev["content"])
        profile = script_profile(text)
        if profile["arabic"] == 0:
            continue  # page without Arabic-script text (e.g. Russian-only title page)
        rid = f"{src['source_id']}_p{n:04d}"
        gt_dir = DATA / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        # GT goes through the versioned store: a Wikisource revision becomes a new version only while
        # the current GT still comes from Wikisource; human-corrected versions are never overwritten.
        source_ref = f"wikisource:{rev['title']}@rev{lock[rev['title']]}"
        wiki_status = QUALITY_TO_STATUS.get(rev["quality"], "silver_unproofread")
        current = store.current(rid)
        if current is None or (current["source"].startswith("wikisource:") and current["source"] != source_ref):
            store.commit(rid, text, wiki_status, author=rev["user"], source=source_ref,
                         note="Wikisource import" if current is None else "Wikisource revision update")
            current = store.current(rid)
        gt_status = wiki_status if current["source"].startswith("wikisource:") else current["status"]
        meta = {"id": rid, "source_id": src["source_id"], "page": n, "wikisource_title": rev["title"],
                "revid": lock[rev["title"]], "revision_timestamp": rev["timestamp"], "revision_user": rev["user"],
                "wikisource_quality": rev["quality"], "script_profile": profile, **conv,
                "gt_license": src["gt_license"]}
        if "cyrl" in variants:
            ctext, _ = wikitext_to_text(variants["cyrl"]["content"])
            (gt_dir / f"{rid}.modern_tatar.txt").write_text(ctext + "\n", encoding="utf-8")
            meta["modern_tatar"] = {"wikisource_title": variants["cyrl"]["title"],
                                    "revid": lock[variants["cyrl"]["title"]],
                                    "wikisource_quality": variants["cyrl"]["quality"]}

        image_rel = Path("raw") / src["source_id"] / f"p{n:04d}.jpg"
        image_path = DATA / image_rel
        meta_path = gt_dir / f"{rid}.json"
        previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if not image_path.exists():
            url, finfo = commons_page_image_url(src["commons_file"], n, width)
            meta["image_sha256"] = download(url, image_path)
            meta["image_url"] = url
            meta["commons_file_sha1"] = finfo.get("sha1")
            time.sleep(0.5)
        else:
            # Keep image provenance across rebuilds; look it up (no download) if it was never recorded.
            meta["image_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if previous.get("image_url"):
                meta["image_url"] = previous["image_url"]
                meta["commons_file_sha1"] = previous.get("commons_file_sha1")
            else:
                url, finfo = commons_page_image_url(src["commons_file"], n, width)
                meta["image_url"], meta["commons_file_sha1"] = url, finfo.get("sha1")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        rows.append({
            "id": rid, "image_path": str(image_rel), "source_id": src["source_id"], "page": str(n),
            "type": src["type"], "script_style": src["script_style"], "approx_year": str(src["approx_year"]),
            "ground_truth_path": f"ground_truth/{rid}.txt", "split": src["split"],
            "subset": "core" if profile["arabic_share"] >= CORE_ARABIC_SHARE else "mixed",
            "language": src["language"], "gt_status": gt_status,
            "license": f"image: {src['image_license']}; GT: {src['gt_license']}",
            "notes": f"uncertain_spans={len(conv['uncertain_spans'])}; arabic_share={profile['arabic_share']:.2f}",
        })
        print(f"  {rid}: {profile['arabic']} arabic / {profile['cyrillic']} cyrillic letters, "
              f"quality={rev['quality']}, subset={rows[-1]['subset']}")
    return rows


def build_unlabeled(src: dict, width: int) -> dict:
    rid = src["source_id"]
    image_rel = Path("raw") / rid / "image.jpg"
    if not (DATA / image_rel).exists():
        url, _ = commons_page_image_url(src["commons_file"], None, width)
        download(url, DATA / image_rel)
    return {"id": rid, "image_path": str(image_rel), "source_id": rid, "page": "1", "type": src["type"],
            "script_style": src["script_style"], "approx_year": str(src["approx_year"]),
            "ground_truth_path": "", "split": "unlabeled", "subset": "photo", "language": src["language"],
            "gt_status": "none", "license": src["image_license"], "notes": src.get("notes", "")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-lock", action="store_true", help="re-pin pages to their latest revisions")
    parser.add_argument("--width", type=int, default=2000, help="max page image width to request (px)")
    args = parser.parse_args()

    sources = yaml.safe_load((DATA / "sources.yaml").read_text(encoding="utf-8"))["sources"]
    lock_path = DATA / "wikisource_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    manifest_path = DATA / "benchmark_manifest.csv"
    existing = {r["id"]: r for r in read_manifest(manifest_path)} if manifest_path.exists() else {}

    for src in sources:
        print(f"== {src['source_id']}")
        if src["kind"] == "wikisource":
            for row in build_source(src, lock, args.update_lock, args.width):
                existing[row["id"]] = row
        elif src["kind"] == "commons_image":
            row = build_unlabeled(src, args.width)
            existing[row["id"]] = row

    overrides_path = DATA / "gt_overrides.yaml"
    overrides = (yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}).get("overrides", {}) \
        if overrides_path.exists() else {}
    for rid, ov in overrides.items():
        if rid in existing:
            existing[rid]["gt_status"] = ov["gt_status"]
            note = f"override: {ov['gt_status']} (data/gt_overrides.yaml)"
            if note not in existing[rid].get("notes", ""):
                existing[rid]["notes"] = f"{existing[rid].get('notes', '')}; {note}".strip("; ")

    lock_path.write_text(json.dumps(dict(sorted(lock.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_manifest(list(existing.values()), manifest_path)
    print(f"manifest: {len(existing)} rows -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
