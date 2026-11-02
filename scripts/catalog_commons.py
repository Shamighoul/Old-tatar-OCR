"""Catalogue candidate Old Tatar sources on Wikimedia Commons (metadata only, no scans downloaded).

    python scripts/catalog_commons.py                       # -> data/candidates/commons_tatar.csv
    python scripts/catalog_commons.py --category "Tatar-language books" --depth 1

For every file: title, year (from metadata or title), pages, size, licence, author, whether it
is already a benchmark source, and a rough orthography guess by year (<1920 iske imlâ,
1920–1928 yaña imlâ, >=1928 Latin/Cyrillic = out of scope).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "old-tatar-ocr-catalog/0.1 (research; https://github.com/Shamighoul/Old-tatar-OCR)"}
FIELDS = ["file", "year", "orthography_guess", "pages", "size_mb", "mime", "license", "author", "categories",
          "in_benchmark", "url"]


def api(**params) -> dict:
    params = {"format": "json", "formatversion": "2", **params}
    for attempt in range(5):
        r = requests.get(API, params=params, headers=HEADERS, timeout=60)
        if r.status_code in (429, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def members(category: str, depth: int) -> tuple[set[str], set[str]]:
    files, seen_cats, queue = set(), set(), [(category, 0)]
    while queue:
        cat, d = queue.pop()
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        cont: dict = {}
        while True:
            res = api(action="query", list="categorymembers", cmtitle=f"Category:{cat}", cmlimit=500,
                      cmtype="file|subcat", **cont)
            for m in res["query"]["categorymembers"]:
                if m["ns"] == 6:
                    files.add(m["title"][5:])
                elif m["ns"] == 14 and d < depth:
                    queue.append((m["title"][9:], d + 1))
            if "continue" not in res:
                break
            cont = res["continue"]
    return files, seen_cats


def year_of(text: str) -> int | None:
    years = [int(y) for y in re.findall(r"\b(1[6-9]\d\d)\b", text or "")]
    return min(years) if years else None


def orthography(year: int | None) -> str:
    if year is None:
        return "unknown"
    if year < 1920:
        return "iske_imla"
    if year <= 1928:
        return "yana_imla"
    return "out_of_scope (latin/cyrillic)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", action="append", default=None)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--out", default=str(ROOT / "data" / "candidates" / "commons_tatar.csv"))
    args = p.parse_args()
    categories = args.category or ["Tatar-language books"]

    used = {s.get("commons_file") for s in
            yaml.safe_load((ROOT / "data" / "sources.yaml").read_text(encoding="utf-8"))["sources"]}
    files: set[str] = set()
    for cat in categories:
        f, cats = members(cat, args.depth)
        files |= f
        print(f"{cat}: {len(f)} files in {len(cats)} categories", file=sys.stderr)

    rows = []
    names = sorted(files)
    for i in range(0, len(names), 40):
        res = api(action="query", titles="|".join("File:" + n for n in names[i:i + 40]), prop="imageinfo|categories",
                  iiprop="size|mime|url|extmetadata", cllimit=500)
        for page in res["query"]["pages"]:
            ii = (page.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata", {})
            val = lambda k: re.sub(r"<[^>]+>", "", str(em.get(k, {}).get("value", ""))).strip()  # noqa: E731
            name = page["title"][5:]
            year = year_of(val("DateTimeOriginal")) or year_of(name) or year_of(val("ImageDescription"))
            rows.append({
                "file": name, "year": year or "", "orthography_guess": orthography(year),
                "pages": ii.get("pagecount", 1 if ii.get("mime", "").startswith("image/") else ""),
                "size_mb": round(ii.get("size", 0) / 1e6, 1), "mime": ii.get("mime", ""),
                "license": val("LicenseShortName"), "author": val("Artist")[:80],
                "categories": "; ".join(c["title"][9:] for c in page.get("categories", []))[:200],
                "in_benchmark": name in used, "url": ii.get("descriptionurl", ""),
            })
    rows.sort(key=lambda r: (r["year"] == "", r["year"] or 0, r["file"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["orthography_guess"]] = by.get(r["orthography_guess"], 0) + 1
    print(f"{len(rows)} files -> {out}; by orthography guess: {by}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
