"""Minimal Wikimedia API helpers shared by the data scripts (polite User-Agent, retries)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

WS_API = "https://wikisource.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "old-tatar-ocr-benchmark/0.1 (research; https://github.com/Shamighoul/Old-tatar-OCR)"}


def api(url: str, **params) -> dict:
    params = {"format": "json", "formatversion": "2", **params}
    r = None
    for attempt in range(5):
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def commons_page_image_url(file: str, page: int | None, width: int) -> tuple[str, dict]:
    """Thumbnail URL of ``page`` of a Commons PDF/DjVu (or of an image when page is None)."""
    info = api(COMMONS_API, action="query", titles="File:" + file, prop="imageinfo",
               iiprop="size|sha1|url")["query"]["pages"][0]["imageinfo"][0]
    w = min(width, info.get("width") or width)
    params = {"iiurlwidth": w}
    if page is not None:
        params["iiurlparam"] = f"page{page}-{w}px"
    thumb = api(COMMONS_API, action="query", titles="File:" + file, prop="imageinfo",
                iiprop="url", **params)["query"]["pages"][0]["imageinfo"][0]
    return thumb.get("thumburl") or thumb["url"], info


def download(url: str, target: Path) -> str:
    """Download to ``target``; return sha256 of the content."""
    target.parent.mkdir(parents=True, exist_ok=True)
    r = None
    for attempt in range(5):
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt + 1)
            continue
        r.raise_for_status()
        target.write_bytes(r.content)
        return hashlib.sha256(r.content).hexdigest()
    r.raise_for_status()
    return ""
