"""Helpers shared by vision-language-model backends."""

from __future__ import annotations

import base64
import math
import os
import re
import xml.etree.ElementTree as ET
from io import BytesIO

from PIL import Image


def threads() -> int:
    return int(os.environ.get("OLD_TATAR_OCR_THREADS", os.cpu_count() or 1))


def resize_to_pixel_budget(image: Image.Image, min_pixels: int, max_pixels: int, factor: int = 28) -> Image.Image:
    """Qwen-VL style resize: keep aspect, pixel count in [min, max], sides multiple of ``factor``."""
    w, h = image.size
    pixels = w * h
    scale = 1.0
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    nw = max(factor, int(w * scale) // factor * factor)
    nh = max(factor, int(h * scale) // factor * factor)
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), Image.Resampling.BICUBIC)


def image_to_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


_DROP_TAGS = ("Description", "Deletion", "Illegible", "Gap")
_ANY_TAG = re.compile(r"<[^>]+>")


def churro_xml_to_text(xml: str) -> tuple[str, bool]:
    """Extract page text from CHURRO's ``HistoricalDocument`` XML.

    Mirrors CHURRO's evaluation tooling (drop Description/Deletion/Illegible/Gap,
    keep Header/Body/Footer text). Unlike CHURRO, unparsable XML (typically output
    truncated at ``max_tokens``) falls back to tag stripping instead of an empty
    string; the returned flag records whether the XML parsed.
    """
    if "<" not in xml:
        return xml, True
    s = xml
    for tag in _DROP_TAGS:
        s = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", s, flags=re.S)
        s = re.sub(rf"<{tag}\b[^>]*/>", "", s)
    start = s.find("<HistoricalDocument")
    try:
        root = ET.fromstring(s[start:] if start >= 0 else s)
        sections = []
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] in {"Header", "Body", "Footer"}:
                lines = [t.strip() for t in node.itertext() if t.strip()]
                if lines:
                    sections.append("\n".join(lines))
        if sections:
            return "\n".join(sections), True
        return "\n".join(t.strip() for t in root.itertext() if t.strip()), True
    except ET.ParseError:
        text = _ANY_TAG.sub("\n", s)
        return "\n".join(t.strip() for t in text.splitlines() if t.strip()), False
