"""PAGE XML (2019) export/import for line-level annotation.

Why PAGE XML: it is imported/exported by eScriptorium and Transkribus, read by kraken's
trainer (``ketos train -f page``) and stores exactly what Phase 2 needs per line:
polygon, baseline, transcription (+ optional confidence) and reading order.

* ``to_page_xml`` writes machine pre-annotations (drafts) for human correction.
* ``parse_page_xml`` reads corrected files back (any PAGE namespace version; regions and
  lines are taken in ReadingOrder if present, otherwise in document order).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from ocr.base import OCRLine

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


@dataclass
class PageLine:
    id: str
    text: str
    polygon: list[tuple[int, int]]
    baseline: list[tuple[int, int]] | None = None
    confidence: float | None = None
    region_id: str = ""


@dataclass
class PageDoc:
    image_filename: str
    width: int
    height: int
    lines: list[PageLine] = field(default_factory=list)
    comments: str = ""

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


def _points(pts: list[tuple[int, int]]) -> str:
    return " ".join(f"{int(x)},{int(y)}" for x, y in pts)


def _parse_points(s: str) -> list[tuple[int, int]]:
    out = []
    for pair in s.split():
        x, y = pair.split(",")
        out.append((int(float(x)), int(float(y))))
    return out


def _line_polygon(line: OCRLine) -> list[tuple[int, int]]:
    if line.polygon:
        return [tuple(p) for p in line.polygon]
    if line.bbox:
        x0, y0, x1, y1 = line.bbox
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    raise ValueError("line has neither polygon nor bbox")


def to_page_xml(image_filename: str, width: int, height: int, lines: list[OCRLine],
                creator: str, comments: str = "") -> str:
    """One TextRegion with all lines in the given (reading) order."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    polys = [_line_polygon(l) for l in lines]
    if polys:
        xs = [x for p in polys for x, _ in p]
        ys = [y for p in polys for _, y in p]
        region = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
    else:
        region = [(0, 0), (width, 0), (width, height), (0, height)]
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<PcGts xmlns="{PAGE_NS}">',
        f"  <Metadata><Creator>{escape(creator)}</Creator><Created>{now}</Created>"
        f"<LastChange>{now}</LastChange><Comments>{escape(comments)}</Comments></Metadata>",
        f'  <Page imageFilename="{escape(image_filename)}" imageWidth="{width}" imageHeight="{height}">',
        '    <ReadingOrder><OrderedGroup id="ro1"><RegionRefIndexed index="0" regionRef="r1"/></OrderedGroup></ReadingOrder>',
        '    <TextRegion id="r1" type="paragraph">',
        f'      <Coords points="{_points(region)}"/>',
    ]
    for i, (line, poly) in enumerate(zip(lines, polys), 1):
        conf = f' conf="{line.confidence:.4f}"' if line.confidence is not None else ""
        out.append(f'      <TextLine id="r1_l{i}">')
        out.append(f'        <Coords points="{_points(poly)}"/>')
        if line.baseline:
            out.append(f'        <Baseline points="{_points(line.baseline)}"/>')
        out.append(f"        <TextEquiv{conf}><Unicode>{escape(line.text)}</Unicode></TextEquiv>")
        out.append("      </TextLine>")
    out += ["    </TextRegion>", "  </Page>", "</PcGts>", ""]
    return "\n".join(out)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in el if _local(c.tag) == name), None)


def parse_page_xml(xml: str) -> PageDoc:
    root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    page = next(el for el in root.iter() if _local(el.tag) == "Page")
    meta = next((el for el in root.iter() if _local(el.tag) == "Comments"), None)
    regions = {r.get("id"): r for r in page.iter() if _local(r.tag) == "TextRegion"}
    order = [el.get("regionRef") for el in sorted(
        (e for e in page.iter() if _local(e.tag) == "RegionRefIndexed"), key=lambda e: int(e.get("index", 0)))]
    ordered = [regions[r] for r in order if r in regions] + [r for rid, r in regions.items() if rid not in order]
    doc = PageDoc(image_filename=page.get("imageFilename", ""), width=int(page.get("imageWidth", 0)),
                  height=int(page.get("imageHeight", 0)), comments=(meta.text or "") if meta is not None else "")
    for region in ordered:
        for line in (c for c in region if _local(c.tag) == "TextLine"):
            coords, base = _child(line, "Coords"), _child(line, "Baseline")
            equivs = [c for c in line if _local(c.tag) == "TextEquiv"]
            equiv = min(equivs, key=lambda e: int(e.get("index", 0))) if equivs else None
            unicode_el = _child(equiv, "Unicode") if equiv is not None else None
            doc.lines.append(PageLine(
                id=line.get("id", ""), text=(unicode_el.text or "") if unicode_el is not None else "",
                polygon=_parse_points(coords.get("points", "")) if coords is not None else [],
                baseline=_parse_points(base.get("points", "")) if base is not None else None,
                confidence=float(equiv.get("conf")) if equiv is not None and equiv.get("conf") else None,
                region_id=region.get("id", ""),
            ))
    return doc
