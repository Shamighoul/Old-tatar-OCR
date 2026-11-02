"""Synthetic Arabic-script pages for *pipeline smoke tests* and optional pretraining.

Lines are rendered right-to-left with proper Arabic shaping (PIL + libraqm) and written as
PAGE XML (polygon + baseline + text), so they flow through exactly the same training code as
real annotations. Labels are correct by construction. Synthetic data may only ever be used in
train — never in validation/test (ТЗ §13).
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from benchmark.pagexml import to_page_xml
from ocr.base import OCRLine

DEFAULT_FONT = "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Regular.ttf"


def render_page(lines: list[str], font_path: str = DEFAULT_FONT, font_size: int = 40, width: int = 1400,
                margin: int = 60, line_gap: int = 28, noise: bool = True, seed: int = 0) -> tuple[Image.Image, list[OCRLine]]:
    rng = random.Random(seed)
    font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.RAQM)
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    boxes = [probe.textbbox((0, 0), t, font=font, direction="rtl") for t in lines]
    line_h = max(b[3] - b[1] for b in boxes) if boxes else font_size
    height = margin * 2 + len(lines) * (line_h + line_gap)
    page = Image.new("L", (width, height), 245 if noise else 255)
    draw = ImageDraw.Draw(page)
    out_lines = []
    y = margin
    for text, (bx0, by0, bx1, by1) in zip(lines, boxes):
        w = bx1 - bx0
        x_right = width - margin - (rng.randint(0, 20) if noise else 0)
        x = x_right - w - bx0
        draw.text((x, y - by0), text, font=font, fill=rng.randint(0, 40) if noise else 0, direction="rtl")
        x0, x1 = x + bx0, x + bx1
        top, bottom = y - 4, y + (by1 - by0) + 4
        baseline_y = y - by0 + int(font.getmetrics()[0] * 0.95) - 1
        out_lines.append(OCRLine(text=text, bbox=(x0, top, x1, bottom),
                                 polygon=[(x0, top), (x1, top), (x1, bottom), (x0, bottom)],
                                 baseline=[(x0, baseline_y), (x1, baseline_y)]))
        y += line_h + line_gap
    if noise:
        page = page.filter(ImageFilter.GaussianBlur(0.6))
    return page.convert("RGB"), out_lines


def make_synthetic_set(texts: list[str], out: Path, lines_per_page: int = 12, seed: int = 0,
                       font_path: str = DEFAULT_FONT) -> list[Path]:
    """Render ``texts`` (one line each) into pages: <out>/synth_NNNN.{png,xml}."""
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    xmls = []
    for i in range(0, len(texts), lines_per_page):
        chunk = texts[i:i + lines_per_page]
        image, lines = render_page(chunk, font_path=font_path, font_size=rng.choice([34, 38, 42]), seed=seed + i)
        name = f"synth_{i // lines_per_page:04d}"
        image.save(out / f"{name}.png")
        xml = to_page_xml(f"{name}.png", image.width, image.height, lines,
                          creator="old-tatar-ocr synthetic (Amiri, raqm)", comments="SYNTHETIC — train only")
        (out / f"{name}.xml").write_text(xml, encoding="utf-8")
        xmls.append(out / f"{name}.xml")
    return xmls
