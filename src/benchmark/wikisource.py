"""Convert Wikisource ProofreadPage wikitext into plain ground-truth text.

Conversion rules (version ``CONVERSION_VERSION``; documented in data/README.md):

* ``<noinclude>…</noinclude>`` (page quality header/footer) and HTML comments removed.
* Templates, innermost first:
  ``{{ditto}}`` -> ``〃`` (U+3003, the printed ditto mark);
  ``{{?|X}}`` (transcriber unsure) -> ``X``, recorded in ``uncertain_spans``;
  ``{{SIC|as printed|correction}}`` -> *as printed* (OCR GT must match the page);
  ``{{lang|code|X}}`` -> ``X``;
  ``{{tt/икетелле сүзлек|a|b|…}}`` (two-column word list) -> one cell per line, in order;
  formatting wrappers (``{{c|X}}``, ``{{uc|X}}``, ``{{font-size|120%|X}}`` …) -> ``X``;
  ``{{=}}`` -> ``=``; ``{{*}}`` -> ``•``;
  ``{{rule}}``, ``{{nop}}``, ``{{missing image}}`` -> nothing;
  any other template is dropped and listed in ``unknown_templates``.
* Wiki tables (``{| … |}``): one output line per table row, cells joined by a space
  in source order; table/row markup and cell attributes removed.
* HTML entities decoded (``&zwnj;`` -> U+200C ZWNJ, ``&mdash;`` -> —).
* ``<br>`` and block tags (div/p/h1-6/li/ol/ul/center/poem) -> line breaks; other tags
  stripped keeping their content.
* Wiki list markers (``*``, ``#``, ``:``) at line start, bold/italic quotes and
  ``[[link|label]]`` markup removed (they are wiki formatting, not printed text).
* Lines are stripped; runs of blank lines collapsed to one.

The text itself (letters, ZWNJ, punctuation, digits, Cyrillic passages) is preserved.
"""

from __future__ import annotations

import html
import re

CONVERSION_VERSION = "2"

_NOINCLUDE = re.compile(r"<noinclude>.*?</noinclude>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")  # innermost template (no braces inside)
_DROP_TEMPLATES = {"rule", "nop", "clear", "gap", "missing image"}
_LIST_TEMPLATES = {"tt/икетелле сүзлек"}
_WRAPPER_TEMPLATES = {"c", "center", "left", "right", "uc", "lc", "sc", "font-size", "larger", "smaller",
                      "small", "big", "x-larger", "x-smaller", "bold", "italic", "nowrap"}
_SYMBOL_TEMPLATES = {"=": "=", "*": "•", "!": "|"}
_CELL_ATTR = re.compile(r"^\s*[a-z-]+\s*=\s*\"[^\"]*\"(?:\s+[a-z-]+\s*=\s*\"[^\"]*\")*\s*\|(?!\|)", re.I)


def _convert_tables(s: str) -> str:
    out: list[str] = []
    row: list[str] = []
    depth = 0
    for line in s.split("\n"):
        stripped = line.strip()
        if stripped.startswith("{|"):
            depth += 1
            continue
        if depth and stripped.startswith("|}"):
            depth -= 1
            if row:
                out.append(" ".join(row))
                row = []
            continue
        if depth and stripped.startswith("|-"):
            if row:
                out.append(" ".join(row))
                row = []
            continue
        if depth and stripped[:1] in ("|", "!"):
            for cell in re.split(r"\|\||!!", stripped[1:]):
                cell = _CELL_ATTR.sub("", cell).strip()
                if cell:
                    row.append(cell)
            continue
        if depth and row and stripped:
            row[-1] += " " + stripped  # continuation of a multi-line cell
            continue
        out.append(line)
    if row:
        out.append(" ".join(row))
    return "\n".join(out)
_BR = re.compile(r"<br\s*/?>", re.I)
_BLOCK = re.compile(r"</?(?:div|p|h[1-6]|li|ol|ul|center|poem|blockquote|table|tr)\b[^>]*>", re.I)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_LIST_MARK = re.compile(r"^[*#:;]+\s*", re.M)
_BOLD_ITALIC = re.compile(r"'{2,}")


def wikitext_to_text(wikitext: str) -> tuple[str, dict]:
    info = {"uncertain_spans": [], "unknown_templates": [], "ditto_marks": 0,
            "conversion_version": CONVERSION_VERSION}
    s = _NOINCLUDE.sub("", wikitext)
    s = _COMMENT.sub("", s)

    def _template(m: re.Match) -> str:
        name, *args = [a.strip() for a in m.group(1).split("|")]
        key = name.lower()
        if key == "ditto":
            info["ditto_marks"] += 1
            return "〃"
        if key == "?":
            span = args[0] if args else "?"
            info["uncertain_spans"].append(span)
            return span
        if key == "sic":
            return args[0] if args else ""
        if key == "lang":
            return args[-1] if args else ""
        if key in _LIST_TEMPLATES:
            return "\n" + "\n".join(args) + "\n"
        if key in _WRAPPER_TEMPLATES:
            return args[-1] if args else ""
        if key in _SYMBOL_TEMPLATES and not args:
            return _SYMBOL_TEMPLATES[key]
        if key in _DROP_TEMPLATES:
            return ""
        info["unknown_templates"].append(m.group(1).strip()[:200])
        return ""

    while _TEMPLATE.search(s):
        s = _TEMPLATE.sub(_template, s)

    s = _convert_tables(s)
    s = _BR.sub("\n", s)
    s = _BLOCK.sub("\n", s)
    s = _TAG.sub("", s)
    s = _LINK.sub(r"\1", s)
    s = _LIST_MARK.sub("", s)
    s = _BOLD_ITALIC.sub("", s)
    s = html.unescape(s)

    lines = [line.strip() for line in s.replace("\r", "").split("\n")]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip(), info


def script_profile(text: str) -> dict[str, float | int]:
    """Letter counts by script; ``arabic_share`` drives the core/mixed subset split."""
    arabic = len(re.findall(r"[ؠ-يٮ-ەݐ-ݿ]", text))
    cyrillic = len(re.findall(r"[Ѐ-ӿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = arabic + cyrillic + latin
    return {"arabic": arabic, "cyrillic": cyrillic, "latin": latin,
            "arabic_share": arabic / total if total else 0.0}
