"""Arabic-script text normalization.

Two strictly separated concerns:

1. ``source_preserving_text`` — the only cleanup ever applied to stored OCR output
   (``raw_ocr``). It is transport-level only (line endings, BOM, outer whitespace) and
   never replaces one letter by another.

2. ``normalize_for_evaluation`` — lossy, profile-based folding used *only* inside metric
   computation, so that models are not penalised for encoding choices that are
   visually indistinguishable or merely conventional (Persian vs Arabic Yeh, NFD vs NFC,
   Tatweel, ZWNJ, presentation forms ...). The result is never written back to
   predictions.

Every rule has a stable ID; profiles are ordered lists of rule IDs. The rule table is
the single source of truth for docs (``describe_profiles()``) and is unit-tested rule
by rule in ``tests/test_arabic_normalization.py``.

Old-Tatar-specific caveats (documented in docs/architecture.md):

* Yaña imlâ (1920–1927 reformed Arabic script) uses ``ى`` (U+0649), ``ە`` (U+06D5) and
  ``ھ`` (U+06BE) as distinct letters. The ``standard`` profile folds them (Y1, H1),
  which hides real orthographic distinctions; use the ``nfc`` profile to see them.
* ``ڭ``/``ݣ`` (ң), ``گ`` (г), ``ۋ`` (в), ``ۇ``/``ۆ``/``ۈ`` are real Tatar letters and are
  never folded into Kaf / Waw.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

__all__ = [
    "PROFILES",
    "RULES",
    "describe_profiles",
    "is_arabic_letter",
    "normalize_for_evaluation",
    "source_preserving_text",
]


# --------------------------------------------------------------------------- source

def source_preserving_text(text: str | None) -> str:
    """Lossless-in-content cleanup allowed on stored OCR output.

    Only: CRLF/CR -> LF, drop a leading BOM, strip outer whitespace.
    Letters, marks, ZWNJ, Tatweel, digits and punctuation are kept as produced.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("﻿"):
        text = text[1:]
    return text.strip()


# --------------------------------------------------------------------------- helpers

def _map(table: dict[str, str]) -> Callable[[str], str]:
    trans = str.maketrans(table)
    return lambda s: s.translate(trans)


def _remove(chars: str) -> Callable[[str], str]:
    trans = str.maketrans("", "", chars)
    return lambda s: s.translate(trans)


_ARABIC_PRESENTATION = re.compile("[ﭐ-﷿ﹰ-﻿]")


def _fold_presentation_forms(s: str) -> str:
    # NFKC restricted to Arabic Presentation Forms-A/B so that compatibility characters
    # of other scripts are untouched. U+FEFF (BOM) lives in the range but has no
    # decomposition; zero-width removal handles it.
    out = _ARABIC_PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), s)
    return unicodedata.normalize("NFC", out)


_ARABIC_LETTER_RANGES = (
    (0x0620, 0x064A),  # Arabic letters
    (0x066E, 0x06D3),  # extended letters (Persian/Urdu/Turkic additions)
    (0x06D5, 0x06D5),  # AE (Tatar/Kazakh ә in reformed scripts)
    (0x06EE, 0x06EF),
    (0x06FA, 0x06FC),
    (0x06FF, 0x06FF),
    (0x0750, 0x077F),  # Arabic Supplement (e.g. U+0763 kaf with three dots)
    (0x08A0, 0x08C9),  # Arabic Extended-A letters
)


def is_arabic_letter(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_LETTER_RANGES)


def _arabic_only(s: str) -> str:
    # Keep Arabic-script letters (+ their combining marks) and spaces; everything else
    # (Cyrillic, Latin, digits, punctuation) becomes a space.
    kept = []
    for ch in s:
        if is_arabic_letter(ch) or (kept and unicodedata.category(ch) == "Mn" and kept[-1] != " "):
            kept.append(ch)
        else:
            kept.append(" ")
    return "".join(kept)


_WS = re.compile(r"\s+")

# Diacritics (harakat, tanwin, shadda, sukun, superscript alef, Quranic annotation marks).
_ARABIC_DIACRITICS = (
    "".join(chr(c) for c in range(0x064B, 0x0660))
    + "ٰ"
    + "".join(chr(c) for c in range(0x06D6, 0x06DD))
    + "".join(chr(c) for c in range(0x06DF, 0x06E5))
    + "ۧۨ"
    + "".join(chr(c) for c in range(0x06EA, 0x06EE))
)

_ZERO_WIDTH_AND_BIDI = (
    "​‌‍‎‏؜­﻿"
    "‪‫‬‭‮⁦⁧⁨⁩"
)


# --------------------------------------------------------------------------- rules

@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    apply: Callable[[str], str]


RULES: dict[str, Rule] = {
    r.id: r
    for r in [
        Rule("N1", "Unicode NFC (canonical composition; e.g. kraken emits NFD: ا+ٓ -> آ)",
             lambda s: unicodedata.normalize("NFC", s)),
        Rule("P1", "Arabic presentation forms (U+FB50–FDFF, U+FE70–FEFF) -> base letters (NFKC on those code points only)",
             _fold_presentation_forms),
        Rule("Z1", "Remove zero-width and bidi control characters: ZWSP, ZWNJ, ZWJ, LRM, RLM, ALM, soft hyphen, BOM, embeddings/isolates",
             _remove(_ZERO_WIDTH_AND_BIDI)),
        Rule("T1", "Remove Tatweel / Kashida (U+0640)", _remove("ـ")),
        Rule("Y1", "Yeh variants -> Arabic Yeh U+064A: Farsi Yeh U+06CC, Alef Maksura U+0649, Yeh Barree U+06D2",
             _map({"ی": "ي", "ى": "ي", "ے": "ي"})),
        Rule("K1", "Kaf variants -> Arabic Kaf U+0643: Keheh U+06A9, Swash Kaf U+06AA",
             _map({"ک": "ك", "ڪ": "ك"})),
        Rule("K2", "Ng variants -> U+06AD (ڭ): Kaf with three dots above U+0763 (ݣ), Ngoeh U+06B1",
             _map({"ݣ": "ڭ", "ڱ": "ڭ"})),
        Rule("H1", "Heh variants -> Arabic Heh U+0647: Heh Goal U+06C1, Heh Doachashmee U+06BE, AE U+06D5, Teh Marbuta Goal U+06C3",
             _map({"ہ": "ه", "ھ": "ه", "ە": "ه", "ۃ": "ة"})),
        Rule("D1", "Arabic-Indic (U+0660–0669) and Extended/Persian (U+06F0–06F9) digits -> ASCII 0–9",
             _map({**{chr(0x0660 + i): str(i) for i in range(10)},
                   **{chr(0x06F0 + i): str(i) for i in range(10)}})),
        Rule("Q1", "Punctuation variants: ، -> ,  ؛ -> ;  ؟ -> ?  ۔ -> .  and quotes/ditto marks « » “ ” „ ″ 〃 -> \"",
             _map({"،": ",", "؛": ";", "؟": "?", "۔": ".",
                   "«": '"', "»": '"', "“": '"', "”": '"',
                   "„": '"', "″": '"', "〃": '"'})),
        Rule("S1", "Collapse all whitespace (incl. newlines, NBSP) to one space; strip",
             lambda s: _WS.sub(" ", s).strip()),
        Rule("DI1", "Remove Arabic diacritics: harakat/tanwin/shadda/sukun U+064B–065F, superscript alef U+0670, Quranic marks",
             _remove(_ARABIC_DIACRITICS)),
        Rule("A1", "Alef variants -> bare Alef U+0627: آ أ إ ٱ (U+0622, U+0623, U+0625, U+0671)",
             _map({"آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا"})),
        Rule("PU1", "Remove all punctuation and symbols (Unicode categories P*, S*)",
             lambda s: "".join(ch for ch in s if unicodedata.category(ch)[0] not in "PS")),
        Rule("AR1", "Keep only Arabic-script letters (+ attached marks); everything else -> space",
             _arabic_only),
    ]
}

# Order matters: canonical forms first, folding next, whitespace last.
PROFILES: dict[str, tuple[str, ...]] = {
    # Exact comparison of the stored strings.
    "raw": (),
    # Minimal, encoding-only: the basis of the headline "CER".
    "nfc": ("N1", "S1"),
    # "Normalized CER": folds visually identical / conventional variants.
    "standard": ("N1", "P1", "Z1", "T1", "Y1", "K1", "K2", "H1", "D1", "Q1", "N1", "S1"),
    # Additionally ignores diacritics, hamza-on-alef and punctuation.
    "loose": ("N1", "P1", "Z1", "T1", "Y1", "K1", "K2", "H1", "D1", "Q1",
              "DI1", "A1", "PU1", "N1", "S1"),
    # Standard folding restricted to Arabic-script letters (mixed Arabic/Cyrillic pages).
    "arabic_only": ("N1", "P1", "Z1", "T1", "Y1", "K1", "K2", "H1", "AR1", "N1", "S1"),
}


def normalize_for_evaluation(text: str | None, profile: str = "standard") -> str:
    """Apply the named evaluation profile. Never use the result as stored OCR output."""
    if profile not in PROFILES:
        raise ValueError(f"unknown normalization profile {profile!r}; choose from {sorted(PROFILES)}")
    s = text or ""
    for rule_id in PROFILES[profile]:
        s = RULES[rule_id].apply(s)
    return s


def describe_profiles() -> str:
    """Markdown description of all rules and profiles (used in docs)."""
    lines = ["| Rule | Description |", "|---|---|"]
    lines += [f"| {r.id} | {r.description} |" for r in RULES.values()]
    lines += ["", "| Profile | Rules (in order) |", "|---|---|"]
    lines += [f"| {name} | {' → '.join(rules) or '(none)'} |" for name, rules in PROFILES.items()]
    return "\n".join(lines)
