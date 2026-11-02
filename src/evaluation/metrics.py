"""OCR metrics: CER, WER, exact line accuracy, bag-of-words F1, confusion pairs.

Conventions
-----------
* Units are Unicode code points *after* the chosen normalization profile
  (``text.arabic_normalization``); words are whitespace-separated tokens.
* Page/corpus rates are micro-averaged: ``sum(edits) / sum(reference length)``.
  A rate can exceed 1.0 when the hypothesis has many insertions (e.g. VLM
  hallucinations / repetition loops) — this is reported as is, never clipped.
* Empty reference: rate is 0.0 if the hypothesis is empty, else 1.0 per page
  (and contributes its insertions to the corpus numerator with 0 denominator).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
from typing import Iterable, Sequence

try:
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein
except ImportError:  # pragma: no cover - rapidfuzz is a core dependency
    _RFLevenshtein = None


def _levenshtein_python(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance (unit costs) over strings or lists of tokens."""
    if _RFLevenshtein is not None:
        return _RFLevenshtein.distance(a, b)
    return _levenshtein_python(a, b)


@dataclass
class ErrorCounts:
    edits: int = 0
    ref_len: int = 0
    hyp_len: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def rate(self) -> float:
        if self.ref_len == 0:
            return 0.0 if self.edits == 0 else 1.0
        return self.edits / self.ref_len

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def to_dict(self) -> dict:
        return {**asdict(self), "rate": self.rate}


def _counts(ref: Sequence, hyp: Sequence) -> ErrorCounts:
    if _RFLevenshtein is not None:
        ops = _RFLevenshtein.editops(ref, hyp)
        kinds = Counter(op.tag for op in ops)
        return ErrorCounts(
            edits=len(ops), ref_len=len(ref), hyp_len=len(hyp),
            substitutions=kinds["replace"], insertions=kinds["insert"], deletions=kinds["delete"],
        )
    return ErrorCounts(edits=_levenshtein_python(ref, hyp), ref_len=len(ref), hyp_len=len(hyp))


def char_counts(ref: str, hyp: str) -> ErrorCounts:
    return _counts(ref, hyp)


def word_counts(ref: str, hyp: str) -> ErrorCounts:
    return _counts(ref.split(), hyp.split())


def cer(ref: str, hyp: str) -> float:
    return char_counts(ref, hyp).rate


def wer(ref: str, hyp: str) -> float:
    return word_counts(ref, hyp).rate


def corpus_rate(counts: Iterable[ErrorCounts]) -> float:
    total = sum(counts, ErrorCounts())
    return total.rate


def exact_line_accuracy(ref_lines: Sequence[str], hyp_lines: Sequence[str]) -> float | None:
    """Share of reference lines reproduced exactly, position by position.

    Only meaningful with line-level ground truth whose lines correspond 1:1 to the
    recognised lines (same segmentation). Returns None when no reference lines exist.
    """
    if not ref_lines:
        return None
    hits = sum(1 for i, r in enumerate(ref_lines) if i < len(hyp_lines) and hyp_lines[i] == r)
    return hits / len(ref_lines)


def bag_of_words(ref: str, hyp: str) -> dict:
    """Order-independent word overlap (precision/recall/F1).

    Separates *recognition* quality from *reading-order* errors: a model that reads
    every word correctly but in the wrong line order gets high BoW-F1 but high WER.
    """
    r, h = Counter(ref.split()), Counter(hyp.split())
    overlap = sum((r & h).values())
    precision = overlap / sum(h.values()) if h else (1.0 if not r else 0.0)
    recall = overlap / sum(r.values()) if r else (1.0 if not h else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "overlap": overlap,
            "ref_words": sum(r.values()), "hyp_words": sum(h.values())}


def char_confusions(ref: str, hyp: str) -> Counter:
    """Counter of (reference_char, hypothesis_char) edit pairs.

    Substitutions are ``(r, h)``, deletions ``(r, "")``, insertions ``("", h)``.
    """
    pairs: Counter = Counter()
    if _RFLevenshtein is None:  # pragma: no cover
        return pairs
    for op in _RFLevenshtein.editops(ref, hyp):
        if op.tag == "replace":
            pairs[(ref[op.src_pos], hyp[op.dest_pos])] += 1
        elif op.tag == "delete":
            pairs[(ref[op.src_pos], "")] += 1
        elif op.tag == "insert":
            pairs[("", hyp[op.dest_pos])] += 1
    return pairs
