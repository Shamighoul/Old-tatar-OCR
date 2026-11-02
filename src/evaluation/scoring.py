"""Turn (ground truth, prediction) pairs into the metric tables stored in metrics.json.

Metric names
------------
* ``cer`` / ``wer`` — profile ``nfc`` (NFC + whitespace only): the headline numbers.
* ``cer_normalized`` / ``wer_normalized`` — profile ``standard`` (Yeh/Kaf/Heh folding,
  presentation forms, Tatweel, zero-width, digits, punctuation variants).
* ``cer_loose`` — profile ``loose`` (+ no diacritics, bare alef, no punctuation).
* ``cer_arabic_only`` — profile ``arabic_only`` (only Arabic-script letters): fair for
  Arabic-only recognizers on mixed Arabic/Cyrillic pages.
* ``bow_f1`` — order-independent word F1 on ``standard`` (recognition vs reading order).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from evaluation.metrics import ErrorCounts, bag_of_words, char_confusions, char_counts, word_counts
from text.arabic_normalization import normalize_for_evaluation

PROFILES = ("nfc", "standard", "loose", "arabic_only")
GROUP_KEYS = ("type", "source_id", "script_style", "subset")


def page_scores(gt: str, hyp: str) -> dict:
    out: dict = {}
    for profile in PROFILES:
        r, h = normalize_for_evaluation(gt, profile), normalize_for_evaluation(hyp, profile)
        out[profile] = {"char": char_counts(r, h), "word": word_counts(r, h)}
    r, h = normalize_for_evaluation(gt, "standard"), normalize_for_evaluation(hyp, "standard")
    out["bow"] = bag_of_words(r, h)
    return out


def _summarize(pages: list[dict]) -> dict:
    tot = {p: {"char": sum((x["scores"][p]["char"] for x in pages), ErrorCounts()),
               "word": sum((x["scores"][p]["word"] for x in pages), ErrorCounts())} for p in PROFILES}
    ref_w = sum(x["scores"]["bow"]["ref_words"] for x in pages)
    hyp_w = sum(x["scores"]["bow"]["hyp_words"] for x in pages)
    overlap = sum(x["scores"]["bow"]["overlap"] for x in pages)
    prec = overlap / hyp_w if hyp_w else 0.0
    rec = overlap / ref_w if ref_w else 0.0
    return {
        "pages": len(pages),
        "documents": len({x["row"]["source_id"] for x in pages}),
        "ref_chars": tot["nfc"]["char"].ref_len,
        "cer": tot["nfc"]["char"].rate,
        "wer": tot["nfc"]["word"].rate,
        "cer_normalized": tot["standard"]["char"].rate,
        "wer_normalized": tot["standard"]["word"].rate,
        "cer_loose": tot["loose"]["char"].rate,
        "cer_arabic_only": tot["arabic_only"]["char"].rate,
        "bow_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "counts": {p: {k: v.to_dict() for k, v in tot[p].items()} for p in PROFILES},
    }


def score_run(rows: list[dict], gts: dict[str, str], preds: dict[str, dict], top_k: int = 40) -> dict:
    """``rows``: manifest rows with GT; ``preds``: id -> prediction record."""
    pages = []
    confusions: Counter = Counter()
    for row in rows:
        pred = preds.get(row["id"])
        hyp = (pred or {}).get("text", "") if pred and not pred.get("error") else ""
        gt = gts[row["id"]]
        scores = page_scores(gt, hyp)
        pages.append({"row": row, "scores": scores, "pred": pred})
        confusions.update(char_confusions(normalize_for_evaluation(gt, "standard"),
                                          normalize_for_evaluation(hyp, "standard")))

    result = {"overall": _summarize(pages) if pages else {}}
    for key in GROUP_KEYS:
        groups: dict[str, list] = defaultdict(list)
        for p in pages:
            groups[p["row"].get(key, "")].append(p)
        result[f"by_{key}"] = {g: {k: v for k, v in _summarize(ps).items() if k != "counts"}
                               for g, ps in sorted(groups.items())}

    times = [p["pred"]["processing_time"] for p in pages if p["pred"] and not p["pred"].get("error")]
    result["runtime"] = {"per_page_mean_s": mean(times) if times else None,
                         "per_page_median_s": median(times) if times else None,
                         "total_recognition_s": sum(times)}
    result["failures"] = [p["row"]["id"] for p in pages if not p["pred"] or p["pred"].get("error")]
    result["confusions_top"] = [[r, h, n] for (r, h), n in confusions.most_common(top_k)]
    result["pages"] = [{
        "id": p["row"]["id"], "source_id": p["row"]["source_id"], "subset": p["row"].get("subset", ""),
        "cer": p["scores"]["nfc"]["char"].rate, "wer": p["scores"]["nfc"]["word"].rate,
        "cer_normalized": p["scores"]["standard"]["char"].rate,
        "cer_arabic_only": p["scores"]["arabic_only"]["char"].rate,
        "bow_f1": p["scores"]["bow"]["f1"],
        "ref_chars": p["scores"]["nfc"]["char"].ref_len, "hyp_chars": p["scores"]["nfc"]["char"].hyp_len,
        "confidence": (p["pred"] or {}).get("confidence"),
        "processing_time": (p["pred"] or {}).get("processing_time"),
    } for p in pages]
    return result


def summary_view(m: dict) -> dict:
    """Small dict with the headline numbers (used for console output and reports)."""
    o = m.get("overall", {})
    return {k: o.get(k) for k in ("pages", "documents", "cer", "wer", "cer_normalized",
                                  "wer_normalized", "cer_loose", "cer_arabic_only", "bow_f1")}
