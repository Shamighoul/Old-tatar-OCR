"""Specialised Arabic *handwriting* recognizer.

Phoenix (arXiv:2608.19385) is a compact CNN+BiLSTM+CTC kraken model trained on Arabic
manuscript lines (Agapet, Omar, TariMa). It is run through the kraken adapter so it
shares segmentation and reading order with the other line-based backends.

Licence: CC BY-NC-SA 2.0 -> research benchmarking only, not for a commercial product.
"""

from __future__ import annotations

from ocr.kraken_backend import KrakenBackend


class PhoenixBackend(KrakenBackend):
    name = "arabic-htr-phoenix"
