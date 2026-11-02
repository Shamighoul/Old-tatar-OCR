"""Registry of OCR backends.

Each entry maps a CLI name to an adapter class (imported lazily) plus default config.
Unavailable backends (missing package, binary, credentials, unacceptable licence,
insufficient hardware) stay registered and report *why* instead of breaking the system.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from ocr.base import OCRBackend


@dataclass(frozen=True)
class BackendSpec:
    name: str
    target: str  # "module:Class"
    description: str
    license: str
    commercial_use: str  # "yes" | "no" | "check" | "n/a"
    defaults: dict[str, Any] = field(default_factory=dict)
    hidden: bool = False  # test-only backends are not part of comparisons


_SPECS = [
    BackendSpec("dummy", "ocr.dummy_backend:DummyBackend",
                "Test backend returning fixed text (no model)", "n/a", "n/a", hidden=True),
    BackendSpec("paddle", "ocr.paddle_backend:PaddleBackend",
                "PaddleOCR 3.x PP-OCRv5 det + arabic_PP-OCRv5_mobile_rec (Arabic/Persian/Urdu/Uyghur...)",
                "Apache-2.0", "yes", {"lang": "ar"}),
    BackendSpec("easyocr", "ocr.easyocr_backend:EasyOCRBackend",
                "EasyOCR CRAFT det + Arabic-script CRNN (ar, fa)", "Apache-2.0", "yes",
                {"langs": ["ar", "fa"]}),
    BackendSpec("tesseract", "ocr.tesseract_backend:TesseractBackend",
                "Tesseract 5 LSTM, ara+fas traineddata (system binary)", "Apache-2.0", "yes",
                {"lang": "ara+fas"}),
    BackendSpec("kraken-arabic-script", "ocr.kraken_backend:KrakenBackend",
                "Kraken blla + OpenITI printed Arabic-script base model (ar/fa/ur/ota print)",
                "CC0-1.0 (model), Apache-2.0 (kraken)", "yes", {"model": "openiti-arabic-script"}),
    BackendSpec("kraken-ottoman", "ocr.kraken_backend:KrakenBackend",
                "Kraken blla + OpenITI printed Ottoman Turkish model", "CC0-1.0 (model)", "yes",
                {"model": "openiti-ottoman"}),
    BackendSpec("kraken-persian", "ocr.kraken_backend:KrakenBackend",
                "Kraken blla + OpenITI printed Persian model", "CC0-1.0 (model)", "yes",
                {"model": "openiti-persian"}),
    BackendSpec("kraken-arabic", "ocr.kraken_backend:KrakenBackend",
                "Kraken blla + OpenITI printed Arabic model", "CC0-1.0 (model)", "yes",
                {"model": "openiti-arabic"}),
    BackendSpec("kraken-ppocrv6", "ocr.kraken_backend:KrakenBackend",
                "Kraken blla + PP-OCRv6 (medium) multilingual print+handwriting line recognizer",
                "Apache-2.0", "yes", {"model": "ppocrv6-medium"}),
    BackendSpec("kraken-custom", "ocr.kraken_backend:KrakenBackend",
                "Kraken blla + a locally fine-tuned recognizer (pass model_path via --backend-config)",
                "depends on base model and training data", "check", hidden=True),
    BackendSpec("arabic-htr-phoenix", "ocr.arabic_htr_backend:PhoenixBackend",
                "Kraken blla + Phoenix CNN-BiLSTM-CTC Arabic manuscript HTR (handwriting)",
                "CC-BY-NC-SA-2.0", "no", {"model": "phoenix"}),
    BackendSpec("churro", "ocr.churro_backend:ChurroBackend",
                "CHURRO-3B historical VLM (Qwen2.5-VL-3B), GGUF Q4_K_M via llama.cpp (CPU)",
                "qwen-research (weights), Apache-2.0 (code)", "no", {"engine": "gguf", "quant": "Q4_K_M", "mmproj": "Q8_0"}),
    BackendSpec("churro-hf", "ocr.churro_backend:ChurroBackend",
                "CHURRO-3B reference transformers inference (bf16, needs >= ~10 GB RAM/VRAM)",
                "qwen-research (weights)", "no", {"engine": "hf"}),
    BackendSpec("ottoman-azra", "ocr.ottoman_backend:AzraBackend",
                "Azra-1-Mini 0.8B Ottoman Turkish line VLM (Qwen3.5) + kraken line segmentation",
                "Apache-2.0", "yes"),
    BackendSpec("transkribus-ottoman", "ocr.ottoman_backend:TranskribusBackend",
                "Transkribus OttomanTurkish_generic (PyLaia, model 56496) via metagrapho API",
                "proprietary service (credits)", "check"),
]

SPECS: dict[str, BackendSpec] = {s.name: s for s in _SPECS}


def _resolve(spec: BackendSpec) -> type[OCRBackend] | None:
    module_name, _, cls_name = spec.target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def availability(name: str) -> tuple[bool, str]:
    spec = SPECS[name]
    try:
        cls = _resolve(spec)
    except Exception as exc:  # broken optional import inside the adapter module
        return False, f"adapter import failed: {exc!r}"
    return cls.check_available()


def list_backends(include_hidden: bool = False) -> list[dict[str, Any]]:
    rows = []
    for spec in _SPECS:
        if spec.hidden and not include_hidden:
            continue
        ok, reason = availability(spec.name)
        rows.append({"name": spec.name, "available": ok, "reason": reason,
                     "license": spec.license, "commercial_use": spec.commercial_use,
                     "description": spec.description})
    return rows


def create_backend(name: str, **overrides: Any) -> OCRBackend:
    if name not in SPECS:
        raise KeyError(f"unknown backend {name!r}; known: {', '.join(sorted(SPECS))}")
    spec = SPECS[name]
    cls = _resolve(spec)
    backend = cls(**{**spec.defaults, **overrides})
    backend.name = name
    return backend
