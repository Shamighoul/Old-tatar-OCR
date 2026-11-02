"""Kraken adapter: baseline line segmentation (blla) + CTC line recognizer.

Recognition models are pinned Zenodo/HF files (``MODEL_CATALOG``) fetched through
``ocr.model_store``. Segmentation uses kraken's default BLLA model unless a path is
given. Lines are segmented with ``text_direction='horizontal-rl'`` so that reading
order is right-to-left/top-to-bottom, and kraken's BiDi reordering returns text in
logical order.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

from PIL import Image

from ocr.base import OCRBackend, OCRLine, OCRResult
from ocr.model_store import ModelFile, fetch

_Z = "https://zenodo.org/api/records/{rec}/files/{name}/content"

MODEL_CATALOG: dict[str, ModelFile] = {
    m.key: m
    for m in [
        ModelFile("openiti-arabic-script", _Z.format(rec=7050270, name="all_arabic_scripts.mlmodel"),
                  "all_arabic_scripts.mlmodel", "md5:f4a6488f8db357c641217015ff7ab06b", 16277053,
                  "CC0-1.0", "doi:10.5281/zenodo.7050270"),
        ModelFile("openiti-ottoman", _Z.format(rec=7050342, name="ottoman_best.mlmodel"),
                  "ottoman_best.mlmodel", "md5:3c0db8cfab67fd709c7e6b49c237591d", 16282182,
                  "CC0-1.0", "doi:10.5281/zenodo.7050342"),
        ModelFile("openiti-persian", _Z.format(rec=7051644, name="persian_best.mlmodel"),
                  "persian_best.mlmodel", "md5:1c322f2defb5f2f96f95c15cc68a5169", 16280254,
                  "CC0-1.0", "doi:10.5281/zenodo.7051644"),
        ModelFile("openiti-arabic", _Z.format(rec=7050296, name="arabic_best.mlmodel"),
                  "arabic_best.mlmodel", "md5:e92ce0a8f79f933678d172c5e3928d34", 16279935,
                  "CC0-1.0", "doi:10.5281/zenodo.7050296"),
        ModelFile("ppocrv6-medium", _Z.format(rec=21788410, name="medium.safetensors"),
                  "medium.safetensors", "md5:e3411a453ce3b9e9efae3f8631d85762", 63779644,
                  "Apache-2.0", "doi:10.5281/zenodo.21788410"),
        ModelFile("ppocrv6-small", _Z.format(rec=21788405, name="small.safetensors"),
                  "small.safetensors", "md5:07279cd68cd4402807542799f12d0f4a", 13065380,
                  "Apache-2.0", "doi:10.5281/zenodo.21788405"),
        ModelFile("phoenix",
                  "https://huggingface.co/factlogic/phoenix-arabic-manuscript-htr/resolve/"
                  "aa6fadf8eb66fa9d566bc56b50592e28ef85185a/model.mlmodel",
                  "model.mlmodel",
                  "sha256:2896fef9d9665cbb82fba8faa3bf0c628ac6a5cc62eb678f40c707db833aebea", None,
                  "CC-BY-NC-SA-2.0", "hf:factlogic/phoenix-arabic-manuscript-htr@aa6fadf"),
    ]
}


class KrakenBackend(OCRBackend):
    name = "kraken"

    @classmethod
    def check_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("kraken") is None:
            return False, "package 'kraken' not installed (pip install -e '.[kraken]')"
        return True, "ok"

    def _load(self) -> None:
        import torch
        from kraken.configs import RecognitionInferenceConfig
        from kraken.tasks import RecognitionTaskModel

        from layout.kraken_layout import KrakenLineSegmenter

        torch.set_num_threads(int(os.environ.get("OLD_TATAR_OCR_THREADS", os.cpu_count() or 1)))

        if self.config.get("model_path"):  # locally fine-tuned model (Phase 3)
            rec_path = Path(self.config["model_path"]).expanduser()
            rec_id = f"{rec_path.name}@{hashlib.sha256(rec_path.read_bytes()).hexdigest()[:12]}"
            self._license = self.config.get("license", "inherits base model + training data licences")
        else:
            model_key = self.config["model"]
            model_file = MODEL_CATALOG[model_key]
            rec_path = fetch(model_file)
            rec_id = f"{model_key}@{model_file.checksum.split(':')[1][:12]} [{model_file.source}]"
            self._license = model_file.license
        self._segmenter = KrakenLineSegmenter(self.config.get("segmentation_model"),
                                              self.config.get("text_direction", "horizontal-rl"))
        self._rec = RecognitionTaskModel.load_model(rec_path)
        self._rec_cfg = RecognitionInferenceConfig(padding=self.config.get("padding", 16),
                                                   bidi_reordering=self.config.get("bidi_reordering", True),
                                                   num_line_workers=self.config.get("num_line_workers", 2))
        self.model_version = f"{rec_id} + seg {self._segmenter.version}"

    def _recognize(self, image: Image.Image) -> OCRResult:
        seg = self._segmenter.segment(image)
        lines: list[OCRLine] = []
        for rec in self._rec.predict(image, seg, self._rec_cfg):
            boundary = [tuple(map(int, p)) for p in (getattr(rec, "boundary", None) or [])]
            bbox = None
            if boundary:
                xs, ys = zip(*boundary)
                bbox = (min(xs), min(ys), max(xs), max(ys))
            confs = list(rec.confidences or [])
            lines.append(OCRLine(
                text=rec.prediction,
                confidence=sum(confs) / len(confs) if confs else None,
                bbox=bbox, polygon=boundary or None,
                baseline=[tuple(map(int, p)) for p in (getattr(rec, "baseline", None) or [])] or None,
            ))
        return OCRResult(
            text="\n".join(l.text for l in lines),
            model_name=self.name, model_version=self.model_version, lines=lines,
            metadata={"n_lines": len(lines), "license": self._license},
        )
