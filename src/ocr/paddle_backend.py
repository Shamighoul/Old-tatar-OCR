"""PaddleOCR 3.x adapter (PP-OCRv5 detection + Arabic-script recognition).

``lang="ar"`` selects ``arabic_PP-OCRv5_mobile_rec`` (Arabic, Persian, Uyghur, Urdu,
Pashto, Kurdish, Sindhi, Balochi). Detected boxes are merged into visual rows and read
right-to-left (``layout.reading_order``).

``reverse_text`` exists because Arabic CTC recognizers sometimes emit characters in
visual (left-to-right) order; it is a fixed, documented setting (checked against
ground truth once), never guessed per page.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
from PIL import Image

from layout.reading_order import merge_rows_rtl
from ocr.base import OCRBackend, OCRLine, OCRResult


class PaddleBackend(OCRBackend):
    name = "paddle"

    @classmethod
    def check_available(cls) -> tuple[bool, str]:
        for pkg in ("paddle", "paddleocr"):
            if importlib.util.find_spec(pkg) is None:
                return False, f"package '{pkg}' not installed (pip install -e '.[paddle]')"
        return True, "ok"

    def _load(self) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        import paddleocr
        from paddleocr import PaddleOCR

        kwargs = dict(
            lang=self.config.get("lang", "ar"),
            ocr_version=self.config.get("ocr_version", "PP-OCRv5"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self.config.get("device", "cpu"),
            # paddle 3.3 + oneDNN fails on PP-OCRv5 PIR programs on CPU
            # ("ConvertPirAttribute2RuntimeAttribute not support"), so oneDNN is off by default.
            enable_mkldnn=self.config.get("enable_mkldnn", False),
            cpu_threads=int(os.environ.get("OLD_TATAR_OCR_THREADS", os.cpu_count() or 1)),
        )
        # Library defaults are used unless overridden (fair out-of-the-box baseline).
        if self.config.get("det_model"):
            kwargs["text_detection_model_name"] = self.config["det_model"]
        if self.config.get("det_limit_side_len"):
            kwargs["text_det_limit_side_len"] = self.config["det_limit_side_len"]
            kwargs["text_det_limit_type"] = self.config.get("det_limit_type", "max")
        self._ocr = PaddleOCR(**kwargs)
        self.model_version = (f"paddleocr {paddleocr.__version__} lang={kwargs['lang']} "
                              f"{kwargs['ocr_version']} det={self.config.get('det_model', 'default')}")

    def _recognize(self, image: Image.Image) -> OCRResult:
        bgr = np.asarray(image)[:, :, ::-1].copy()
        res = list(self._ocr.predict(bgr))[0]
        texts, scores, boxes = res["rec_texts"], res["rec_scores"], res["rec_boxes"]
        reverse = self.config.get("reverse_text", False)
        raw = [OCRLine(text=t[::-1] if reverse else t, confidence=float(s),
                       bbox=tuple(int(v) for v in b))
               for t, s, b in zip(texts, scores, boxes)]
        lines = merge_rows_rtl(raw)
        return OCRResult(
            text="\n".join(l.text for l in lines),
            model_name=self.name, model_version=self.model_version, lines=lines,
            metadata={"n_boxes": len(raw), "n_rows": len(lines),
                      "det_model": res.get("model_settings", {}).get("text_det_model", None)
                      if isinstance(res.get("model_settings"), dict) else None},
        )
