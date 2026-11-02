"""Ottoman Turkish recognizers.

* ``AzraBackend`` — OttomanNLP/Azra-1-Mini-0.8b (Apache-2.0), a Qwen3.5 0.8B VLM
  fine-tuned on Ottoman Turkish *line* images (Nesih print + Rika handwriting) with
  Arabic-script targets. It is line-level, so pages are segmented with kraken BLLA
  (right-to-left reading order) and each polygon-extracted line is transcribed with
  the prompt and decoding settings from the model card.

* ``TranskribusBackend`` — the Transkribus OttomanTurkish_generic / _Print models have
  no public weights and are only reachable through the paid metagrapho API, which
  also uploads page images to a third-party server. Registered as unavailable.
  Note that OttomanTurkish_Print (model 52502) outputs a *Latin* "half-transcription",
  i.e. not usable for the Arabic-script ``raw_ocr`` target at all.
"""

from __future__ import annotations

import importlib.util
import os

from PIL import Image

from ocr.base import OCRBackend, OCRLine, OCRResult
from ocr.vlm_utils import threads

AZRA_MODEL = "OttomanNLP/Azra-1-Mini-0.8b"
AZRA_REVISION = "7c1c429e9f"  # verified 2026-11-02
AZRA_PROMPT = "Görseldeki Osmanlıca metni transkribe et:"


def _pad_to_multiple(image: Image.Image, multiple: int = 64) -> Image.Image:
    w, h = image.size
    nw, nh = -(-w // multiple) * multiple, -(-h // multiple) * multiple
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), Image.Resampling.LANCZOS)  # as in the model card


class AzraBackend(OCRBackend):
    name = "ottoman-azra"

    @classmethod
    def check_available(cls) -> tuple[bool, str]:
        for pkg in ("transformers", "torch", "kraken"):
            if importlib.util.find_spec(pkg) is None:
                return False, f"package '{pkg}' not installed (pip install -e '.[hf,kraken]')"
        return True, "ok"

    def _load(self) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        from layout.kraken_layout import KrakenLineSegmenter

        torch.set_num_threads(threads())
        revision = self.config.get("revision", AZRA_REVISION)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(AZRA_MODEL, revision=revision)
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            AZRA_MODEL, revision=revision, dtype=dtype).to(self._device).eval()
        self._segmenter = KrakenLineSegmenter(self.config.get("segmentation_model"))
        self.model_version = (f"{AZRA_MODEL}@{revision} ({str(dtype).replace('torch.', '')}, "
                              f"transformers {transformers.__version__}) + seg {self._segmenter.version}")

    def _transcribe_line(self, line_image: Image.Image) -> tuple[str, float | None]:
        import torch

        img = _pad_to_multiple(line_image)
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": AZRA_PROMPT}]}]
        inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=self.config.get("max_new_tokens", 256), do_sample=False,
                repetition_penalty=1.2, no_repeat_ngram_size=3,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[:, inputs["input_ids"].shape[1]:]
        scores = self._model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        conf = float(scores.exp().mean()) if scores.numel() else None
        text = self._processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text.strip(), conf

    def _recognize(self, image: Image.Image) -> OCRResult:
        lines = []
        for region in self._segmenter.lines(image):
            if region.image.width < 8 or region.image.height < 8:
                continue
            text, conf = self._transcribe_line(region.image)
            lines.append(OCRLine(text=text, confidence=conf, bbox=region.bbox,
                                 polygon=region.polygon, baseline=region.baseline))
        return OCRResult(text="\n".join(l.text for l in lines), model_name=self.name,
                         model_version=self.model_version, lines=lines,
                         metadata={"n_lines": len(lines), "prompt": AZRA_PROMPT})


class TranskribusBackend(OCRBackend):
    name = "transkribus-ottoman"
    model_version = "Transkribus OttomanTurkish_generic (model 56496)"

    @classmethod
    def check_available(cls) -> tuple[bool, str]:
        reason = ("no public weights; only via the paid Transkribus metagrapho API (Organisation plan, "
                  "credits) which uploads images to a third-party server — not implemented in the "
                  "baseline. OttomanTurkish_Print (52502) outputs Latin transliteration, not Arabic script.")
        if not (os.environ.get("TRANSKRIBUS_USER") and os.environ.get("TRANSKRIBUS_PASSWORD")):
            return False, reason
        return False, "credentials found, but the metagrapho client is not implemented yet: " + reason

    def _recognize(self, image: Image.Image) -> OCRResult:  # pragma: no cover
        raise NotImplementedError
