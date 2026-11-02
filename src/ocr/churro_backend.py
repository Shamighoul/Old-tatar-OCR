"""CHURRO-3B adapter (stanford-oval/churro-3B, Qwen2.5-VL-3B fine-tuned on ~100K
historical pages, 46 language clusters incl. Arabic, Persian, Ottoman Turkish).

Prompting follows the official toolkit (github.com/stanford-oval/Churro): system
message "Transcribe the entirety of this historical document to XML format.", the
page image as the only user content, greedy decoding with repetition penalty 1.05.
The XML answer is converted to plain text with ``vlm_utils.churro_xml_to_text``; the
raw XML is kept in ``metadata['raw_output']``.

Engines
-------
* ``gguf`` (default here): llama.cpp with mradermacher/churro-3B-GGUF. Default
  ``quant=Q4_K_M`` + ``mmproj=Q8_0`` (2.8 GB; chosen because of a ~0.3 MB/s link on
  the baseline machine); ``quant=Q8_0`` + ``mmproj=f16`` is the near-lossless option.
  Quantization is a deviation from the reference model and is recorded in
  ``model_version``.
* ``hf``: reference transformers inference in bf16 (needs ~10 GB RAM/VRAM).

Image budget: the reference processor allows up to 4,014,080 pixels (5120 visual
tokens). On CPU we default to ``max_pixels=1,605,632`` (2048 tokens) — a documented
speed/accuracy trade-off; set ``max_pixels`` in the backend config to change it.

Licence: weights under the Qwen Research License (non-commercial).
"""

from __future__ import annotations

import importlib.util
import math

from PIL import Image

from ocr.base import OCRBackend, OCRResult
from ocr.model_store import ModelFile, fetch
from ocr.vlm_utils import churro_xml_to_text, image_to_data_uri, resize_to_pixel_budget, threads

HF_MODEL = "stanford-oval/churro-3B"
HF_REVISION = "ca2150ea465d"  # model repo commit verified 2026-11-02
GGUF_REPO = "mradermacher/churro-3B-GGUF"
GGUF_REVISION = "d4c90488a1b7afb9c20ad55629bffe2d6d6712b8"
SYSTEM_PROMPT = "Transcribe the entirety of this historical document to XML format."
REF_MIN_PIXELS = 401_408
CPU_MAX_PIXELS = 1_605_632


def _gguf(filename: str, sha256: str, size: int) -> ModelFile:
    return ModelFile(f"churro-3B-GGUF@{GGUF_REVISION[:12]}",
                     f"https://huggingface.co/{GGUF_REPO}/resolve/{GGUF_REVISION}/{filename}",
                     filename, f"sha256:{sha256}", size, "qwen-research", f"hf:{GGUF_REPO}@{GGUF_REVISION[:12]}")


GGUF_FILES = {
    "Q8_0": _gguf("churro-3B.Q8_0.gguf", "4cc4890947115ec5b63c422afceb3587dcde3185377520816f73666e4ffa57d4", 3285475392),
    "Q4_K_M": _gguf("churro-3B.Q4_K_M.gguf", "59267742188ad89be608093a599ae1b58d59bc931c0fafee4abc09f1bc825a2d", 1929902144),
    "mmproj-f16": _gguf("churro-3B.mmproj-f16.gguf", "e8509f7a8aec68639ecb2a3eb1dfae3949a07a9b954c70d630ad40a20cef90b9", 1338428800),
    "mmproj-Q8_0": _gguf("churro-3B.mmproj-Q8_0.gguf", "02c7f9d645e070736269a4a177cd4f919b8e700be706ca9d3aee83665974c823", 847768960),
}


class ChurroBackend(OCRBackend):
    name = "churro"

    @classmethod
    def check_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("huggingface_hub") is None:
            return False, "package 'huggingface_hub' not installed"
        return True, "ok"

    # ----------------------------------------------------------------------- load
    def _load(self) -> None:
        engine = self.config.get("engine", "gguf")
        if engine == "gguf":
            self._load_gguf()
        elif engine == "hf":
            self._load_hf()
        else:
            raise ValueError(f"unknown CHURRO engine {engine!r}")

    def _load_gguf(self) -> None:
        if importlib.util.find_spec("llama_cpp") is None:
            raise RuntimeError("llama-cpp-python not installed (pip install -e '.[gguf]')")
        import llama_cpp
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        quant = self.config.get("quant", "Q4_K_M")
        mmproj = self.config.get("mmproj", "Q8_0")
        model_path = fetch(GGUF_FILES[quant])
        proj_path = fetch(GGUF_FILES[f"mmproj-{mmproj}"])
        handler = Qwen25VLChatHandler(clip_model_path=str(proj_path), verbose=False)
        self._llm = Llama(model_path=str(model_path), chat_handler=handler,
                          n_ctx=self.config.get("n_ctx", 8192), n_threads=threads(),
                          n_batch=self.config.get("n_batch", 512), logits_all=False, verbose=False)
        self.model_version = (f"{GGUF_REPO}@{GGUF_REVISION[:12]} {quant} + mmproj-{mmproj} "
                              f"(llama-cpp-python {llama_cpp.__version__}, max_pixels="
                              f"{self.config.get('max_pixels', CPU_MAX_PIXELS)})")

    def _load_hf(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        revision = self.config.get("revision", HF_REVISION)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" or self.config.get("cpu_bf16", True) else torch.float32
        self._processor = AutoProcessor.from_pretrained(HF_MODEL, revision=revision)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            HF_MODEL, revision=revision, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
        self._device = device
        self.model_version = f"{HF_MODEL}@{revision} transformers {dtype}".replace("torch.", "")

    # ------------------------------------------------------------------ recognize
    def _prepare(self, image: Image.Image) -> Image.Image:
        return resize_to_pixel_budget(image, self.config.get("min_pixels", REF_MIN_PIXELS),
                                      self.config.get("max_pixels", CPU_MAX_PIXELS))

    def _recognize(self, image: Image.Image) -> OCRResult:
        img = self._prepare(image)
        if self.config.get("engine", "gguf") == "gguf":
            raw, confidence, usage = self._generate_gguf(img)
        else:
            raw, confidence, usage = self._generate_hf(img)
        text, parsed = churro_xml_to_text(raw)
        return OCRResult(text=text, model_name=self.name, model_version=self.model_version,
                         confidence=confidence,
                         metadata={"raw_output": raw, "xml_parsed": parsed, "input_size": list(img.size),
                                   "usage": usage})

    def _generate_gguf(self, img: Image.Image) -> tuple[str, float | None, dict]:
        resp = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_to_data_uri(img)}}]}],
            temperature=0.0, repeat_penalty=self.config.get("repetition_penalty", 1.05),
            max_tokens=self.config.get("max_new_tokens", 4096), logprobs=True, top_logprobs=1,
        )
        choice = resp["choices"][0]
        confidence = None
        content_lp = ((choice.get("logprobs") or {}).get("content")) or []
        if content_lp:
            confidence = sum(math.exp(t["logprob"]) for t in content_lp) / len(content_lp)
        usage = dict(resp.get("usage") or {})
        usage["finish_reason"] = choice.get("finish_reason")
        return choice["message"]["content"] or "", confidence, usage

    def _generate_hf(self, img: Image.Image) -> tuple[str, float | None, dict]:
        import torch

        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "image", "image": img}]}]
        inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=self.config.get("max_new_tokens", 4096),
                                       do_sample=False, repetition_penalty=self.config.get("repetition_penalty", 1.05),
                                       output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[:, inputs["input_ids"].shape[1]:]
        scores = self._model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        confidence = float(scores.exp().mean()) if scores.numel() else None
        text = self._processor.batch_decode(gen, skip_special_tokens=True)[0]
        return text, confidence, {"completion_tokens": int(gen.shape[1]), "prompt_tokens": int(inputs["input_ids"].shape[1])}
