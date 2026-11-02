"""Real-model smoke tests (download weights). Run with: pytest --run-heavy -m heavy"""

from pathlib import Path

import pytest
from PIL import Image

from evaluation.metrics import cer
from ocr.registry import availability, create_backend
from text.arabic_normalization import normalize_for_evaluation

DATA = Path(__file__).resolve().parents[1] / "data"
PAGE = "ws_tatar_maktabe_1915_p0004"


@pytest.mark.heavy
@pytest.mark.parametrize("name", ["kraken-ppocrv6", "kraken-ottoman", "paddle"])
def test_real_backend_reads_old_tatar_page(name):
    ok, reason = availability(name)
    if not ok:
        pytest.skip(reason)
    image = DATA / "raw" / "ws_tatar_maktabe_1915" / "p0004.jpg"
    if not image.exists():
        pytest.skip("run scripts/fetch_wikisource_benchmark.py first")
    result = create_backend(name).recognize(Image.open(image))
    gt = (DATA / "ground_truth" / f"{PAGE}.txt").read_text(encoding="utf-8")
    score = cer(normalize_for_evaluation(gt), normalize_for_evaluation(result.text))
    assert score < 0.5, f"{name}: normalized CER {score:.2%}"
