import json

import pytest
from PIL import Image

from layout.reading_order import merge_rows_rtl
from ocr.base import OCRLine, OCRResult, weighted_confidence
from ocr.registry import SPECS, availability, create_backend, list_backends
from ocr.vlm_utils import churro_xml_to_text, resize_to_pixel_budget
from pipeline import confidence_band, run_ocr
from preprocessing import PreprocessConfig


def test_dummy_backend_contract():
    b = create_backend("dummy", text="بر\nایکی", confidence=0.8)
    r = b.recognize(Image.new("L", (50, 30), 255))
    assert isinstance(r, OCRResult)
    assert r.text == "بر\nایکی" and r.model_name == "dummy" and r.model_version
    assert r.processing_time >= 0 and r.confidence == pytest.approx(0.8)
    assert len(r.lines) == 2 and r.lines[0].bbox == (0, 0, 50, 30)


def test_result_json_roundtrip():
    r = OCRResult(text="كتاب", model_name="m", model_version="v", confidence=0.5,
                  lines=[OCRLine("كتاب", 0.5, (1, 2, 3, 4), [(1, 2), (3, 4)], [(1, 3), (3, 3)])],
                  metadata={"k": 1})
    d = json.loads(json.dumps(r.to_dict(), ensure_ascii=False))
    assert OCRResult.from_dict(d) == r


def test_weighted_confidence():
    lines = [OCRLine("aaaa", 1.0), OCRLine("a", 0.0), OCRLine("x", None)]
    assert weighted_confidence(lines) == pytest.approx(0.8)
    assert weighted_confidence([OCRLine("a")]) is None


def test_registry_unknown_backend():
    with pytest.raises(KeyError):
        create_backend("does-not-exist")


def test_every_backend_reports_availability_without_crashing():
    for name in SPECS:
        ok, reason = availability(name)
        assert isinstance(ok, bool) and isinstance(reason, str) and reason
    assert any(r["name"] == "paddle" for r in list_backends())


def test_transkribus_is_registered_but_unavailable():
    ok, reason = availability("transkribus-ottoman")
    assert ok is False and "metagrapho" in reason


def test_rtl_row_merging():
    boxes = [OCRLine("left", 0.5, (10, 100, 100, 130)), OCRLine("right", 1.0, (200, 102, 300, 128)),
             OCRLine("top", 0.9, (10, 10, 300, 40))]
    rows = merge_rows_rtl(boxes)
    assert [r.text for r in rows] == ["top", "right left"]
    assert rows[1].bbox == (10, 100, 300, 130)


def test_churro_xml_extraction():
    xml = ("<HistoricalDocument><Metadata><Description>a page</Description></Metadata><Page>"
           "<Header><Line>سرلوحه</Line></Header><Body><Paragraph><Line>بر سطر</Line>"
           "<Line>ایکنچی <Illegible/>سطر</Line></Paragraph></Body></Page></HistoricalDocument>")
    text, ok = churro_xml_to_text(xml)
    assert ok and text.split("\n") == ["سرلوحه", "بر سطر", "ایکنچی سطر"]
    truncated, ok = churro_xml_to_text("<HistoricalDocument><Page><Body><Line>بر سطر</Line><Line>ایک")
    assert not ok and "بر سطر" in truncated


def test_pixel_budget_resize():
    img = Image.new("RGB", (4000, 6000))
    out = resize_to_pixel_budget(img, 400_000, 1_600_000)
    assert out.width * out.height <= 1_600_000 and out.width % 28 == 0 and out.height % 28 == 0
    small = resize_to_pixel_budget(Image.new("RGB", (100, 100)), 400_000, 1_600_000)
    assert small.width * small.height >= 300_000


def test_pipeline_result_schema(tmp_path):
    p = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(p)
    out = run_ocr(p, create_backend("dummy", text="سوز", confidence=0.95), PreprocessConfig(), include_lines=True)
    assert out["raw_ocr"] == "سوز" and out["corrected_ocr"] is None
    assert out["modern_tatar"] is None and out["translation"] == {}
    assert out["model"] == "dummy" and out["confidence_band"] == "high"
    assert out["lines"][0]["text"] == "سوز"
    json.dumps(out, ensure_ascii=False)


def test_confidence_bands():
    assert [confidence_band(x) for x in (None, 0.95, 0.8, 0.1)] == [None, "high", "medium", "low"]
