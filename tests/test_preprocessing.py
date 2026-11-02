import numpy as np
import pytest
from PIL import Image, ImageDraw

from preprocessing import PreprocessConfig, load_preprocess_config, preprocess
from preprocessing import steps


def text_page(w=800, h=1000, angle=0.0) -> Image.Image:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for y in range(80, h - 80, 40):
        d.rectangle([60, y, w - 60, y + 12], fill="black")  # dense "text lines"
    return img.rotate(angle, fillcolor="white", resample=Image.Resampling.BICUBIC) if angle else img


def test_original_config_only_exif_and_does_not_mutate_input():
    img = text_page()
    before = np.asarray(img).copy()
    res = preprocess(img, PreprocessConfig())
    assert [s["step"] for s in res.steps] == ["exif_orientation"]
    assert np.array_equal(np.asarray(img), before)
    assert np.array_equal(np.asarray(res.image), before)
    assert res.image is not img


def test_exif_orientation_applied():
    img = Image.new("RGB", (40, 20), "white")
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 CW
    img.info["exif"] = exif.tobytes()
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    loaded = Image.open(BytesIO(buf.getvalue()))
    res = preprocess(loaded, PreprocessConfig())
    assert res.image.size == (20, 40)
    assert res.steps[0]["exif_orientation"] == 6


@pytest.mark.parametrize("angle", [-3.0, 2.0])
def test_deskew_estimates_rotation(angle):
    est = steps.estimate_skew(text_page(angle=angle))
    assert est == pytest.approx(-angle, abs=0.5)


def test_deskew_skips_tiny_angles():
    res = preprocess(text_page(), PreprocessConfig(deskew=True))
    log = [s for s in res.steps if s["step"] == "deskew"][0]
    assert log["applied"] is False


def test_document_detection_and_perspective():
    bg = Image.new("RGB", (1000, 1000), (40, 40, 40))
    d = ImageDraw.Draw(bg)
    quad = [(220, 150), (800, 200), (760, 900), (180, 850)]
    d.polygon(quad, fill="white")
    found = steps.detect_document(bg)
    assert found is not None
    assert np.allclose(found, np.array(quad, dtype=float), atol=15)
    res = preprocess(bg, PreprocessConfig(document_detection=True, perspective_correction=True))
    assert any(s["step"] == "perspective_correction" for s in res.steps)
    assert np.asarray(res.image).mean() > 200  # mostly page, little background


def test_document_detection_none_on_plain_scan():
    assert steps.detect_document(text_page()) is None


def test_resize_caps_longest_side():
    res = preprocess(text_page(800, 1000), PreprocessConfig(max_side=500))
    assert max(res.image.size) == 500


def test_contrast_and_binarize():
    img = text_page()
    assert preprocess(img, PreprocessConfig(contrast="clahe")).image.size == img.size
    out = preprocess(img, PreprocessConfig(binarize=True)).image
    assert set(np.unique(np.asarray(out.convert("L")))) <= {0, 255}
    assert PreprocessConfig().binarize is False


def test_config_loading(tmp_path):
    assert load_preprocess_config("original").name == "original"
    scan = load_preprocess_config("scan")
    assert scan.deskew and not scan.binarize
    p = tmp_path / "bad.yaml"
    p.write_text("deskew: true\nfoo: 1\n")
    with pytest.raises(ValueError):
        load_preprocess_config(str(p))
