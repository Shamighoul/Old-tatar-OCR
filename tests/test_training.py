from pathlib import Path

import yaml
from PIL import Image

from benchmark.gt_store import GTStore
from benchmark.manifest import COLUMNS, write_manifest
from benchmark.pagexml import parse_page_xml, to_page_xml
from ocr.base import OCRLine
from training.dataset import build_training_set, eligible_pages, sample_pages
from training.kraken_finetune import FinetuneConfig, build_command
from training.synthetic import make_synthetic_set, render_page


def make_data(tmp_path, pages):
    """pages: list of (page_id, source_id, split, status)"""
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    store = GTStore(data)
    rows = []
    for pid, source, split, status in pages:
        Image.new("RGB", (200, 80), "white").save(data / "raw" / f"{pid}.jpg")
        xml = to_page_xml(f"{pid}.png", 200, 80, [OCRLine("بر سطر", None, (5, 5, 195, 40))], creator="human")
        store.commit(pid, "بر سطر", status, "annotator", "annotation", page_xml=xml)
        r = {c: "" for c in COLUMNS}
        r.update(id=pid, image_path=f"raw/{pid}.jpg", source_id=source, page="1", type="printed",
                 script_style="iske_imla_naskh", approx_year="1906", ground_truth_path=f"ground_truth/{pid}.txt",
                 split=split, gt_status=status)
        rows.append(r)
    write_manifest(rows, data / "benchmark_manifest.csv")
    return data


def test_training_set_never_contains_test_or_unverified_pages(tmp_path):
    data = make_data(tmp_path, [("a1", "A", "train", "gold_verified"), ("a2", "A", "train", "silver_unproofread"),
                                ("b1", "B", "val", "silver_proofread"), ("t1", "T", "test", "gold_verified")])
    ts = build_training_set(data, tmp_path / "ts")
    assert {p["id"] for p in ts.pages} == {"a1", "b1"}
    assert ts.list_file("train").read_text().count(".xml") == 1 and not (tmp_path / "ts" / "test").exists()
    doc = parse_page_xml((tmp_path / "ts" / "train" / "a1.xml").read_text(encoding="utf-8"))
    assert doc.image_filename == "a1.jpg" and (tmp_path / "ts" / "train" / "a1.jpg").exists()
    assert len(ts.version) == 16


def test_subsets_are_nested_and_cover_sources_first(tmp_path):
    pages = [{"id": f"{s}{i}", "source_id": s} for s in "ABCD" for i in range(5)]
    five, ten = sample_pages(pages, 5, seed=3), sample_pages(pages, 10, seed=3)
    assert [p["id"] for p in ten[:5]] == [p["id"] for p in five]
    assert len({p["source_id"] for p in five}) == 4


def test_ketos_command_fine_tunes_with_codec_union_and_early_stopping(tmp_path):
    model = tmp_path / "base.safetensors"
    model.write_bytes(b"x")
    cmd = build_command(FinetuneConfig(base=str(model), train_list="t.txt", val_list="v.txt", out_dir=str(tmp_path)))
    s = " ".join(cmd)
    for flag in ("-f page", f"-i {model}", "--resize union", "-t t.txt", "-e v.txt", "-q early", "-u NFD", "-B 1",
                 "--deterministic"):
        assert flag in s


def test_synthetic_pages_render_and_round_trip(tmp_path):
    img, lines = render_page(["بزنڭ مطبوعات", "كتبخانه سی"], seed=0)
    assert img.mode == "RGB" and len(lines) == 2 and lines[0].bbox[2] <= img.width
    xmls = make_synthetic_set(["بر", "ایکی", "اوچ"], tmp_path, lines_per_page=2)
    assert len(xmls) == 2
    doc = parse_page_xml(xmls[0].read_text(encoding="utf-8"))
    assert doc.text == "بر\nایکی" and (tmp_path / doc.image_filename).exists()
