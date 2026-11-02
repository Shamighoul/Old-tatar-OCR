import importlib.util
from pathlib import Path

import pytest
import yaml
from PIL import Image

from benchmark.gt_store import GTStore, init_from_manifest
from benchmark.manifest import COLUMNS, read_manifest, write_manifest
from benchmark.pagexml import parse_page_xml, to_page_xml
from benchmark.sources import validate_sources
from ocr.base import OCRLine

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- versioned GT ------------------------------------------------------------------------

def test_versions_are_immutable_and_current_follows_promotions(tmp_path):
    store = GTStore(tmp_path)
    v1 = store.commit("p1", "بر سطر", "silver_unproofread", "wiki", "wikisource:x@rev1")
    draft = store.commit("p1", "بر سطز", "draft_unverified", "model", "machine:m")
    v3 = store.commit("p1", "بر سطر\nایکی", "gold_verified", "annotator", "annotation:p1.xml", note="fixed")
    assert [v["version"] for v in store.history("p1")] == ["v001", "v002", "v003"]
    assert draft["promoted"] is False and store.current("p1")["version"] == "v003"
    assert store.text("p1", "v001") == "بر سطر\n"  # untouched by later commits
    assert (tmp_path / "ground_truth" / "p1.txt").read_text(encoding="utf-8") == "بر سطر\nایکی\n"
    assert v3["parent"] == "v002" and v3["cer_vs_parent"] > 0 and v1["cer_vs_parent"] is None
    assert "CER(v001 -> v003" in store.diff("p1", "v001", "v003")


def test_drafts_can_never_become_ground_truth(tmp_path):
    store = GTStore(tmp_path)
    with pytest.raises(ValueError):
        store.commit("p1", "x", "draft_unverified", "m", "machine", promote=True)
    store.commit("p1", "x", "draft_unverified", "m", "machine")
    assert store.current("p1") is None and not (tmp_path / "ground_truth" / "p1.txt").exists()


def test_existing_version_file_is_never_overwritten(tmp_path):
    store = GTStore(tmp_path)
    store.commit("p1", "a", "silver_unproofread", "w", "s")
    (tmp_path / "gt_history" / "p1" / "history.jsonl").unlink()  # simulate damaged history
    with pytest.raises(FileExistsError):
        store.commit("p1", "b", "gold_verified", "w", "s")


def test_init_from_manifest_is_idempotent(tmp_path):
    (tmp_path / "ground_truth").mkdir()
    (tmp_path / "ground_truth" / "p1.txt").write_text("سوز\n", encoding="utf-8")
    row = {c: "" for c in COLUMNS}
    row.update(id="p1", image_path="x.jpg", source_id="s", page="1", type="printed", script_style="x",
               approx_year="1915", ground_truth_path="ground_truth/p1.txt", split="test", gt_status="silver_unproofread")
    write_manifest([row], tmp_path / "benchmark_manifest.csv")
    store = GTStore(tmp_path)
    assert init_from_manifest(store, tmp_path / "benchmark_manifest.csv") == 1
    assert init_from_manifest(store, tmp_path / "benchmark_manifest.csv") == 0
    assert store.current("p1")["status"] == "silver_unproofread"


# --- PAGE XML ------------------------------------------------------------------------------

def test_page_xml_roundtrip_keeps_order_polygons_baselines_text():
    lines = [OCRLine("حاضرگه بزنڭ", 0.91, (10, 10, 200, 40), [(10, 10), (200, 10), (200, 40), (10, 40)],
                     [(10, 35), (200, 35)]),
             OCRLine("كتبخانه‌سی & <ok>", None, (10, 50, 180, 80))]
    xml = to_page_xml("p1.jpg", 300, 400, lines, creator="test")
    doc = parse_page_xml(xml)
    assert doc.image_filename == "p1.jpg" and (doc.width, doc.height) == (300, 400)
    assert [l.text for l in doc.lines] == ["حاضرگه بزنڭ", "كتبخانه‌سی & <ok>"]
    assert doc.lines[0].baseline == [(10, 35), (200, 35)] and doc.lines[0].confidence == pytest.approx(0.91)
    assert doc.lines[1].polygon == [(10, 50), (180, 50), (180, 80), (10, 80)]
    assert doc.text == "حاضرگه بزنڭ\nكتبخانه‌سی & <ok>"


def test_parse_foreign_page_xml_follows_reading_order():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
 <Page imageFilename="a.png" imageWidth="10" imageHeight="10">
  <ReadingOrder><OrderedGroup id="g"><RegionRefIndexed index="1" regionRef="B"/>
   <RegionRefIndexed index="0" regionRef="A"/></OrderedGroup></ReadingOrder>
  <TextRegion id="B"><TextLine id="b1"><Coords points="0,5 9,5 9,9 0,9"/>
   <TextEquiv index="1"><Unicode>alt</Unicode></TextEquiv><TextEquiv index="0"><Unicode>ایکی</Unicode></TextEquiv>
  </TextLine></TextRegion>
  <TextRegion id="A"><TextLine id="a1"><Coords points="0,0 9,0 9,4 0,4"/>
   <TextEquiv><Unicode>بر</Unicode></TextEquiv></TextLine></TextRegion>
 </Page></PcGts>"""
    assert parse_page_xml(xml).text == "بر\nایکی"


# --- sources / families ------------------------------------------------------------------

def src(sid, split, family=None, **kw):
    s = {"source_id": sid, "kind": "commons_pages", "title": "t", "approx_year": 1900, "type": "printed",
         "script_style": "iske_imla_naskh", "language": "tt-Arab", "image_license": "PD", "commons_file": "f.pdf",
         "pages": [1], "planned_split": split, "gt_license": "TBD", "split": "unlabeled"}
    if family:
        s["family"] = family
    s.update(kw)
    return s


def test_family_must_not_span_splits():
    errors, _ = validate_sources([src("a", "train", "series"), src("b", "test", "series"), src("c", "val")])
    assert any("series" in e and "spans" in e for e in errors)
    assert validate_sources([src("a", "train", "s1"), src("b", "test", "s2")])[0] == []


def test_handwritten_source_requires_writer():
    errors, _ = validate_sources([src("a", "train", type="handwritten")])
    assert any("writer" in e for e in errors)


# --- import -> export round trip ------------------------------------------------------------

def test_import_corrected_xml_then_export_lines(tmp_path):
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    Image.new("RGB", (300, 120), "white").save(data / "raw" / "p1.jpg")
    row = {c: "" for c in COLUMNS}
    row.update(id="bookA_p0001", image_path="raw/p1.jpg", source_id="bookA", page="1", type="printed",
               script_style="iske_imla_naskh", approx_year="1906", split="unlabeled", subset="phase2_pilot",
               gt_status="none")
    write_manifest([row], data / "benchmark_manifest.csv")
    (data / "sources.yaml").write_text(yaml.safe_dump({"sources": [src("bookA", "train")]}), encoding="utf-8")
    store = GTStore(data)
    store.commit("bookA_p0001", "بر سطز", "draft_unverified", "model", "machine:m")
    xml = to_page_xml("bookA_p0001.jpg", 300, 120, [OCRLine("بر سطر", None, (10, 10, 290, 50)),
                                                     OCRLine("ایکنچی", None, (10, 60, 290, 100))], creator="human")
    xml_path = tmp_path / "bookA_p0001.xml"
    xml_path.write_text(xml, encoding="utf-8")

    importer = load_script("import_annotations")
    assert importer.main([str(xml_path), "--status", "gold_verified", "--author", "A", "--data", str(data)]) == 0
    r = read_manifest(data / "benchmark_manifest.csv")[0]
    assert r["split"] == "train" and r["gt_status"] == "gold_verified" and r["ground_truth_path"]
    assert store.current("bookA_p0001")["has_page_xml"]

    counts = load_script("export_lines").export(data, tmp_path / "lines", {"gold_verified"}, pad=2)
    assert counts == {"train": 2}
    jl = (tmp_path / "lines" / "lines.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jl) == 2 and '"text": "ایکنچی"' in jl[1]
    assert (tmp_path / "lines" / "train" / "bookA_p0001_l001.png").exists()
