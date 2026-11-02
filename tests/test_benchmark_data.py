from benchmark.manifest import COLUMNS, read_manifest, source_leakage, validate, write_manifest
from benchmark.wikisource import script_profile, wikitext_to_text


def row(i, source, split, gt="gt.txt", **kw):
    base = {c: "" for c in COLUMNS}
    base.update(id=f"p{i}", image_path="img.jpg", source_id=source, page=str(i), type="printed",
                script_style="iske_imla_naskh", approx_year="1915", ground_truth_path=gt, split=split,
                gt_status="silver_unproofread")
    base.update(kw)
    return base


def test_split_leakage_detected():
    rows = [row(1, "bookA", "train"), row(2, "bookA", "test"), row(3, "bookB", "test")]
    assert source_leakage(rows) == {"bookA": {"train", "test"}}
    errors = validate(rows, check_files=False)
    assert any("bookA" in e and "split by source_id" in e for e in errors)


def test_valid_manifest_passes_and_roundtrips(tmp_path):
    rows = [row(1, "bookA", "train"), row(2, "bookB", "test"), row(3, "photo", "unlabeled", gt="", gt_status="none")]
    assert validate(rows, check_files=False) == []
    path = tmp_path / "m.csv"
    write_manifest(rows, path)
    assert [r["id"] for r in read_manifest(path)] == ["p1", "p2", "p3"]


def test_eval_split_requires_ground_truth_and_valid_type():
    errors = validate([row(1, "a", "test", gt=""), row(2, "b", "test", type="typed")], check_files=False)
    assert any("requires ground_truth_path" in e for e in errors)
    assert any("type must be" in e for e in errors)


def test_missing_files_reported(tmp_path):
    manifest = tmp_path / "m.csv"
    errors = validate([row(1, "a", "test")], manifest, check_files=True)
    assert any("not found" in e for e in errors)


def test_wikitext_conversion_rules():
    wt = ('<noinclude><pagequality level="1" user="X" /></noinclude><h2>مقدمه</h2>\n'
          "بزنڭ&zwnj;ده {{?|چوالتولریدر}}<br>ایکی<!---->نچی\n"
          "*{{ditto}} كتاب {{SIC|اوقو|اوقی}} {{lang|tt|سوز}}{{rule}}\n"
          "{{tt/икетелле сүзлек\n|ئات\n|Лошадь\n}}\n"
          '{|class="wikitable"\n!ع.\n|عەرەبچە\n|-\n!ف.\n|فارسىُچا\n|}\n'
          "<noinclude></noinclude>")
    text, info = wikitext_to_text(wt)
    assert text.split("\n") == ["مقدمه", "", "بزنڭ‌ده چوالتولریدر", "ایکینچی",
                                "〃 كتاب اوقو سوز", "", "ئات", "Лошадь", "", "ع. عەرەبچە", "ف. فارسىُچا"]
    assert info["uncertain_spans"] == ["چوالتولریدر"]
    assert info["ditto_marks"] == 1 and info["unknown_templates"] == []


def test_unknown_templates_are_reported():
    _, info = wikitext_to_text("سوز {{mystery|x}}")
    assert info["unknown_templates"] == ["mystery|x"]


def test_script_profile():
    p = script_profile("ئات Лошадь")
    assert p["arabic"] == 3 and p["cyrillic"] == 6 and 0 < p["arabic_share"] < 0.5
