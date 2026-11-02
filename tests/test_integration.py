"""End-to-end without model weights: tiny manifest -> evaluate (dummy backend) -> metrics files."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from benchmark.manifest import COLUMNS, write_manifest
from evaluation.evaluate import main as evaluate_main

ROOT = Path(__file__).resolve().parents[1]
GT = "حاضرگه بزنڭ مطبوعات\nعالمیمزده اصول صوتیه"


@pytest.fixture
def tiny_benchmark(tmp_path):
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "ground_truth").mkdir()
    rows = []
    for i, (source, ptype) in enumerate([("bookA", "printed"), ("bookA", "printed"), ("msB", "handwritten")]):
        Image.new("RGB", (120, 80), "white").save(data / "raw" / f"{i}.png")
        (data / "ground_truth" / f"{i}.txt").write_text(GT + "\n", encoding="utf-8")
        r = {c: "" for c in COLUMNS}
        r.update(id=f"page{i}", image_path=f"raw/{i}.png", source_id=source, page=str(i), type=ptype,
                 script_style="iske_imla_naskh", approx_year="1915", ground_truth_path=f"ground_truth/{i}.txt",
                 split="test", subset="core", gt_status="gold_verified")
        rows.append(r)
    manifest = data / "benchmark_manifest.csv"
    write_manifest(rows, manifest)
    return manifest, tmp_path


def run_eval(manifest, out, text):
    cfg = out / "backend.yaml"
    cfg.write_text(yaml.safe_dump({"text": text}, allow_unicode=True), encoding="utf-8")
    code = evaluate_main(["--model", "dummy", "--manifest", str(manifest), "--split", "test",
                          "--backend-config", str(cfg), "--output-dir", str(out / "results")])
    assert code == 0
    return next((out / "results" / "dummy").iterdir())


def test_perfect_prediction_scores_zero(tiny_benchmark):
    manifest, tmp = tiny_benchmark
    run_dir = run_eval(manifest, tmp, GT)
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["overall"]["cer"] == 0 and m["overall"]["wer"] == 0 and m["overall"]["pages"] == 3
    assert set(m["by_type"]) == {"printed", "handwritten"}
    assert set(m["by_source_id"]) == {"bookA", "msB"}
    preds = [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 3 and preds[0]["text"] == GT
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert {"git", "hardware", "libraries", "dataset", "seed", "preprocessing"} <= set(cfg)


def test_imperfect_prediction_and_rescore(tiny_benchmark):
    manifest, tmp = tiny_benchmark
    hyp = GT.replace("ڭ", "ك").replace("ی", "ي")  # only conventional variants differ
    run_dir = run_eval(manifest, tmp, hyp)
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["overall"]["cer"] > 0
    assert m["overall"]["cer_normalized"] < m["overall"]["cer"]  # ی/ي folded, ڭ/ك kept
    assert evaluate_main(["--rescore", str(run_dir)]) == 0
    m2 = json.loads((run_dir / "metrics.json").read_text())
    assert m2["overall"]["cer"] == pytest.approx(m["overall"]["cer"])


def test_cli_ocr_outputs_json(tmp_path):
    img = tmp_path / "p.png"
    Image.new("RGB", (60, 40), "white").save(img)
    out = subprocess.run([sys.executable, str(ROOT / "cli.py"), "ocr", str(img), "--model", "dummy"],
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    assert result["model"] == "dummy" and "raw_ocr" in result and result["modern_tatar"] is None


def test_cli_models_lists_backends():
    out = subprocess.run([sys.executable, str(ROOT / "cli.py"), "models", "--json"],
                         capture_output=True, text=True, check=True)
    names = {r["name"] for r in json.loads(out.stdout)}
    assert {"paddle", "churro", "kraken-ottoman", "transkribus-ottoman"} <= names


def test_resume_skips_done_pages_and_retries_failures(tiny_benchmark):
    manifest, tmp = tiny_benchmark
    run_dir = run_eval(manifest, tmp, GT)
    preds = run_dir / "predictions.jsonl"
    lines = preds.read_text(encoding="utf-8").splitlines()
    failed = json.loads(lines[1])
    failed.update(text="", error="RuntimeError: simulated")
    preds.write_text("\n".join([lines[0], json.dumps(failed, ensure_ascii=False)]) + "\n", encoding="utf-8")
    assert evaluate_main(["--resume", str(run_dir)]) == 0
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["status"] == "ok" and m["overall"]["pages"] == 3 and m["overall"]["cer"] == 0
