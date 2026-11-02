#!/usr/bin/env bash
# Full Phase 1 protocol. Idempotent: finished runs are skipped, interrupted runs resumed,
# so it is safe to re-run after a crash or reboot. One model at a time (honest timings).
#
#   setsid nohup bash scripts/run_phase1.sh > results/logs/phase1.log 2>&1 &
#
# Steps: resume partial runs -> E1 every backend on `core` (original) -> E2 `scan` vs
# `original` -> E3 CHURRO-3B and Azra (waits for weights) -> report, error analysis, examples.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export OLD_TATAR_OCR_SEG_CACHE=${OLD_TATAR_OCR_SEG_CACHE:-$HOME/.cache/old_tatar_ocr/segmentation}
LOG=results/logs
mkdir -p "$LOG" results/summary
log() { echo "=== $(date -Is) $*"; }

# Weights download in parallel (network-bound) while CPU-bound runs proceed.
if ! $PY scripts/download_models.py --check churro azra 2>/dev/null; then
  $PY scripts/download_models.py churro azra >> "$LOG/downloads.log" 2>&1 &
fi

run() {  # run <model> <preprocess> [subset]
  local model=$1 pre=$2 subset=${3:-core}
  local tag="$pre-$subset"
  if ls results/"$model"/*-"$tag"/metrics.json >/dev/null 2>&1; then log "skip $model $tag (done)"; return; fi
  local partial
  partial=$(ls -d results/"$model"/*-"$tag" 2>/dev/null | tail -1)
  if [ -n "$partial" ]; then
    log "resume $partial"
    $PY -m evaluation.evaluate --resume "$partial" >> "$LOG/$model-$tag.log" 2>&1
  else
    log "run $model $tag"
    $PY -m evaluation.evaluate --model "$model" --subset "$subset" --preprocess "$pre" --tag "$tag" \
        --save-lines >> "$LOG/$model-$tag.log" 2>&1
  fi
  log "    exit=$? $(grep -E '^normalized CER' "$LOG/$model-$tag.log" | tail -1)"
}

# 0. Interrupted runs of any kind (config.yaml written, metrics.json missing).
for d in results/*/*/; do
  if [ -f "$d/config.yaml" ] && [ ! -f "$d/metrics.json" ]; then
    m=$(basename "$(dirname "$d")")
    log "resume interrupted $d"
    $PY -m evaluation.evaluate --resume "$d" >> "$LOG/$m-resume.log" 2>&1
    log "    exit=$?"
  fi
done

# E1: every available CPU backend on the core pages, original images.
for m in kraken-ppocrv6 paddle kraken-ottoman kraken-arabic-script kraken-persian kraken-arabic \
         arabic-htr-phoenix easyocr; do
  run "$m" original
done

# E2: preprocessing effect for the strongest line-based and box-based models.
for m in kraken-ppocrv6 kraken-ottoman paddle; do
  run "$m" scan
done

# E3: vision-language models once their weights are present.
until $PY scripts/download_models.py --check churro 2>/dev/null; do sleep 60; done
run churro original
until $PY scripts/download_models.py --check azra 2>/dev/null; do sleep 60; done
run ottoman-azra original

# Reports.
log "report"
$PY -m evaluation.report > "$LOG/report.log" 2>&1
$PY -m evaluation.error_analysis --group core > "$LOG/error_analysis.log" 2>&1
$PY scripts/make_examples.py --pages ws_tatar_maktabe_1915_p0004 ws_rus_maktaplare_1925_p0006 \
    ws_tatarcha_uryscha_1927_p0013 --lines 6 --out docs/examples >> "$LOG/report.log" 2>&1

# Qualitative phone-photo check (no ground truth): original vs mobile preprocessing.
for pre in original mobile; do
  $PY cli.py ocr data/raw/commons_bukvar_1912_photo/image.jpg --model kraken-ppocrv6 --preprocess "$pre" \
      --lines --out "results/summary/photo_1912_${pre}.json" > /dev/null 2>> "$LOG/report.log"
done
log ALL_DONE
