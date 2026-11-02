#!/usr/bin/env bash
# Phase 1 baseline: every available backend on the same labelled test pages.
#
#   bash scripts/run_baseline.sh                 # default model list, original images
#   MODELS="paddle churro" bash scripts/run_baseline.sh
#   PREPROCESS=scan TAG=scan bash scripts/run_baseline.sh
#   SUBSET=core bash scripts/run_baseline.sh
#
# Segmentations are cached so that all kraken-based recognizers (and the line VLM)
# read exactly the same lines.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
MODELS=${MODELS:-"kraken-ppocrv6 kraken-ottoman kraken-arabic-script kraken-persian kraken-arabic arabic-htr-phoenix paddle easyocr"}
PREPROCESS=${PREPROCESS:-original}
TAG=${TAG:-$PREPROCESS}
SPLIT=${SPLIT:-test}
SUBSET_ARGS=()
if [ -n "${SUBSET:-}" ]; then SUBSET_ARGS=(--subset "$SUBSET"); TAG="$TAG-$SUBSET"; fi
export OLD_TATAR_OCR_SEG_CACHE=${OLD_TATAR_OCR_SEG_CACHE:-$HOME/.cache/old_tatar_ocr/segmentation}
mkdir -p results/logs

for model in $MODELS; do
  echo "=== $(date -Is) $model ($PREPROCESS)"
  $PY -m evaluation.evaluate --model "$model" --split "$SPLIT" "${SUBSET_ARGS[@]}" \
      --preprocess "$PREPROCESS" --tag "$TAG" --save-lines \
      > "results/logs/${model}-${TAG}.log" 2>&1
  echo "    exit=$? log=results/logs/${model}-${TAG}.log"
  grep -E "^(CER|normalized CER|runtime/page|status)" "results/logs/${model}-${TAG}.log" | sed 's/^/    /'
done
