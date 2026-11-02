# CPU image for the Phase 1 baseline (OCR backends + evaluation).
#   docker build -t old-tatar-ocr .
#   docker run --rm -v $PWD/data:/app/data -v $PWD/results:/app/results \
#       -v $HOME/.cache/old_tatar_ocr:/root/.cache/old_tatar_ocr old-tatar-ocr \
#       python -m evaluation.evaluate --model kraken-ppocrv6 --split test
#
# GPU (optional): build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
# and run with --gpus all; the transformers backends (churro-hf, ottoman-azra) then use CUDA.
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG EXTRAS=paddle,kraken,hf,gguf,easyocr,data,dev

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir "torch>=2.9,<2.12" torchvision --index-url ${TORCH_INDEX} \
 && CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-cache-dir -e ".[${EXTRAS}]"

COPY . .
ENV OLD_TATAR_OCR_SEG_CACHE=/root/.cache/old_tatar_ocr/segmentation
CMD ["python", "cli.py", "models"]
