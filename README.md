# Iske-tatar-tele-reader — OCR старотатарских текстов (арабская графика)

Исследовательский репозиторий системы распознавания исторических старотатарских текстов,
написанных арабской графикой (иске имля до 1920 г., яңа имля 1920–1927 гг.).

Работа ведётся поэтапно. **Сейчас реализованы Phase 0 (исследование) и Phase 1 (baseline):**
воспроизводимая инфраструктура, которая запускает несколько готовых OCR/HTR-моделей на одних и тех же
реальных старотатарских страницах и честно измеряет CER/WER.

| документ | содержание |
|---|---|
| [docs/model_research.md](docs/model_research.md) | Phase 0: модели, лицензии, применимость |
| [docs/baseline_report.md](docs/baseline_report.md) | Phase 1: результаты сравнения (**предварительные**, прогоны не завершены) |
| [docs/architecture.md](docs/architecture.md) | модули и архитектурные решения (выбор / альтернативы / риски) |
| [docs/dataset.md](docs/dataset.md) | стандарт данных, бенчмарк, нужная человеческая разметка |
| [docs/experiments.md](docs/experiments.md) | протокол и журнал экспериментов |

Целевой конвейер (OCR, транскрипция и перевод — независимые модули):

```
image → preprocessing → layout / lines → OCR/HTR (арабица, raw_ocr) → [Phase 4] современный татарский → [Phase 5] ru / ba
```

`raw_ocr` никогда не исправляется языковой моделью незаметно: будущая пост-коррекция пишет в отдельное поле `corrected_ocr`.

## Установка

Python 3.10–3.13. Ядро (метрики, нормализация, предобработка, раннер) ставится за секунды; каждый
OCR-движок — отдельный extra.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "torch>=2.9,<2.12" torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-сборка
pip install -e ".[paddle,kraken,hf,gguf,easyocr,data,dev]"
```

Минимально (только тесты и метрики): `pip install -e ".[dev]"`. Docker (CPU, GPU опционально): см. `Dockerfile`.

Веса моделей скачиваются при первом использовании (Zenodo / Hugging Face, с проверкой checksum) в
`~/.cache/old_tatar_ocr/models` (меняется `OLD_TATAR_OCR_MODELS`, см. `.env.example`).

## Данные

Изображения страниц не хранятся в git. Восстановить бенчмарк (45 размеченных страниц из 3 книг 1915–1927 гг.,
изображения — Public Domain с Wikimedia Commons, GT — транскрипции Wikisource по закреплённым ревизиям):

```bash
python scripts/fetch_wikisource_benchmark.py
python -m benchmark.manifest validate data/benchmark_manifest.csv
```

## Использование

Список бэкендов, их лицензий и причин недоступности:

```bash
python cli.py models
```

OCR одного изображения:

```bash
python cli.py ocr data/raw/ws_tatar_maktabe_1915/p0004.jpg --model kraken-ppocrv6
```

Фактический вывод (сокращён; заголовок «مقدمه» модель прочитала как «doлao» — пример реальной ошибки):

```json
{
  "raw_ocr": "doлao\nحاضرگه بزنڭ مطبوعات عالمیمزده اصول صوتیه اوزره ثرتيب ایدلمش\nالفبالاريمزبيك آز وبيك يا ݣا أش بولوب تورالار. ...",
  "corrected_ocr": null,
  "modern_tatar": null,
  "translation": {},
  "confidence": 0.937,
  "confidence_band": "high",
  "model": "kraken-ppocrv6",
  "model_version": "ppocrv6-medium@e3411a453ce3 [doi:10.5281/zenodo.21788410] + seg blla.mlmodel@77a638a83c9e",
  "processing_time": 57.9,
  "ocr_time": 42.5,
  "model_load_time": 15.0,
  "preprocessing": {"config": "original", "steps": [{"step": "exif_orientation", "exif_orientation": 1}]}
}
```

(CPU, 16 потоков; уверенность движка не откалибрована — «high» при заметных ошибках.)

Фото с телефона: `--preprocess mobile` (поиск страницы, перспектива, deskew, контраст). `--lines` добавляет строки с
рамками и уверенностью.

Бенчмарк одной модели:

```bash
python -m evaluation.evaluate --model paddle --manifest data/benchmark_manifest.csv --split test
python -m evaluation.evaluate --model churro --subset core --preprocess scan
```

Формат вывода (поля; значения см. в `docs/baseline_report.md`):

```
model: <backend>
documents: <N sources>
pages: <N pages>
CER: …%   WER: …%
normalized CER: …%   normalized WER: …%
loose CER: …%   arabic-only CER: …%   BoW-F1: …%
runtime/page: …s (median …s), model load: …s
GPU memory: n/a (CPU run)   peak RSS: … MB
  type=printed: CER …, normalized CER …, arabic-only CER … (N pages)
  source_id=<source>: …
```

Результаты: `results/<model>/<timestamp>/{metrics.json,predictions.jsonl,config.yaml}`
(`config.yaml` фиксирует git commit, версию/хэш модели, хэш датасета, конфиг, seed, железо и версии библиотек).
Долгий прогон можно продолжить `--resume <run_dir>`, пересчитать метрики без инференса — `--rescore <run_dir>`.

Весь протокол Phase 1 одной командой (перезапускаемый: готовые прогоны пропускаются, прерванные продолжаются;
веса качаются параллельно с проверкой checksum):

```bash
setsid nohup bash scripts/run_phase1.sh > results/logs/phase1.log 2>&1 &
```

Отдельные шаги:

```bash
python scripts/download_models.py churro azra      # заранее скачать веса (с докачкой)
bash scripts/run_baseline.sh                       # все бэкенды, исходные изображения
PREPROCESS=scan MODELS="kraken-ppocrv6 paddle" bash scripts/run_baseline.sh
python -m evaluation.report                        # → results/summary/baseline_summary.md
python -m evaluation.error_analysis --group core   # → results/summary/error_analysis_core.md
```

Задачи для разметчиков (вычитка silver GT, черновики для неразмеченных страниц, строки с низкой уверенностью):

```bash
python scripts/make_annotation_tasks.py --run results/kraken-ppocrv6/<run> --model kraken-ppocrv6
```

## Тесты

```bash
pytest                 # unit + integration, без скачивания моделей
pytest --run-heavy     # + реальные модели на странице 1915 г.
```

## Структура

```
cli.py                 # models | ocr | benchmark | report
src/ocr/               # OCRBackend, registry, адаптеры (paddle, easyocr, tesseract, kraken, phoenix, churro, azra, transkribus)
src/layout/            # kraken BLLA сегментация строк (+кэш), RTL-порядок
src/preprocessing/     # конфигурируемая предобработка
src/text/              # нормализация арабицы: хранение без потерь vs профили для метрик
src/evaluation/        # metrics, evaluate, scoring, report, repro
src/benchmark/         # манифест (+валидация сплитов), Wikisource → GT
src/pipeline/          # схема результата raw_ocr / corrected_ocr / modern_tatar / translation
configs/preprocess/    # original (по умолчанию), scan, mobile
data/                  # manifest, GT, sources; изображения скачиваются скриптом
scripts/               # fetch_wikisource_benchmark, run_baseline, make_annotation_tasks
docs/                  # исследование, архитектура, данные, эксперименты, отчёт
```

## Приватность

CLI и раннер не сохраняют изображения пользователя и не отправляют их третьим сторонам. Бэкенды,
требующие внешних API (Transkribus, коммерческие VLM), в baseline не подключены.

## Лицензия

Код — Apache-2.0. Лицензии моделей различаются (CHURRO-3B — Qwen Research License, некоммерческая;
Phoenix — CC BY-NC-SA 2.0; OpenITI — CC0; PP-OCRv6, PaddleOCR, Azra — Apache-2.0): см. `python cli.py models`
и `docs/model_research.md`.
