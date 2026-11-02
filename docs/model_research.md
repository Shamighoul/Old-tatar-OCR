# Phase 0. Исследование существующих моделей и сервисов

Дата проверки: **2026-11-02**. Лицензии взяты из model card / репозитория / записи Zenodo,
а не «по умолчанию». Где источник молчит или противоречив, это указано явно.

Целевая задача: страница старотатарского текста (иске имля до 1920 г., яңа имля 1920–1927 гг.)
→ буквальная транскрипция **в арабской графике** (`raw_ocr`). Модель, которая выдаёт латиницу
или современную орфографию, для `raw_ocr` непригодна, даже если «читает» хорошо.

## 1. Сводная таблица

Обозначения: *line* — модель распознаёт вырезанную строку (нужен детектор/сегментатор строк);
*page* — принимает страницу целиком; *box* — детектор + распознаватель фрагментов.

| model | architecture | input level | printed / handwritten | languages | Arabic-script coverage | license | commercial use | GPU requirements | public weights | fine-tuning | expected advantages | expected weaknesses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Lisan.Tatar** (lisan.tatar) | закрытый AI-сервис (не раскрыта) | page (загрузка PDF/изображений, ≤20 стр.) | только печать («обучен работать с печатными шрифтами») | старотатарский, старобашкирский → 4 языка | целевая (иске имля) | условия сервиса, «экспериментальный», без гарантий | не определено, API нет | — (облако) | нет | нет | единственный найденный публичный сервис именно для старотатарского; перевод + транскрипция | нет API/весов → нельзя встроить и воспроизводимо измерить; изображения уходят третьей стороне |
| **Miras-Tat** (ИЯЛИ АН РТ, Мирасханэ) | не раскрыта | page | печать (журнал «Шура» и т.п.) | старотатарский → арабица → кириллица | целевая | не опубликована | нет данных | — | нет | нет | создан филологами для того же материала | в 2020 г. заявлено ~70% точности; в открытом доступе нет ни модели, ни API |
| **Transkribus OttomanTurkish_generic** (ID 56496) | PyLaia (CNN+BiLSTM+CTC) | line (через Transkribus) | печать + рукопись | османский турецкий (1500–1928) | арабо-персидское письмо; 25 480 строк / 789 стр. | Transkribus ToS, кредиты | платно (кредиты) | облако | **нет** | только внутри Transkribus | смешанное обучение print+HTR; CER 11.6% (валидация) | весов нет; только платный API metagrapho (Organisation plan); изображения уходят на сервер READ-COOP; формат вывода (арабица/латиница) в карточке не указан |
| **Transkribus OttomanTurkish_Print_1** (ID 52502) | PyLaia | line | печать (периодика к. XIX – н. XX в.) | османский турецкий | арабская графика на входе | Transkribus ToS | платно | облако | **нет** | только внутри Transkribus | CER 7.2% на валидации | **вывод — латинская «half-transcription»**, т.е. не арабица → не подходит для `raw_ocr` |
| **Kraken 7.1.1** (движок) | VGSL CNN-LSTM-CTC; с 7.x также PP-OCRv6 (CTC) и Party (генеративная); BLLA baseline-сегментация | page → lines (BLLA) → line recognizer | оба (зависит от модели) | любые (модели) | полная поддержка RTL, BiDi, NFD | Apache-2.0 | да | CPU достаточно; GPU ускоряет обучение | да (модели на Zenodo/HTRMoPo) | да (`ketos train/finetune`), eScriptorium | стандарт де-факто для исторических арабографических документов (OpenITI, eScriptorium); маленькие модели, быстрый CPU-инференс | качество сильно зависит от сегментации; модели «из коробки» не видели татарскую графику |
| **OpenITI/KITAB OCR** — Printed Arabic-Script Base Model (Zenodo 7050270) | kraken 4.x CNN-LSTM-CTC, 16 MB | line | печать | ar ~23k, fa ~17k, ur ~11k, ota ~7.1k строк (OpenITI, 2022) | широкое (несколько языков/шрифтов) | **CC0-1.0** | да | CPU | да | да (базовая модель для дообучения) | разнообразие гарнитур; лучшая стартовая точка для fine-tuning печати | «not extensively verified or optimized»; GT нормализован в NFD |
| OpenITI Printed **Ottoman** model (Zenodo 7050342) | то же, дообучена на ota | line | печать | османский | османская печать — визуально ближе всего к татарской печати нач. XX в. | **CC0-1.0** | да | CPU | да | да | османская печать ≈ насх казанских/оренбургских типографий | 7 100 строк, без проверки качества |
| OpenITI Printed **Persian** / **Arabic** / **Urdu** models (Zenodo 7051644 / 7050296 / 14585602) | то же | line | печать | fa / ar / ur | — | CC0-1.0 / CC0-1.0 / Apache-2.0 | да | CPU | да | да | контрольные варианты переноса | — |
| **OpenITI MAKHZAN** v2026.1.2 (Zenodo 19861912) — *датасет* | — (eScriptorium GT: строки + полигоны) | page + lines | 1 423 рукописные стр., 74 печатные (урду) | ar 822, fa 343, ota 241, ur 74 стр., ms, jv | ~20 стилей письма (naskh, nastaliq, shikaste, divani …), X–XX вв. | **CC BY-NC-SA 4.0** | **нет** (NC) | — | — (6.1 GB zip) | для обучения | крупнейший открытый арабографический HTR-GT, есть османский | **татарских документов нет** (проверено по метаданным); NC-лицензия ограничивает коммерческий продукт |
| **PaddleOCR 3.7 PP-OCRv5 Arabic** (`arabic_PP-OCRv5_mobile_rec` + `PP-OCRv5_server_det`) | DB-детектор + SVTR-LCNet CTC | box (детектор строк/фрагментов) | печать (современная) | ar, fa, ug, ur, ps, ku, sd, bal, en | хорошее для современной печати; 81.27% accuracy на собственном наборе | **Apache-2.0** | да | CPU работает (oneDNN надо отключить, см. ниже) | да | да (PaddleOCR train tools) | дёшево, быстро, свой детектор, уверенность по строкам | не историческая модель; детектор режет строки на фрагменты; на paddle 3.3 CPU+oneDNN падает (`ConvertPirAttribute2RuntimeAttribute`) |
| **CHURRO-3B** (stanford-oval/churro-3B) | Qwen2.5-VL-3B, дообучена на CHURRO-DS (~100K стр., 155 коллекций, 46 языковых кластеров) | page (выдаёт XML) | оба | 46 кластеров; среди них Arabic, Persian, Turkish (османский) | NLS на CHURRO-DS (handwriting): Arabic 77.5, Persian 78.0, Turkish 69.1 | веса **Qwen Research License** (`qwen-research`), код Apache-2.0, датасет «research use only» | **нет** (некоммерческая) | bf16 ≈ 7.5 GB весов → комфортно ≥ 12 GB VRAM; на CPU — только квантизованный GGUF (llama.cpp) | да (+ community GGUF: mradermacher/churro-3B-GGUF) | да (LoRA/full, Qwen2.5-VL) | единственная открытая VLM, специально обученная на исторических документах и превосходящая Gemini 2.5 Pro на CHURRO-DS | лицензия запрещает коммерцию; галлюцинации/повторы, свойственные VLM; медленно на CPU; квантизация может снижать качество |
| **PP-OCRv6 for kraken** (tiny/small/medium, Zenodo 21788403/05/10) | CTC line recognizer (лёгкий CNN + non-recurrent neck), 128 px высота строки | line (через kraken BLLA) | **оба** | 44 языка / 10 письменностей | Arabic, Persian, Ottoman, Urdu (OpenITI print + рукописи Muharaf, RASAM, Agapet, TariMa…) | **Apache-2.0** | да | CPU (medium 64 MB) | да | да (kraken) | свежая (2026-08) мультиязычная база print+HTR; на собственных тестах CER: Ottoman 5.9%, Persian 4.7%, Arabic 9.4% | тестовые сплиты случайные по строкам (не по документам) → цифры оптимистичны; разнородные конвенции транскрипции |
| **Party** — Pretrained multilingual Party base model (Zenodo 20642057) | Swin encoder + tiny Llama (40M) decoder, octet-токены, промпт позициями строк | page с готовой сегментацией | оба | многоязычная, вкл. арабский | Arabic (OpenITI, Muharaf, RASAM…) | Apache-2.0 | да | 518 MB, GPU желателен | да | да | генеративная полностраничная модель без галлюцинаций «вне строк» | требует отдельный пакет `party` с GitHub; не протестирована в baseline (время/зависимости) |
| **Azra-1-Mini-0.8b** (OttomanNLP) | Qwen3.5-Vision 0.8B, дообучена | **line** | печать (Nesih) + рукопись (Rika) | османский турецкий (вывод арабицей) | целевая для османского | **Apache-2.0** (база Qwen3.5-0.8B — тоже Apache-2.0) | да | CPU возможен (fp32 ≈ 3.4 GB RAM) | да | да | маленькая; заявлено 80.2% символьной точности на Nesih (≈ Gemini 3.1 Pro 82.8%) | только строки (нужна сегментация); метрики авторские; турецкий промпт |
| kranidiotis/ottoman-ocr-qwen2.5-vl-7b | Qwen2.5-VL-7B + LoRA | line/page | оба | османский → **современный турецкий (латиница)** | — | Apache-2.0 (база 7B — Apache-2.0) | да | ≥ 16 GB VRAM | да | да | — | выдаёт латиницу → не подходит для `raw_ocr`; 7B не для CPU |
| **Phoenix** (factlogic/phoenix-arabic-manuscript-htr) | CNN+BiLSTM+CTC, 5M параметров, kraken `.mlmodel` | line | **рукопись** | арабский (рукописи Agapet, Omar, TariMa) | арабские рукописи IX–XVIII вв. | **CC BY-NC-SA 2.0** | **нет** | CPU | да | да (kraken) | специализированная Arabic HTR, крошечная, быстрая | NC-лицензия; обучена на рукописях, а не на печати |
| rakansuliman/arabic-htr-v2 (HAFITH fine-tune) | SigLIP2 NaFlex + TrOCR decoder, 698M | line | рукопись | арабский | рукописи (81 672 строки) | Apache-2.0 | да | CPU/GPU | да | да | сильная Arabic HTR (CER 12.3% у авторов) | требует выполнения кода из репозитория (`hafith_htr`) — не запускали из соображений безопасности |
| sherif1313/Arabic-English-handwritten-OCR-v3 | Qwen2.5-VL-3B fine-tune | page/line | рукопись | ar, en | арабская рукопись | в карточке Apache-2.0, **но база Qwen2.5-VL-3B под Qwen Research License** → лицензия противоречива | не определено (скорее нет) | ≈ 7.5 GB | да | да | заявлен CER 3.8% | лицензионный конфликт; современный почерк, не исторический |
| NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct | Qwen2-VL-2B fine-tune | page | печать (синтетика, 12 шрифтов) | арабский | современная арабская печать | Apache-2.0 (база Apache-2.0) | да | ≈ 4.4 GB | да | да | структура документа, диакритики | обучена на синтетике; не протестирована в baseline |
| mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0 | Qwen3-VL-2B LoRA | line | печать | fa, ar | персидские строки (56k реальных + 100k синтетики) | Apache-2.0 | да | ≈ 4 GB | да | да | — | только строки, не протестирована |
| Surya OCR 0.23 (datalab) | собственная VLM 689M + детектор | page | печать | 90+ языков (Arabic, Persian) | общая | код Apache-2.0, веса modified AI2 OpenRAIL-M (бесплатно при < $5M выручки) | ограниченно | GPU (vLLM) / CPU (llama.cpp) | да | ограниченно | хорошая layout-аналитика | не историческая; условия лицензии весов; не протестирована |
| EasyOCR 1.7.2 | CRAFT + CRNN | box | печать | ar, fa, ur, ug (+80) | современная печать | Apache-2.0 | да | CPU | да | да | простая установка | старая архитектура, нет исторических данных |
| Tesseract 5 (ara, fas, uig) | LSTM | page | печать | 100+ | современная печать | Apache-2.0 | да | CPU | да | да (tesstrain) | стандартный дешёвый baseline | в системе не установлен (нужен `sudo apt`), зарегистрирован как unavailable |
| Коммерческие VLM API (Gemini 3.x, GPT-5.x, Claude) | закрытые VLM | page | оба | многоязычные | на CHURRO-DS Arabic HTR: Gemini 3.1 Pro 87.5 NLS, Gemini 3 Flash 80.3 | ToS провайдера | да (платно) | облако | нет | ограниченно | сильнейшие на арабских рукописях | отправка изображений третьей стороне, стоимость, невоспроизводимость версий; в baseline не включены (нет ключа, приватность) |

## 2. Что из этого реально запускается в baseline

| backend в репозитории | модель | статус на машине baseline (CPU, 13 GB RAM) |
|---|---|---|
| `paddle` | PP-OCRv5 server det + arabic_PP-OCRv5_mobile_rec | работает (oneDNN выключен) |
| `easyocr` | EasyOCR ar+fa | работает |
| `kraken-arabic-script` / `kraken-ottoman` / `kraken-persian` / `kraken-arabic` | OpenITI print models | `kraken-ottoman` проверен на реальной странице; остальные три — тот же адаптер, на бенчмарке ещё не запускались |
| `kraken-ppocrv6` | PP-OCRv6 medium | работает |
| `arabic-htr-phoenix` | Phoenix (Arabic HTR) | работает (NC-лицензия, только исследование) |
| `churro` | CHURRO-3B GGUF Q4_K_M + mmproj Q8_0 (llama.cpp) | адаптер написан, **ещё не запускался**: веса докачиваются (сеть ~0.3–0.9 МБ/с); квантизация = отклонение от эталона |
| `churro-hf` | CHURRO-3B bf16 (transformers) | эталонный путь для GPU; на этой машине не запускался (RAM) |
| `ottoman-azra` | Azra-1-Mini-0.8b + kraken BLLA | адаптер написан, **ещё не запускался**: веса не скачаны |
| `tesseract` | Tesseract ara+fas | **unavailable**: нет системного бинарника |
| `transkribus-ottoman` | OttomanTurkish_generic | **unavailable**: нет весов, платный API, изображения уходят третьей стороне; Print-модель выдаёт латиницу |

Lisan.Tatar и Miras-Tat не имеют API/весов и не могут быть частью воспроизводимого бенчмарка.
Их можно сравнить вручную на тех же страницах, но это означает загрузку изображений на чужой
сервер — решение за владельцем данных.

## 3. Связанные работы по старотатарскому

* Валишин И.А. (КФУ), «Применение методов компьютерного зрения к распознаванию старотатарского
  текста», 2024 (статья CC BY 4.0): YOLO-детекция строк/слов/символов; код, данные и метрики не опубликованы.
* Татар-информ: проект «Распознавание и перевод рукописных текстов на арабской графике» (Татарстан) — только анонс.
* Открытых старотатарских OCR-датасетов с GT **не найдено**. Найдены: (а) многоязычный Wikisource
  с постраничными транскрипциями нескольких книг 1915–1927 гг. (используем как silver GT, см. `docs/dataset.md`);
  (б) Miras.info — сканы >150 дореволюционных книг (без GT).

## 4. Выводы Phase 0 для выбора baseline

1. Ни одна открытая модель не обучалась на татарской арабице. Ближайшие источники переноса — османская
   печать (OpenITI Ottoman, PP-OCRv6, Azra) и персидская/арабская печать.
2. Для коммерчески чистого продукта годятся: kraken + OpenITI (CC0), PP-OCRv6 (Apache-2.0), PaddleOCR,
   Azra (Apache-2.0). CHURRO-3B, Phoenix и MAKHZAN — только исследования (NC/research-лицензии).
3. Transkribus исключается из-за отсутствия весов, платного API и (для Print-модели) латиницы на выходе.

## Источники

* Lisan.Tatar — https://lisan.tatar/
* Miras-Tat — https://www.business-gazeta.ru/article/457892 ; https://www.tatar-inform.ru/news/v-tatarstane-avtomatiziruyut-rabotu-po-perevodu-rukopisnyx-tekstov-na-arabskoi-grafike-5852325
* Валишин 2024 — https://rdl-journal.ru/article/view/852
* Transkribus OttomanTurkish_generic — https://www.transkribus.org/models/ottoman-turkish-generic
* Transkribus OttomanTurkish_Print — https://www.transkribus.org/models/ottoman-turkish-print ; https://blog.transkribus.org/en/can-ai-read-arabic-script-transkribus
* Transkribus metagrapho API — https://www.transkribus.org/metagrapho
* Kraken — https://pypi.org/project/kraken/ (7.1.1, Python ≥3.10,<3.14)
* OpenITI kraken models — https://zenodo.org/records/7050270 , /7050342 , /7051644 , /7050296 , /14585602
* OpenITI MAKHZAN — https://zenodo.org/records/19861912 ; https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.465
* KITAB OCR — https://kitab-project.org/corpus/ocr
* PaddleOCR PP-OCRv5 multilingual — https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md ; https://huggingface.co/PaddlePaddle/arabic_PP-OCRv5_mobile_rec
* CHURRO — https://huggingface.co/stanford-oval/churro-3B ; https://github.com/stanford-oval/Churro (benchmark_results.json, commit 23c3ac6) ; https://huggingface.co/mradermacher/churro-3B-GGUF
* PP-OCRv6 for kraken — https://zenodo.org/records/21788410 ; Party — https://zenodo.org/records/20642057 ; BLLA — https://zenodo.org/records/22879549
* Azra-1-Mini — https://huggingface.co/OttomanNLP/Azra-1-Mini-0.8b
* Phoenix — https://huggingface.co/factlogic/phoenix-arabic-manuscript-htr
* HAFITH fine-tune — https://huggingface.co/rakansuliman/arabic-htr-v2
* sherif1313 v3 — https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3 ; Qwen2.5-VL-3B licence — https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
* Qari-OCR — https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct
* Ottoman Qwen 7B — https://huggingface.co/kranidiotis/ottoman-ocr-qwen2.5-vl-7b
* Surya — https://pypi.org/project/surya-ocr/
* EasyOCR — https://pypi.org/project/easyocr/
