# data/

Полное описание стандарта и текущего бенчмарка: [`docs/dataset.md`](../docs/dataset.md).

Кратко:

* `benchmark_manifest.csv` — страницы бенчмарка (пути относительно этой папки). Проверка:
  `python -m benchmark.manifest validate data/benchmark_manifest.csv`.
* `ground_truth/` — GT (арабица), параллельная кириллица (`*.modern_tatar.txt`) и происхождение (`*.json`).
  Под git, история изменений GT — история git. Старый GT не перезаписывается без коммита с объяснением.
* `raw/` — изображения страниц, **не хранятся в git**. Восстановить:

  ```bash
  python scripts/fetch_wikisource_benchmark.py
  ```

  Скрипт использует закреплённые ревизии из `wikisource_lock.json`; `--update-lock` берёт последние правки
  Wikisource (меняет версию датасета — хэш пишется в `config.yaml` каждого эксперимента).
* `sources.yaml` — метаданные источников (год, место, издатель, орфография, тип, лицензии, сплит).

Лицензии: изображения — Public Domain (Wikimedia Commons); транскрипции — CC BY-SA 4.0
(участники Wikisource); музейное фото 1912 г. — CC BY-SA 4.0 (Vyacheslav Kirillin).
