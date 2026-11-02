# Данные других арабографических языков для pretraining / transfer (Phase 2–3)

Проверено 2026-11-02 по репозиториям и карточкам датасетов. Правило ТЗ: эти данные — только для обучения;
**validation и test — только старотатарские реальные документы**.

| датасет | язык / письмо | объём | тип | лицензия | коммерческое использование | применимость |
|---|---|---|---|---|---|---|
| OpenITI MAKHZAN v2026.1.2 (Zenodo 19861912) | ar, fa, ota, ur, ms, jv | ~1 500 стр., 208 документов (1 423 рукописных, 74 печатных урду) | рукописи + печать урду | CC BY-NC-SA 4.0 | нет | HTR-pretraining (османские рукописи — 241 стр.); NC |
| OpenITI arabic_print_data (GitHub) | ar, fa, ur, ota (печать) | основа моделей OpenITI (ar ~23k, fa ~17k, ur ~11k, ota ~7.1k строк) | печать | **лицензия в репозитории не указана** | неясно → не использовать без согласования | лучший источник османской/персидской печати; нужно запросить лицензию у OpenITI |
| RASAM 1/2 (calfa-co/rasam-dataset) | арабский (магрибское письмо), берберский | 11 290 строк | рукописи | Apache-2.0 | да | HTR-pretraining; письмо далеко от татарского |
| TariMa (calfa-co/tarima) | арабский магрибский | — | рукописи | Apache-2.0 | да | то же |
| Muharaf (public part) | арабский (архивные документы XIX–XXI вв.) | 1 216 стр. (из 1 644), 36k строк всего | рукописи | CC BY-NC-SA 4.0 (публичная часть); закрытая часть — проприетарная | нет | HTR-pretraining, близкий по эпохе почерк |
| AraMS-28k (arXiv 2608.26921) | арабский (насх, рукъа, магриби + 1 литография) | 28 600 строк, 3 043 стр., 14 книг | рукописи | CC BY-NC-SA 4.0 | нет | HTR-pretraining, рукъа близка к поздним тюркским рукописям |
| KHATT (LDC2015T23) | арабский (современный почерк, 1 000 писцов) | — | рукописи | LDC user agreement, некоммерческое, без распространения | нет | слабая релевантность (современный почерк) |
| Wikimedia Commons «Tatar-language books» | **старотатарский** (иске имля) | 30 книг 1893–1917, ~1 900 стр. | печать | Public Domain | да | основной источник целевых данных Phase 2 (`data/candidates/commons_tatar.csv`) |

## Выводы
* **Коммерчески чистый pretraining** сейчас возможен только на RASAM/TariMa (Apache-2.0, но магрибское письмо)
  и на собственной разметке Public-Domain татарских книг. Османская/персидская печать OpenITI — самая близкая
  к татарской — требует уточнения лицензии у OpenITI.
* MAKHZAN, Muharaf, AraMS-28k — только для исследовательских моделей (NC).
* Готовые модели (OpenITI kraken CC0, PP-OCRv6 Apache-2.0) уже содержат этот перенос — дообучать их на
  старотатарской разметке дешевле, чем повторять pretraining.

## Синтетика (отложено)
Синтетический рендер строк допустим только в train. Блокер: нужен старотатарский текст арабицей, **не** взятый
из тестовых страниц. Кандидаты: (1) расшифровки train-источников после разметки; (2) обратная транслитерация
современного татарского корпуса в иске имля (зависит от Phase 4, даёт неестественную орфографию).
Шрифты — только OFL (Amiri, Scheherazade New, Noto Naskh Arabic). Решение — после первого fine-tuning-эксперимента.

## Источники
* OpenITI MAKHZAN — https://zenodo.org/records/19861912
* OpenITI arabic_print_data — https://github.com/OpenITI/arabic_print_data (поле license пустое, в README лицензии нет)
* RASAM — https://github.com/calfa-co/rasam-dataset ; TariMa — https://github.com/calfa-co/tarima (Apache-2.0)
* Muharaf — https://github.com/MehreenMehreen/muharaf ; https://www.researchgate.net/publication/381471365
* AraMS-28k — https://arxiv.org/html/2608.26921
* KHATT — https://catalog.ldc.upenn.edu/LDC2015T23
* Commons — https://commons.wikimedia.org/wiki/Category:Tatar-language_books
