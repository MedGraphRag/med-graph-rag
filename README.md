# MedGraphRAG

GraphRAG по клиническим рекомендациям Минздрава РФ: сопоставление лекарств с МНН/АТХ и проверка связей противопоказаний и взаимодействий.

## Установка

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ГРЛС

Загрузить выборку Д/П с проверкой покрытия и обработать последний snapshot:

```bash
python -m src.nlp.grls_active --delay 1
python -m src.nlp.grls_etl --repair-mojibake --mapped-only
```

- `data/raw/grls/` — исходные JSON и manifest.
- `data/processed/grls_normalized.parquet` — заполненные связи: `trade_name_norm`, `inn_norm`, `atx_code` (список кодов).
- `data/processed/grls_normalized_records.parquet` — регистрационные записи и происхождение.
- `data/processed/grls_normalized_report.json` — статистика обработки.

Полнота проверяется для Д/П относительно API. Исходные JSON не изменяются; выгрузки и локальная конфигурация не включаются в Git.

## PDF клинических рекомендаций

Скачать PDF по сохранённому манифесту (для пробного запуска — первые три):

```bash
python -m src.nlp.downloader --limit 3
python -m src.nlp.downloader
```

По умолчанию используется `data/processed/cr_manifest.json`, файлы сохраняются в
`data/raw/guidelines/pdf/<CodeVersion>.pdf`. API ожидает `CodeVersion` (например,
`1078_1`), а не внутренний `Id` записи. Допускается манифест с полями `Code` и
`Version`. Уже скачанные PDF пропускаются после проверки сигнатуры и конца файла.
Ответы HTML, в том числе ошибки IIS с HTTP 200, не сохраняются как PDF.
При ошибках процесс возвращает ненулевой код; повторный запуск доскачивает оставшееся.
Пути и паузу можно задать через `--manifest`, `--output-dir`, `--delay`.

### Извлечение текста и таблиц

Запуск из корня проекта после скачивания PDF:

```bash
python -m src.nlp.cr_pipeline --workers 4 --tessdata data/models/tessdata
```

`--workers` задаёт число процессов (по умолчанию 1, минимум 1). Каждый процесс
обрабатывает PDF целиком; главный процесс собирает результаты и записывает Parquet.
Пул создаётся, когда есть несколько документов и `--workers` больше 1.

Страницы с текстовым слоем обрабатываются напрямую. Для страниц без текста
автоматически выполняется OCR через встроенный в PyMuPDF Tesseract: `rus+eng`,
300 DPI. Исходные PDF не изменяются. Нужны языковые модели; отдельный Python-пакет
для OCR не требуется. Скачать модели из официального репозитория Tesseract:

```bash
mkdir -p data/models/tessdata
curl -fL https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/rus.traineddata -o data/models/tessdata/rus.traineddata
curl -fL https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata -o data/models/tessdata/eng.traineddata
```

Папку моделей можно указать через `--tessdata` или переменную `TESSDATA_PREFIX`.
Без этих настроек PyMuPDF попробует найти установленный Tesseract.
`--no-ocr` отключает распознавание и позволяет запускать извлечение без моделей.
Подробности: [OCR в PyMuPDF](https://pymupdf.readthedocs.io/en/latest/faq.html).

Результаты в `data/processed/` (повторный запуск перезаписывает файлы):

- `cr_pages.parquet`: одна строка на страницу, текст в `text`; `text_source` —
  `native`, `ocr` или `empty`; `ocr_attempted` и `ocr_error` — попытка OCR и её ошибка.
- `cr_extraction_report.parquet`: статус документа, число страниц и пустых страниц
  после OCR; `ocr_pages` — число попыток OCR (включая пустые результаты и ошибки),
  `ocr_failed_pages` — число ошибок, `ocr_messages` — сообщения с номерами страниц.
  Статус `partial` означает, что текст получен частично и есть ошибки OCR;
  `error` — текст не получен из-за ошибки, `empty` — текста нет без ошибок OCR.
  Пустой результат OCR сам по себе не считается ошибкой.
- `cr_tables.parquet`: таблицы и причины, по которым их не удалось извлечь.

OCR запускается только на полностью лишённых текста страницах: сканы на странице
с уже существующим текстом не распознаются. OCR не сохраняет надёжно жирное
начертание, поэтому текущий поиск заголовков на сканах ограничен. Извлечение
структуры таблиц из изображений не реализовано; найденные OCR подписи могут
попасть в отчёт таблиц со статусом `unprocessed`. Ошибки распознавания возможны,
в том числе в числах и дозировках; важные фрагменты нужно сверять с PDF.

## Тесты

```bash
python -m pytest -q
```

Основные тесты используют мокированный HTTP. Архитектура и правила разработки — в [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).
