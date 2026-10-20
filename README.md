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

## Тесты

```bash
python -m pytest -q
```

Основные тесты используют мокированный HTTP. Архитектура и правила разработки — в [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).
