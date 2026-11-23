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

Проверить и загрузить уникальный normalized-справочник в Neo4j:

```bash
python -m src.graph.grls_loader --dry-run
python -m src.graph.grls_loader --batch-size 1000
```

Подключение использует `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` из локального
`.env`. Повторный импорт не создаёт дубликаты; МНН и АТХ привязаны к уникальной
связи справочника, без неподтверждённых пар между кодами. Одна связь из поискового
Parquet соответствует одному `GRLSDrug`. Подробные регистрационные источники
импортируются отдельно с `--records`; просмотр источников — `search_records`.
Схема, происхождение и поиск — в [документации графа ГРЛС](docs/grls_graph.md).

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

## Извлечение клинических утверждений

Извлечение утверждений через Instructor и Pydantic, проверка цитат
и сопоставление лекарств с ГРЛС описаны в
[clinical_extraction.md](docs/clinical_extraction.md).

Извлечение по умолчанию работает через Ollama с `gemma4:12b`.
Загрузка модели: `ollama pull gemma4:12b`; настройки — в `.env.example`.
Оценка ответов на размеченной выборке: `python -m src.nlp.clinical_evaluation`.

Последовательный батч по подготовленным страницам:

```bash
python -m src.nlp.clinical_batch --dry-run
python -m src.nlp.clinical_batch --limit 5 --output-dir data/processed/clinical_batches/first
python -m src.nlp.clinical_batch --output-dir data/processed/clinical_batches/full
```

Продолжение запуска: добавьте `--resume` с тем же `--output-dir` и теми же
параметрами выборки; для повторения ошибочных страниц добавьте `--retry-errors`.
Результаты страниц и сводка сохраняются по ходу обработки.
Формат файлов и ограничения описаны в [clinical_extraction.md](docs/clinical_extraction.md).

Небольшая выборка для ручной проверки описана в
[clinical_reference.md](docs/clinical_reference.md).

## Тесты

```bash
python -m pytest -q
```

Основные тесты используют мокированный HTTP. Архитектура и правила разработки — в [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).
