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

## Тесты

```bash
python -m pytest -q
```

Основные тесты используют мокированный HTTP. Архитектура и правила разработки — в [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).
