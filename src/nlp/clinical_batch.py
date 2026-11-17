"""Последовательное извлечение связей из всех подготовленных страниц КР."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

from dotenv import load_dotenv
import polars as pl

from src.nlp.clinical_trial import BASE_DIR, DATA_DIR, main as run_page


def check_saved_page(path: Path, row: dict, model: str) -> None:
    """Не пропускает повреждённый файл или результат для другого входа."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    fragment = saved.get("fragment", {})
    location = f"Страница {row['page_number']}"
    section = row.get("section_title")
    if section and section.strip():
        location += ", " + section.strip()
    if (
        saved.get("model") != model
        or fragment.get("document_id") != row["code_version"]
        or fragment.get("page_number") != row["page_number"]
        or fragment.get("location") != location
        or fragment.get("text") != row["text"]
        or not isinstance(saved.get("relations"), list)
    ):
        raise ValueError(f"Результат {path} не соответствует входным данным; укажите новый --output-dir")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DATA_DIR / "cr_pages.parquet")
    parser.add_argument("--dictionary", type=Path, default=DATA_DIR / "grls_normalized.parquet")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "clinical_trials" / "batch")
    parser.add_argument("--limit", type=int, help="Максимум новых страниц за этот запуск")
    parser.add_argument("--dry-run", action="store_true", help="Показать объём без запросов к модели")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit должен быть >= 1")

    pages = pl.read_parquet(args.pages)
    missing = {"code_version", "page_number", "text"} - set(pages.columns)
    if missing:
        parser.error("В таблице страниц нет полей: " + ", ".join(sorted(missing)))
    if pages.select("code_version", "page_number").is_duplicated().any():
        parser.error("В таблице повторяются документ и номер страницы")
    pages = pages.filter(pl.col("text").str.strip_chars().is_not_null()
                         & (pl.col("text").str.strip_chars() != ""))
    pages = pages.sort("code_version", "page_number")
    print(f"Документов: {pages['code_version'].n_unique()}; непустых страниц: {pages.height}", flush=True)
    if args.dry_run:
        return 0

    load_dotenv(BASE_DIR / ".env")
    missing_env = [name for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
                   if not os.getenv(name, "").strip()]
    if missing_env:
        parser.error("Заполните в .env: " + ", ".join(missing_env))

    completed = 0
    skipped = 0
    for row in pages.iter_rows(named=True):
        document = row["code_version"]
        page = row["page_number"]
        safe_document = re.sub(r"[^a-zA-Z0-9_-]", "_", document)
        output = args.output_dir / f"{safe_document}_page_{page}.json"
        try:
            if output.exists():
                check_saved_page(output, row, os.environ["LLM_MODEL"])
                skipped += 1
                continue
            if completed:
                time.sleep(3)
            print(f"Обрабатываю: {document}, страница {page}", flush=True)
            code = run_page([
                "--pages", str(args.pages), "--dictionary", str(args.dictionary),
                "--document", document, "--page", str(page), "--output", str(output),
            ])
            if code != 0:
                return code
        except Exception as error:
            print(f"Остановлено: {document}, страница {page}: {error}", file=sys.stderr)
            print("Готовые результаты сохранены. Повторите ту же команду для продолжения.", file=sys.stderr)
            return 1
        completed += 1
        if args.limit is not None and completed >= args.limit:
            break

    print(f"Новых страниц обработано: {completed}; готовых пропущено: {skipped}")
    print(f"Результаты: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
