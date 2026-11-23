"""Пробное извлечение клинических связей из одной подготовленной страницы КР."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter

import polars as pl

from src.nlp.clinical_extractor import extract_relations
from src.nlp.clinical_llm import extraction_client
from src.nlp.clinical_relations import ClinicalFragment
from src.nlp.clinical_resolver import resolve_relation_drugs
from src.nlp.grls_resolver import DrugResolver

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"


def read_fragment(
    path: Path,
    document: str | None = None,
    page: int | None = None,
    *,
    shortest: bool = False,
) -> ClinicalFragment:
    """Выбирает одну страницу, не изменяя и не сокращая её текст."""
    if page is not None and not document:
        raise ValueError("Для выбора страницы укажите документ через --document")
    if shortest and page is not None:
        raise ValueError("--shortest нельзя сочетать с --page")

    pages = pl.read_parquet(path)
    required = {"code_version", "page_number", "text"}
    missing = required - set(pages.columns)
    if missing:
        raise ValueError("В таблице страниц нет полей: " + ", ".join(sorted(missing)))

    selected = pages
    if document:
        selected = selected.filter(pl.col("code_version") == document)
    if page is not None:
        selected = selected.filter(pl.col("page_number") == page)
    elif not shortest:
        # Это только выборка для пробы, а не поиск всех связей документа.
        selected = selected.filter(
            pl.col("text").str.contains(r"(?i)противопоказ|взаимодейств")
        )

    if shortest:
        selected = selected.filter(pl.col("text").str.contains(r"\S"))
        if selected.is_empty():
            raise ValueError("Непустая страница не найдена. Проверьте --document и --pages.")
        selected = selected.with_columns(
            pl.col("text").str.len_chars().alias("_page_length")
        ).sort(["_page_length", "code_version", "page_number"])
    else:
        selected = selected.sort(["code_version", "page_number"])

    if selected.is_empty():
        raise ValueError(
            "Подходящая страница не найдена. Проверьте --document и --page. "
            "Без --page выбираются страницы со словами о противопоказаниях "
            "или взаимодействиях."
        )

    row = selected.row(0, named=True)
    duplicates = pages.filter(
        (pl.col("code_version") == row["code_version"])
        & (pl.col("page_number") == row["page_number"])
    )
    if duplicates.height != 1:
        raise ValueError("Для выбранного документа и страницы найдено несколько записей")

    location = f"Страница {row['page_number']}"
    section = row.get("section_title")
    if section and section.strip():
        location += ", " + section.strip()

    return ClinicalFragment(
        document_id=row["code_version"],
        location=location,
        page_number=row["page_number"],
        text=row["text"],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DATA_DIR / "cr_pages.parquet")
    parser.add_argument("--dictionary", type=Path, default=DATA_DIR / "grls_normalized.parquet")
    parser.add_argument("--document", help="CodeVersion документа, например 115_3")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--page", type=int, help="Номер страницы PDF, начиная с 1")
    selection.add_argument("--shortest", action="store_true", help="Самая короткая непустая страница")
    parser.add_argument("--output", type=Path, help="Новый JSON-файл для проверки результатов")
    args = parser.parse_args(argv)
    if args.page is not None and (args.page < 1 or not args.document):
        parser.error("--page должен быть >= 1 и требует --document")

    fragment = read_fragment(args.pages, args.document, args.page, shortest=args.shortest)
    resolver = DrugResolver(args.dictionary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or DATA_DIR / "clinical_trials" / f"trial_{timestamp}.json"
    if output.exists():
        raise FileExistsError("Файл результата уже существует; укажите новый --output")

    print(f"Документ: {fragment.document_id}; {fragment.location}", flush=True)
    print(f"Длина страницы: {len(fragment.text)} символов", flush=True)
    with extraction_client() as (client, model):
        print("Отправляю страницу модели; ожидаю JSON с утверждениями...", flush=True)
        extraction_started = perf_counter()
        try:
            result = extract_relations(
                fragment, client=client, model=model,
            )
        finally:
            extraction_seconds = perf_counter() - extraction_started
            print(f"Время извлечения связей: {extraction_seconds:.2f} с", flush=True)

    print("Ответ модели получен; сопоставляю названия с ГРЛС...", flush=True)
    resolution_started = perf_counter()
    resolved = [resolve_relation_drugs(relation, resolver) for relation in result.relations]
    resolution_seconds = perf_counter() - resolution_started
    print(f"Время сопоставления с ГРЛС: {resolution_seconds:.2f} с", flush=True)
    report = {
        "model": model,
        "reasoning_effort": os.getenv("LLM_REASONING_EFFORT", "").strip() or None,
        "timings_seconds": {
            "extraction": round(extraction_seconds, 3),
            "drug_resolution": round(resolution_seconds, 3),
        },
        "manual_review": "not_requested",
        "fragment": fragment.model_dump(),
        "relations": [item.model_dump() for item in resolved],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    quotes_found = sum(
        item.relation.source.quote in fragment.text for item in resolved
    )
    print(f"Извлечено утверждений: {len(resolved)}")
    print(f"Цитаты найдены в тексте: {quotes_found}/{len(resolved)}")
    statuses = Counter(item.relation.status for item in resolved)
    print(f"Статусы: {dict(statuses)}")
    print(f"Результат: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
