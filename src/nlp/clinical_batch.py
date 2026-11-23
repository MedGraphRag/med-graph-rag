"""Последовательное извлечение утверждений из подготовленных страниц КР."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Literal

import polars as pl
from pydantic import Field

from src.nlp.clinical_extractor import extract_relations
from src.nlp.clinical_llm import extraction_client
from src.nlp.clinical_relations import ClinicalFragment, ExtractionModel, NonBlankText
from src.nlp.clinical_resolver import ResolvedClinicalRelation, resolve_relation_drugs
from src.nlp.grls_resolver import DrugResolver

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class BatchInputError(ValueError):
    """Ошибка параметров или контрольных точек, которую можно вывести без секретов."""


class PreparedPage(ExtractionModel):
    code_version: NonBlankText
    page_number: int = Field(ge=1, strict=True)
    text: str | None
    section_title: str | None = None

    def fragment(self) -> ClinicalFragment:
        location = f"Страница {self.page_number}"
        if self.section_title and self.section_title.strip():
            location += ", " + self.section_title.strip()
        return ClinicalFragment(
            document_id=self.code_version, page_number=self.page_number,
            location=location, text=self.text,
        )


class PageResult(ExtractionModel):
    fragment: ClinicalFragment
    relations: list[ResolvedClinicalRelation]
    manual_review: Literal["not_requested"] = "not_requested"


def read_pages(path: Path, documents: list[str] | None = None,
               limit: int | None = None) -> list[PreparedPage]:
    """Проверяет выбранные страницы до первого запроса; сохраняет текст целиком."""
    if limit is not None and limit < 1:
        raise BatchInputError("--limit должен быть >= 1")
    table = pl.read_parquet(path)
    required = {"code_version", "page_number", "text"}
    missing = required - set(table.columns)
    if missing:
        raise BatchInputError("В таблице страниц нет полей: " + ", ".join(sorted(missing)))
    if documents:
        unknown = set(documents) - set(table["code_version"].to_list())
        if unknown:
            raise BatchInputError("Документы не найдены: " + ", ".join(sorted(unknown)))
        table = table.filter(pl.col("code_version").is_in(documents))
    columns = sorted(required | ({"section_title"} & set(table.columns)))
    pages = [PreparedPage.model_validate(row) for row in table.select(columns).iter_rows(named=True)]
    keys = [(page.code_version, page.page_number) for page in pages]
    if len(keys) != len(set(keys)):
        raise BatchInputError("Для документа и страницы найдено несколько записей")
    if not pages:
        raise BatchInputError("Таблица выбранных страниц пуста")
    return sorted(pages, key=lambda page: (page.code_version, page.page_number))[:limit]


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_filename(page: PreparedPage) -> str:
    # Идентификатор документа никогда не становится частью пути.
    digest = hashlib.sha256(page.code_version.encode("utf-8")).hexdigest()
    return f"{digest}_{page.page_number}.json"


def write_json(path: Path, value: dict) -> None:
    """Публикует только полностью записанный файл, включая контрольные точки."""
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def input_signature(pages: list[PreparedPage], dictionary: Path) -> dict:
    encoded = json.dumps([page.model_dump() for page in pages], ensure_ascii=False,
                         sort_keys=True).encode("utf-8")
    # Изменившиеся проверки/промпт требуют нового запуска, а не смешения результатов.
    modules = ["clinical_batch", "clinical_extractor", "clinical_llm", "clinical_relations",
               "clinical_validation", "clinical_resolver", "grls_resolver"]
    return {
        "pages_sha256": hashlib.sha256(encoded).hexdigest(),
        "dictionary_sha256": digest_file(dictionary),
        "pipeline_sha256": {name: digest_file(Path(__file__).with_name(name + ".py"))
                            for name in modules},
    }


def read_checkpoint(path: Path, page: PreparedPage) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("page") != page.model_dump():
        raise BatchInputError(f"Контрольная точка не соответствует странице: {path.name}")
    status = record.get("status")
    if status == "completed":
        result = PageResult.model_validate(record.get("result"))
        if result.fragment != page.fragment():
            raise BatchInputError(f"Изменён источник контрольной точки: {path.name}")
        record["result"] = result.model_dump()
    elif status == "skipped_empty":
        if page.text and page.text.strip():
            raise BatchInputError("Непустая страница отмечена как пустая")
    elif status == "error":
        if not isinstance(record.get("error_type"), str):
            raise BatchInputError("В контрольной точке нет типа ошибки")
    else:
        raise BatchInputError(f"Неизвестный статус контрольной точки: {path.name}")
    return record


def make_summary(pages: list[PreparedPage], records: dict, *, interrupted=False) -> dict:
    counts = Counter(record["status"] for record in records.values())
    statuses = Counter()
    relation_count = 0
    for record in records.values():
        if record["status"] == "completed":
            for item in record["result"]["relations"]:
                relation_count += 1
                statuses[item["relation"]["status"]] += 1
    return {
        "total_pages": len(pages), "completed_pages": counts["completed"],
        "empty_pages": counts["skipped_empty"], "error_pages": counts["error"],
        "pending_pages": len(pages) - len(records),
        "relations": relation_count, "relation_statuses": dict(statuses),
        "interrupted": interrupted, "manual_review": "not_requested",
    }


def run_batch(pages: list[PreparedPage], dictionary: Path, output: Path,
              *, resume=False, retry_errors=False) -> int:
    if retry_errors and not resume:
        raise BatchInputError("--retry-errors требует --resume")
    signature = input_signature(pages, dictionary)
    manifest_path = output / "manifest.json"
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                or manifest.get("input") != signature):
            raise BatchInputError("Входные данные или код изменились; укажите новый --output-dir")
        if ("model" not in manifest
                or (manifest["model"] is not None and not isinstance(manifest["model"], str))):
            raise BatchInputError("Повреждён manifest.json: отсутствует корректная модель")
    else:
        # Проверяем справочник до создания каталога результата.
        resolver = DrugResolver(dictionary)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": 1, "input": signature, "model": None,
                    "created_at": datetime.now(timezone.utc).isoformat()}
        write_json(manifest_path, manifest)
    checkpoints = output / "pages"
    checkpoints.mkdir(exist_ok=True)
    records = {}
    for page in pages:
        name = page_filename(page)
        record = read_checkpoint(checkpoints / name, page)
        if record is not None:
            records[name] = record
    pending = [page for page in pages if page_filename(page) not in records
               or (retry_errors and records[page_filename(page)]["status"] == "error")]
    interrupted = False
    summary_path = output / "summary.json"
    write_json(summary_path, make_summary(pages, records))
    try:
        # Пустые страницы фиксируются без открытия клиента LLM.
        nonempty = []
        for page in pending:
            if page.text and page.text.strip():
                nonempty.append(page)
            else:
                record = {"page": page.model_dump(), "status": "skipped_empty"}
                write_json(checkpoints / page_filename(page), record)
                records[page_filename(page)] = record
        if nonempty:
            if resume:
                resolver = DrugResolver(dictionary)
            with extraction_client() as (client, model):
                if manifest["model"] is not None and manifest["model"] != model:
                    raise BatchInputError("Модель изменилась; укажите новый --output-dir")
                reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip() or None
                if (manifest["model"] is not None
                        and manifest.get("reasoning_effort") != reasoning_effort):
                    raise BatchInputError("Режим reasoning изменился; укажите новый --output-dir")
                manifest["model"] = model
                manifest["reasoning_effort"] = reasoning_effort
                write_json(manifest_path, manifest)
                for index, page in enumerate(nonempty, 1):
                    print(f"[{index}/{len(nonempty)}] {page.code_version}, "
                          f"страница {page.page_number}", flush=True)
                    record = {"page": page.model_dump()}
                    try:
                        fragment = page.fragment()
                        result = extract_relations(fragment, client=client, model=model)
                        resolved = [resolve_relation_drugs(item, resolver) for item in result.relations]
                        record.update(status="completed", result=PageResult(
                            fragment=fragment, relations=resolved,
                        ).model_dump())
                    except Exception as error:
                        # Текст исключения SDK может содержать ключи и тело ответа.
                        record.update(status="error", error_type=type(error).__name__)
                        print(f"Ошибка: {type(error).__name__}; продолжаю", file=sys.stderr, flush=True)
                    # Ошибки записи прерывают запуск: результат нельзя считать сохранённым.
                    write_json(checkpoints / page_filename(page), record)
                    records[page_filename(page)] = record
                    write_json(summary_path, make_summary(pages, records))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        summary = make_summary(pages, records, interrupted=interrupted)
        write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Результаты: {output}", flush=True)
    return 130 if interrupted else int(summary["error_pages"] > 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DATA_DIR / "cr_pages.parquet")
    parser.add_argument("--dictionary", type=Path, default=DATA_DIR / "grls_normalized.parquet")
    parser.add_argument("--document", action="append", help="CodeVersion; можно повторять")
    parser.add_argument("--limit", type=int, help="Максимум страниц, включая пустые")
    parser.add_argument("--output-dir", type=Path, help="Новый каталог результата")
    parser.add_argument("--resume", action="store_true", help="Продолжить существующий --output-dir")
    parser.add_argument("--retry-errors", action="store_true", help="Повторить ошибки при --resume")
    parser.add_argument("--dry-run", action="store_true", help="Проверить страницы без LLM и записи файлов")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit должен быть >= 1")
    if args.resume and args.output_dir is None:
        parser.error("--resume требует --output-dir")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors требует --resume")
    try:
        pages = read_pages(args.pages, args.document, args.limit)
        if args.dry_run:
            print(json.dumps({"total_pages": len(pages),
                              "empty_pages": sum(not page.text or not page.text.strip() for page in pages),
                              "documents": sorted({page.code_version for page in pages})}, ensure_ascii=False))
            return 0
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = args.output_dir or DATA_DIR / "clinical_batches" / f"batch_{timestamp}"
        return run_batch(pages, args.dictionary, output,
                         resume=args.resume, retry_errors=args.retry_errors)
    except BatchInputError as error:
        print(f"Батч не выполнен: {error}", file=sys.stderr)
        return 1
    except FileExistsError:
        print("Каталог результата уже существует; используйте --resume "
              "или новый --output-dir.", file=sys.stderr)
        return 1
    except (ValueError, OSError) as error:
        print(f"Батч не выполнен ({type(error).__name__}). "
              "Проверьте входные файлы, параметры и каталог результата.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
