import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import polars as pl

from src.nlp.cr_extractor import (
    EmptyPDFError,
    extract_section_headings,
    extract_tables,
    read_pdf_pages,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "cr_manifest.json"
PDF_DIR = BASE_DIR / "data" / "raw" / "guidelines" / "pdf"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "cr_pages.parquet"
REPORT_PATH = BASE_DIR / "data" / "processed" / "cr_extraction_report.parquet"
TABLES_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "cr_tables.parquet"

SOURCE_URL = "https://cr.minzdrav.gov.ru/view-cr/{code_version}"


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "records",
            "items",
            "Data",
            "data",
            "list",
            "clinical_recommendations",
        ):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError("Манифест не содержит списка документов")


def _extract_document(
    job: tuple[dict, str, Path],
    *, ocr: bool = True, tessdata: str | None = None,
) -> tuple[list[dict], dict, list[dict]]:
    record, code_version, pdf_path = job
    pages_output = []
    tables_output = []

    report_row = {
        "code_version": code_version,
        "name": record.get("Name"),
        "pdf_path": str(pdf_path),
        "status": "error",
        "page_count": 0,
        "empty_pages": 0,
        "ocr_pages": 0,
        "ocr_failed_pages": 0,
        "ocr_messages": [],
        "pdf_repaired": None,
        "message": None,
        "table_status": "not_attempted",
        "table_message": None,
    }

    try:
        pages, repaired = read_pdf_pages(pdf_path, ocr=ocr, tessdata=tessdata)
        empty_pages = sum(not page["text"] for page in pages)
        ocr_messages = [
            f"Page {page['page']}: {page['ocr_error']}"
            for page in pages if page["ocr_error"] is not None
        ]
        has_text = bool(pages) and empty_pages < len(pages)
        status = "ok" if has_text else "empty"
        if ocr_messages:
            status = "partial" if has_text else "error"

        report_row.update(
            status=status,
            page_count=len(pages),
            empty_pages=empty_pages,
            ocr_pages=sum(page["ocr_attempted"] for page in pages),
            ocr_failed_pages=len(ocr_messages),
            ocr_messages=ocr_messages,
            pdf_repaired=repaired,
            message=(
                f"OCR failed on {len(ocr_messages)} pages; see ocr_messages"
                if ocr_messages else None if pages else "PDF contains no pages"
            ),
        )

        current_section = None

        for page in pages:
            headings = extract_section_headings(
                page["formatted_text"]
            )

            if headings:
                current_section = headings[-1]

            pages_output.append(
                {
                    "document_id": record.get("Id"),
                    "code": record.get("Code"),
                    "version": record.get("Version"),
                    "code_version": code_version,
                    "title": record.get("Name"),
                    "source_url": SOURCE_URL.format(
                        code_version=code_version
                    ),
                    "page_number": page["page"],
                    "section_title": current_section,
                    "section_headings": headings,
                    "text": page["text"],
                    "text_source": page["text_source"],
                    "ocr_attempted": page["ocr_attempted"],
                    "ocr_error": page["ocr_error"],
                }
            )

        if not pages:
            return pages_output, report_row, tables_output

        try:
            tables = extract_tables(pdf_path, pages=pages)
            if not tables:
                report_row["table_status"] = "no_tables_detected"
            elif any(table["status"] == "unprocessed" for table in tables):
                report_row["table_status"] = (
                    "partial" if any(table["status"] == "extracted" for table in tables)
                    else "unprocessed"
                )
            else:
                report_row["table_status"] = "extracted"

            for table in tables:
                tables_output.append(
                    {
                        "document_id": record.get("Id"),
                        "code_version": code_version,
                        "page_number": table["page"],
                        "table_index": table["table_index"],
                        "caption": table["caption"],
                        "rows": (
                            json.dumps(
                                table["rows"],
                                ensure_ascii=False,
                            )
                            if table["rows"] is not None
                            else None
                        ),
                        "status": table["status"],
                        "message": table["message"],
                    }
                )

        except (OSError, ValueError, RuntimeError) as error:
            report_row.update(table_status="error", table_message=str(error))
            logger.warning(
                "%s: failed to extract tables: %s",
                code_version,
                error,
            )

    except EmptyPDFError as error:
        report_row.update(status="empty", message=str(error))

    except (OSError, ValueError, RuntimeError) as error:
        report_row["status"] = (
            "missing_pdf" if not pdf_path.exists() else "error"
        )
        report_row["message"] = str(error)

        logger.warning("%s: %s", code_version, error)

    return pages_output, report_row, tables_output


def run_pipeline(
    manifest_path: Path = MANIFEST_PATH,
    pdf_dir: Path = PDF_DIR,
    output_path: Path = OUTPUT_PATH,
    report_path: Path = REPORT_PATH,
    table_output_path: Path = TABLES_OUTPUT_PATH,
    *,
    workers: int = 1,
    ocr: bool = True,
    tessdata: str | None = None,
) -> tuple[int, int]:
    if workers < 1:
        raise ValueError("workers must be at least 1")

    destinations = {path.resolve() for path in (output_path, report_path, table_output_path)}
    raw_directory = pdf_dir.resolve()
    if len(destinations) != 3 or any(
        path == manifest_path.resolve() or path.is_relative_to(raw_directory)
        for path in destinations
    ):
        raise ValueError("Parquet outputs must be distinct and outside the input directory")

    records = load_manifest(manifest_path)

    pages_output = []
    report_output = []
    tables_output = []

    seen_versions = set()
    jobs = []

    for record in records:
        code_version = record.get("CodeVersion")

        if not code_version:
            if (
                record.get("Code") is not None
                and record.get("Version") is not None
            ):
                code_version = f"{record['Code']}_{record['Version']}"
            else:
                continue

        if code_version in seen_versions:
            continue

        seen_versions.add(code_version)

        pdf_path = pdf_dir / f"{code_version}.pdf"
        jobs.append((record, code_version, pdf_path))

    pool_context = (
        ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            mp_context=get_context("spawn"),
        )
        if workers > 1 and len(jobs) > 1
        else nullcontext()
    )
    extract_document = partial(_extract_document, ocr=ocr, tessdata=tessdata)
    with pool_context as pool:
        results = (
            pool.map(extract_document, jobs)
            if pool is not None
            else map(extract_document, jobs)
        )
        for pages, report, tables in results:
            pages_output.extend(pages)
            report_output.append(report)
            tables_output.extend(tables)
            logger.info(
                "%s: status=%s, pages=%d, OCR attempted=%d, OCR errors=%d",
                report["code_version"], report["status"], report["page_count"],
                report["ocr_pages"], report["ocr_failed_pages"],
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_output_path.parent.mkdir(parents=True, exist_ok=True)

    page_schema = {
        "document_id": pl.Int64,
        "code": pl.Int64,
        "version": pl.Int64,
        "code_version": pl.String,
        "title": pl.String,
        "source_url": pl.String,
        "page_number": pl.Int64,
        "section_title": pl.String,
        "section_headings": pl.List(pl.String),
        "text": pl.String,
        "text_source": pl.String,
        "ocr_attempted": pl.Boolean,
        "ocr_error": pl.String,
    }

    report_schema = {
        "code_version": pl.String,
        "name": pl.String,
        "pdf_path": pl.String,
        "status": pl.String,
        "page_count": pl.Int64,
        "empty_pages": pl.Int64,
        "ocr_pages": pl.Int64,
        "ocr_failed_pages": pl.Int64,
        "ocr_messages": pl.List(pl.String),
        "pdf_repaired": pl.Boolean,
        "message": pl.String,
        "table_status": pl.String,
        "table_message": pl.String,
    }

    table_schema = {
        "document_id": pl.Int64,
        "code_version": pl.String,
        "page_number": pl.Int64,
        "table_index": pl.Int64,
        "caption": pl.String,
        "rows": pl.String,
        "status": pl.String,
        "message": pl.String,
    }

    pl.DataFrame(
        pages_output,
        schema=page_schema,
    ).write_parquet(output_path)

    pl.DataFrame(
        report_output,
        schema=report_schema,
    ).write_parquet(report_path)

    pl.DataFrame(
        tables_output,
        schema=table_schema,
    ).write_parquet(table_output_path)

    successful = sum(
        row["status"] == "ok"
        for row in report_output
    )

    failed_or_empty = len(report_output) - successful

    extracted_tables = sum(
        row["status"] == "extracted"
        for row in tables_output
    )

    unprocessed_tables = sum(
        row["status"] == "unprocessed"
        for row in tables_output
    )

    logger.info(
        "Обработка завершена: документов=%d, успешно=%d, проблемных=%d, "
        "страниц=%d, таблиц извлечено=%d, не обработано=%d",
        len(report_output),
        successful,
        failed_or_empty,
        len(pages_output),
        extracted_tables,
        unprocessed_tables,
    )

    return successful, failed_or_empty


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Extract clinical guidelines")
    parser.add_argument(
        "--no-ocr", action="store_true",
        help="disable OCR fallback on pages without native text",
    )
    parser.add_argument(
        "--tessdata", help="Tesseract language data directory (rus and eng)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of PDF extraction processes (default: 1)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    run_pipeline(workers=args.workers, ocr=not args.no_ocr, tessdata=args.tessdata)
