
import json
import logging
from pathlib import Path

import polars as pl

from src.nlp.cr_extractor import extract_pages, extract_section_headings

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "cr_manifest.json"
PDF_DIR = BASE_DIR / "data" / "raw" / "guidelines"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "cr_pages.parquet"
REPORT_PATH = BASE_DIR / "data" / "processed" / "cr_extraction_report.parquet"
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


def run_pipeline(
    manifest_path: Path = MANIFEST_PATH,
    pdf_dir: Path = PDF_DIR,
    output_path: Path = OUTPUT_PATH,
    report_path: Path = REPORT_PATH,
) -> tuple[int, int]:
    records = load_manifest(manifest_path)
    pages_output = []
    report_output = []
    seen_versions = set()

    for record in records:
        code_version = record.get("CodeVersion")
        if not code_version:
            if record.get("Code") is not None and record.get("Version") is not None:
                code_version = f"{record['Code']}_{record['Version']}"
            else:
                continue

        if code_version in seen_versions:
            continue
        seen_versions.add(code_version)

        pdf_path = pdf_dir / f"{code_version}.pdf"

        report_row = {
            "code_version": code_version,
            "name": record.get("Name"),
            "pdf_path": str(pdf_path),
            "status": "error",
            "page_count": 0,
            "empty_pages": 0,
            "message": None,
        }

        try:
            pages = extract_pages(pdf_path)
            empty_pages = sum(not page["text"] for page in pages)

            report_row.update(
                status="ok" if pages and empty_pages < len(pages) else "empty",
                page_count=len(pages),
                empty_pages=empty_pages,
            )

            current_section = None

            for page in pages:
                headings = extract_section_headings(page["formatted_text"])

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
                        "text": page["text"],
                    }
                )

        except (OSError, ValueError, RuntimeError) as error:
            report_row["status"] = (
                "missing_pdf" if not pdf_path.exists() else "error"
            )
            report_row["message"] = str(error)
            logger.warning("%s: %s", code_version, error)

        report_output.append(report_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    page_schema = {
        "document_id": pl.Int64,
        "code": pl.Int64,
        "version": pl.Int64,
        "code_version": pl.String,
        "title": pl.String,
        "source_url": pl.String,
        "page_number": pl.Int64,
        "section_title": pl.String,
        "text": pl.String,
    }

    report_schema = {
        "code_version": pl.String,
        "name": pl.String,
        "pdf_path": pl.String,
        "status": pl.String,
        "page_count": pl.Int64,
        "empty_pages": pl.Int64,
        "message": pl.String,
    }

    pl.DataFrame(pages_output, schema=page_schema).write_parquet(output_path)
    pl.DataFrame(report_output, schema=report_schema).write_parquet(report_path)

    successful = sum(row["status"] == "ok" for row in report_output)
    failed_or_empty = len(report_output) - successful

    logger.info(
        "Обработка завершена: документов=%d, успешно=%d, проблемных=%d, страниц=%d",
        len(report_output),
        successful,
        failed_or_empty,
        len(pages_output),
    )

    return successful, failed_or_empty


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_pipeline()
