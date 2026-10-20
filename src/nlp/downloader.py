"""Скачивание PDF клинических рекомендаций по CodeVersion из манифеста."""

import argparse
import json
import logging
from pathlib import Path
import re
import time

from curl_cffi import CurlHttpVersion, requests
from curl_cffi.requests.errors import RequestsError
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "cr_manifest.json"
PDF_DIR = BASE_DIR / "data" / "raw" / "guidelines" / "pdf"
API_URL = "https://apicr.minzdrav.gov.ru/api.ashx"


class Recommendation(BaseModel):
    code_version: str = Field(alias="CodeVersion", pattern=r"^[0-9]+_[0-9]+$")

    @model_validator(mode="before")
    @classmethod
    def resolve_code_version(cls, record):
        # Id — внутренний ключ записи, PDF API принимает именно Code_Version.
        if isinstance(record, dict) and not record.get("CodeVersion"):
            record = dict(record)
            if record.get("Code") is not None and record.get("Version") is not None:
                record["CodeVersion"] = f"{record['Code']}_{record['Version']}"
        return record


def load_records(path: Path) -> list:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "items", "Data", "data", "list", "clinical_recommendations"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Манифест не содержит списка документов")


def is_pdf(content: bytes) -> bool:
    # Content-Type недостаточно: сервер возвращает application/pdf даже для HTML.
    return content.startswith(b"%PDF-") and b"%%EOF" in content[-1024:]


def response_error(response) -> str:
    text = response.content[:2000].decode("utf-8", errors="replace")
    text = " ".join(re.sub(r"<[^>]*>", " ", text).split())
    return (
        f"HTTP {response.status_code}, Content-Type={response.headers.get('content-type')}, "
        f"{len(response.content)} байт; вместо PDF: {text[:400]}"
    )


def fetch_pdf(session, code_version: str, *, attempts=3, delay=1.2) -> bytes:
    for attempt in range(attempts):
        try:
            response = session.get(
                API_URL,
                params={"op": "GetClinrecPdf", "id": code_version},
                headers={"Referer": f"https://cr.minzdrav.gov.ru/view-cr/{code_version}"},
                timeout=180.0,
            )
        except RequestsError:
            if attempt == attempts - 1:
                raise
        else:
            if response.status_code == 200 and is_pdf(response.content):
                return response.content
            retryable = response.status_code in (429, 500, 502, 503, 504)
            # IIS иногда маскирует внутреннюю ошибку статусом HTTP 200.
            retryable |= response.status_code == 200 and b"iisnode encountered an error" in response.content[:2000]
            error = response_error(response)
            if not retryable or attempt == attempts - 1:
                raise ValueError(error)
            logger.warning("%s: %s", code_version, error)
        logger.warning("%s: повтор %d/%d", code_version, attempt + 2, attempts)
        time.sleep(max(delay, 1.0) * 2**attempt)
    raise ValueError("Число попыток должно быть положительным")


def download_records(records: list, output_dir: Path, *, delay=1.2) -> tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    success = skipped = errors = 0

    with requests.Session(
        impersonate="chrome",
        http_version=CurlHttpVersion.V1_1,
        headers={"Accept": "application/pdf", "Accept-Language": "ru-RU,ru;q=0.9"},
    ) as session:
        for index, record in enumerate(records, 1):
            try:
                code_version = Recommendation.model_validate(record).code_version
                path = output_dir / f"{code_version}.pdf"
                if path.exists():
                    if is_pdf(path.read_bytes()):
                        skipped += 1
                        continue
                    # Не затираем ранее сохранённые данные, даже если это HTML.
                    raise ValueError(f"Некорректный PDF уже существует: {path}; переместите его перед повтором")
                time.sleep(delay)
                content = fetch_pdf(session, code_version, delay=delay)
                temporary = path.with_suffix(".pdf.part")
                temporary.write_bytes(content)
                temporary.replace(path)
                success += 1
                logger.info("[%d/%d] %s: сохранено %d байт", index, len(records), path.name, len(content))
            except (RequestsError, ValueError, OSError) as exc:
                errors += 1
                logger.error("[%d/%d] %s", index, len(records), exc)
    logger.info("Итог: загружено %d, пропущено %d, ошибок %d", success, skipped, errors)
    return success, skipped, errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--limit", type=int, help="Проверить только первые N записей")
    parser.add_argument("--delay", type=float, default=1.2, help="Пауза между документами, секунды")
    args = parser.parse_args(argv)
    if args.delay < 0 or (args.limit is not None and args.limit < 1):
        parser.error("delay должен быть >= 0, limit должен быть >= 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        records = load_records(args.manifest)
        if not records:
            raise ValueError("Манифест пуст")
        _, _, errors = download_records(records[:args.limit], args.output_dir, delay=args.delay)
        return int(errors > 0)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
