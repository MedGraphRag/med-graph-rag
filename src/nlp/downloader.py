import json
import logging
from pathlib import Path
import time
from curl_cffi import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "cr_manifest.json"
PDF_DIR = BASE_DIR / "data" / "raw" / "guidelines" / "pdf"


def main():
    if not MANIFEST_PATH.exists():
        logger.error(f"Манифест не найден по пути: {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    records = []
    if isinstance(manifest_data, list):
        records = manifest_data
    elif isinstance(manifest_data, dict):
        for key in ["records", "items", "data", "list", "clinical_recommendations"]:
            if key in manifest_data and isinstance(manifest_data[key], list):
                records = manifest_data[key]
                break
        if not records:
            for v in manifest_data.values():
                if isinstance(v, list):
                    records = v
                    break

    if not records:
        logger.warning("Не удалось найти список документов в манифесте.")
        return

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skipped_count = 0
    error_count = 0

    logger.info(f"Старт выгрузки PDF для {len(records)} документов через curl_cffi...")

    for i, record in enumerate(records, 1):
        cr_id = record.get("Id") or record.get("id") or record.get("cr_id")
        if not cr_id:
            continue

        pdf_path = PDF_DIR / f"{cr_id}.pdf"

        if pdf_path.exists() and pdf_path.stat().st_size > 10 * 1024:
            skipped_count += 1
            continue

        page_url = f"https://cr.minzdrav.gov.ru/recomend/{cr_id}"
        pdf_url = f"https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id={cr_id}"

        headers = {
            'accept': 'application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'accept-language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'authorization': 'Bearer null',
            'origin': 'https://cr.minzdrav.gov.ru',
            'referer': page_url,
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36',
        }

        try:
            time.sleep(1.2)

            # Выполняем GET-запрос на скачивание PDF с имитацией TLS-отпечатка Chrome
            response = requests.get(
                pdf_url,
                headers=headers,
                impersonate="chrome120",
                timeout=45.0
            )

            if response.status_code == 200:
                content = response.content
                
                # Проверка на заглушку WAF (1407 байт или HTML)
                if len(content) < 5 * 1024 or b"html" in content[:100].lower():
                    logger.warning(f"[{i}/{len(records)}] Документ {cr_id}: получена защита/заглушка ({len(content)} байт).")
                    error_count += 1
                    continue

                pdf_path.write_bytes(content)
                logger.info(f"[{i}/{len(records)}] Успешно скачано: {cr_id}.pdf ({len(content)} байт)")
                success_count += 1
            else:
                logger.warning(f"[{i}/{len(records)}] Ошибка {cr_id}: HTTP {response.status_code}")
                error_count += 1

        except Exception as e:
            logger.error(f"[{i}/{len(records)}] Исключение для {cr_id}: {type(e).__name__} - {e}")
            error_count += 1

    logger.info(f"Итог: Успешно: {success_count}, Пропущено: {skipped_count}, Ошибок: {error_count}")


if __name__ == "__main__":
    main()