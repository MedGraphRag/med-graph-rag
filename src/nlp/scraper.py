import json
import time
from typing import List, Any, Dict
from pydantic import BaseModel, ConfigDict, Field
from curl_cffi import requests as crequests
from curl_cffi.requests.errors import RequestsError

from src.config import API_URL, PARAMS, HEADERS, PAYLOAD, OUTPUT_FILE, DATA_DIR


# Pydantic модели
class ClinicalRecommendation(BaseModel):
    model_config = ConfigDict(extra="allow")

    item_id: int = Field(alias="Id")
    name: str = Field(alias="Name")


class Manifest(BaseModel):
    total_count: int
    items: List[ClinicalRecommendation]


# Механизм Retry
def retry_request(max_retries=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RequestsError as e:
                    print(f"Ошибка сети: {e}. Попытка {attempt + 1} из {max_retries}...")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator


@retry_request(max_retries=3, delay=2)
def fetch_manifest() -> Manifest:
    """Сбор данных с имперсонацией Chrome и сохранением полного JSON объекта."""
    response = crequests.post(
        API_URL,
        params=PARAMS,
        headers=HEADERS,
        json=PAYLOAD,
        impersonate="chrome",
        timeout=30.0
    )
    
    response.raise_for_status()
    
    raw_data = response.json().get('Data', [])
    
    parsed_items = [ClinicalRecommendation(**item) for item in raw_data]
    
    return Manifest(
        total_count=len(parsed_items),
        items=parsed_items
    )


def export_to_json(manifest: Manifest):
    """Экспорт манифеста в файл"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json_data = manifest.model_dump(mode='json', by_alias=True)
        json.dump(json_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    print("Начинаем сбор полного манифеста КР...")
    try:
        manifest_data = fetch_manifest()
        total_items = manifest_data.total_count
        
        if total_items >= 30:
            export_to_json(manifest_data)
            print(f"Успех! Собрано {total_items} КР с полным набором полей.")
            print(f"Файл сохранен в: {OUTPUT_FILE}")
        else:
            print(f"Предупреждение: Собрано всего {total_items} КР (ожидалось >= 30).")
            
    except Exception as e:
        print(f"Критическая ошибка при сборе: {e}")
