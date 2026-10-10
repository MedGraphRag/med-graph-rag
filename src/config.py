from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Настройка путей
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

DATA_DIR = PROCESSED_DATA_DIR
OUTPUT_FILE = DATA_DIR / "cr_manifest.json"

# Настройки API
API_URL = 'https://apicr.minzdrav.gov.ru/api.ashx'
PARAMS = {'op': 'GetJsonClinrecsFilterV2'}

HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'content-type': 'application/json',
    'origin': 'https://cr.minzdrav.gov.ru',
    'referer': 'https://cr.minzdrav.gov.ru/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36',
}

# Payload для фильтрации по нозологиям
PAYLOAD = {
    'filters': [
        {
            'fieldName': 'status',
            'filterType': 1,
            'filterValueType': 2,
            'value1': 0,
            'value2': '',
            'values': [],
        },
        {
            'fieldName': 'mkbid',
            'filterType': 9,
            'filterValueType': 1,
            'value1': '',
            'value2': '',
            'values': [3712, 2000],
        },
    ],
    'sortOption': {'fieldName': 'publishdate', 'sortType': 2},
    'pageSize': 120,
    'currentPage': 1,
    'useANDoperator': True,
    'columns': [],
}

class Settings(BaseSettings):
    """
    Единый класс настроек и переменных окружения проекта MedGraphRAG.
    Автоматически считывает значения из файла .env в корне проекта.
    """

    # Настройки базы данных Neo4j
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Bolt URI для подключения к Neo4j",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Имя пользователя Neo4j",
    )
    neo4j_password: str = Field(
        description="Пароль пользователя Neo4j",
    )

    # Настройки LLM (Groq / резерв)
    llm_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL API языковой модели",
    )
    llm_api_key: str = Field(
        default="mock_key",
        description="API ключ для доступа к LLM",
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Имя используемой модели",
    )

    # Конфигурация Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Игнорировать посторонние переменные в .env
    )


# Создаем единственный экземпляр настроек (синглтон) для использования во всем проекте
settings = Settings()

if __name__ == "__main__":
    # Быстрый тест запуска конфига
    print("✓ Файл конфигурации успешно загружен!")
    print(f"Neo4j URI: {settings.neo4j_uri}")
    print(f"Neo4j User: {settings.neo4j_user}")
    print(f"Пароль скрыт: {'*' * len(settings.neo4j_password)}")
    print(f"Корневая папка проекта: {BASE_DIR}")
    print(f"Папка данных: {PROCESSED_DATA_DIR}")