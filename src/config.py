from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Определение базовых директорий проекта
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


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