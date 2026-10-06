import json
import logging
from pathlib import Path

from neo4j import Driver

from src.graph.connection import graph_db

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/icd10.json")


def create_constraints(driver: Driver) -> None:
    """Создает необходимые ограничения в Neo4j."""

    query = """
    CREATE CONSTRAINT icd10_code_unique IF NOT EXISTS
    FOR (n:ICD10)
    REQUIRE n.code IS UNIQUE
    """

    with driver.session() as session:
        session.run(query)

    logger.info("Constraint для ICD10.code создан/уже существует.")


def load_icd10_data(path: Path) -> dict:
    """Загружает нормализованный ICD-10 JSON."""

    if not path.exists():
        raise FileNotFoundError(f"Файл ICD-10 не найден: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    logger.info(
        "Загружен ICD-10: version=%s, language=%s, nodes=%d",
        data.get("version"),
        data.get("language"),
        len(data.get("nodes", [])),
    )

    return data


def import_nodes(driver: Driver, nodes: list[dict]) -> None:
    """Импортирует узлы ICD-10."""

    query = """
    UNWIND $nodes AS node

    MERGE (n:ICD10 {code: node.code})
    SET n.name = node.name,
        n.level = node.level,
        n.parent_code = node.parent_code
    """

    with driver.session() as session:
        session.run(query, nodes=nodes)

    logger.info("Импортировано узлов: %d", len(nodes))


def import_relationships(driver: Driver, nodes: list[dict]) -> None:
    """Создает связи PARENT_OF между узлами ICD-10."""

    query = """
    UNWIND $nodes AS node

    WITH node
    WHERE node.parent_code IS NOT NULL

    MATCH (parent:ICD10 {code: node.parent_code})
    MATCH (child:ICD10 {code: node.code})

    MERGE (parent)-[:PARENT_OF]->(child)
    """

    with driver.session() as session:
        session.run(query, nodes=nodes)

    logger.info("Связи PARENT_OF созданы.")


def verify_import(driver: Driver) -> None:
    """Проверяет результат импорта."""

    query = """
    MATCH (n:ICD10)
    RETURN count(n) AS count
    """

    with driver.session() as session:
        result = session.run(query).single()

    count = result["count"]

    logger.info("В Neo4j найдено узлов ICD10: %d", count)


def import_icd10() -> None:
    """Основная функция импорта ICD-10 в Neo4j."""

    if not graph_db.verify_connection():
        raise RuntimeError("Не удалось подключиться к Neo4j.")

    driver = graph_db.driver

    data = load_icd10_data(DATA_PATH)
    nodes = data["nodes"]

    create_constraints(driver)

    import_nodes(driver, nodes)
    import_relationships(driver, nodes)

    verify_import(driver)

    logger.info("Импорт ICD-10 успешно завершен.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import_icd10()