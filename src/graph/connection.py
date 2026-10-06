import logging

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from src.config import settings

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """
    Singleton-менеджер подключения к Neo4j.

    Создает один Neo4j Driver, предоставляет его
    остальным компонентам приложения и отвечает
    за проверку и закрытие соединения.
    """

    def __init__(self) -> None:
        self._driver: Driver | None = None
        self._connect()

    def _connect(self) -> None:
        """Инициализирует Neo4j Driver."""

        if self._driver is not None:
            return

        try:
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(
                    settings.neo4j_user,
                    settings.neo4j_password,
                ),
            )

            logger.info(
                "Neo4j driver initialized: %s",
                settings.neo4j_uri,
            )

        except (DriverError, Neo4jError):
            logger.exception("Failed to initialize Neo4j driver")
            self._driver = None
            raise

    @property
    def driver(self) -> Driver:
        """Возвращает активный Neo4j Driver."""

        if self._driver is None:
            raise RuntimeError(
                "Neo4j driver is not initialized or already closed."
            )

        return self._driver

    def verify_connection(self) -> bool:
        """Проверяет доступность Neo4j."""

        try:
            self.driver.verify_connectivity()

            logger.info("Neo4j connection verified.")
            return True

        except (DriverError, Neo4jError) as exc:
            logger.error(
                "Neo4j connection verification failed: %s",
                exc,
            )
            return False

    def close(self) -> None:
        """Закрывает Neo4j Driver."""

        if self._driver is None:
            return

        try:
            self._driver.close()
            logger.info("Neo4j connection closed.")

        finally:
            self._driver = None


# Единственный экземпляр подключения приложения.
graph_db = Neo4jConnection()