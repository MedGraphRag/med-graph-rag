"""Командный запуск проверки назначений по подтверждённым связям Neo4j."""

import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from pydantic import ValidationError
from polars.exceptions import PolarsError

from src.graph.prescription_graph import Neo4jPrescriptionChecker
from src.nlp.grls_resolver import DrugResolver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dictionary", type=Path,
        default=Path("data/processed/grls_normalized.parquet"),
        help="Нормализованный справочник лекарств Parquet",
    )
    parser.add_argument("--database", help="Имя базы Neo4j")
    commands = parser.add_subparsers(dest="command", required=True)
    disease = commands.add_parser("drug-disease", help="Лекарство и заболевание")
    disease.add_argument("--drug", required=True)
    disease.add_argument("--icd10", required=True)
    interaction = commands.add_parser("drug-drug", help="Два лекарства")
    interaction.add_argument("--first", required=True)
    interaction.add_argument("--second", required=True)
    args = parser.parse_args(argv)

    try:
        # Настройки загружаются после разбора аргументов, чтобы --help работал без .env.
        from src.config import settings

        resolver = DrugResolver(args.dictionary)
        with GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        ) as driver:
            checker = Neo4jPrescriptionChecker(
                driver, resolver, database=args.database
            )
            if args.command == "drug-disease":
                result = checker.check_drug_disease(args.drug, args.icd10)
            else:
                result = checker.check_drug_drug(args.first, args.second)
        print(result.model_dump_json(indent=2))
    except (DriverError, Neo4jError, ValidationError, PolarsError, OSError, ValueError) as error:
        # Текст исключения может содержать настройки подключения или исходные данные.
        print(
            "Проверка не выполнена. Проверьте настройки Neo4j, справочник "
            f"и формат клинических связей. Тип ошибки — {type(error).__name__}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
