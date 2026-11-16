"""Проверка реального Neo4j на временных синтетических записях."""

import os
from uuid import uuid4

import polars as pl
import pytest
from neo4j import GraphDatabase

from src.graph.prescription_graph import Neo4jPrescriptionChecker
from src.nlp.grls_resolver import DrugResolver


@pytest.mark.skipif(
    os.getenv("RUN_PRESCRIPTION_NEO4J_TESTS") != "1",
    reason="Нужен Neo4j и RUN_PRESCRIPTION_NEO4J_TESTS=1",
)
def test_prescription_checker_against_real_neo4j(tmp_path):
    from src.config import settings

    run_id = uuid4().hex
    first_inn = "synthetic-a-" + run_id
    second_inn = "synthetic-b-" + run_id
    third_inn = "synthetic-c-" + run_id
    code = "SYNTHETIC-" + run_id
    dictionary_path = tmp_path / "synthetic_drugs.parquet"
    pl.DataFrame({
        "trade_name_norm": ["синтетический а", "синтетический б", "синтетический в"],
        "inn_norm": [first_inn, second_inn, third_inn],
        "atx_code": [["TEST01"]] * 3,
    }).write_parquet(dictionary_path)
    query = """
    CREATE (first:INN:PrescriptionCheckerTest {
        key: $first_key, name_norm: $first_inn, test_run_id: $run_id
    })
    CREATE (second:INN:PrescriptionCheckerTest {
        key: $second_key, name_norm: $second_inn, test_run_id: $run_id
    })
    CREATE (:INN:PrescriptionCheckerTest {
        key: $third_key, name_norm: $third_inn, test_run_id: $run_id
    })
    CREATE (disease:ICD10:PrescriptionCheckerTest {
        code: $code, test_run_id: $run_id
    })
    CREATE (first)-[:CONTRAINDICATED_FOR {
        status: 'confirmed', document_id: $document_id, location: 'test page 1',
        quote: $quote, conditions: ['synthetic condition']
    }]->(disease)
    CREATE (first)-[:CONTRAINDICATED_FOR {status: 'disputed'}]->(disease)
    CREATE (first)-[:INTERACTS_WITH {
        status: 'confirmed', document_id: $document_id, location: 'test page 2',
        quote: $quote, description: 'synthetic A affects B',
        conditions: ['synthetic condition']
    }]->(second)
    CREATE (first)-[:INTERACTS_WITH {status: 'unresolved'}]->(second)
    """
    with GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    ) as driver:
        try:
            with driver.session() as session:
                session.run(
                    query, run_id=run_id, first_inn=first_inn,
                    second_inn=second_inn, third_inn=third_inn,
                    first_key="name:" + first_inn,
                    second_key="name:" + second_inn,
                    third_key="name:" + third_inn,
                    code=code, document_id="synthetic-" + run_id,
                    quote="  Synthetic test quote.  ",
                ).consume()
            checker = Neo4jPrescriptionChecker(
                driver, DrugResolver(dictionary_path)
            )
            disease = checker.check_drug_disease("синтетический а", code)
            forward = checker.check_drug_drug("синтетический а", "синтетический б")
            reverse = checker.check_drug_drug("синтетический б", "синтетический а")

            assert disease.status == "confirmed_match"
            assert len(disease.matches) == 1
            assert disease.matches[0].source.quote == "  Synthetic test quote.  "
            assert disease.matches[0].conditions == ("synthetic condition",)
            assert forward.status == reverse.status == "confirmed_match"
            assert len(forward.matches) == 1
            assert forward.matches == reverse.matches
            assert forward.matches[0].first_drug_inn == first_inn
            assert forward.matches[0].second_drug_inn == second_inn
            assert checker.check_drug_drug(
                "синтетический а", "синтетический в"
            ).status == "insufficient_data"
        finally:
            # Удаляются только узлы этого тестового запуска и их связи.
            with driver.session() as session:
                session.run(
                    "MATCH (n:PrescriptionCheckerTest {test_run_id: $run_id}) "
                    "DETACH DELETE n",
                    run_id=run_id,
                ).consume()
