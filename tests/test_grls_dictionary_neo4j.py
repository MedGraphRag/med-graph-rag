"""Opt-in canonical dictionary tests in an isolated membership namespace."""
import os
import uuid

import polars as pl
import pytest

from src.graph.grls_dictionary import import_dictionary, link_dictionary_sources
from src.graph.grls_loader import import_grls
from src.graph.grls_search import GRLSGraphResolver
from src.nlp.grls_etl import build_normalized_row

pytestmark = pytest.mark.skipif(os.getenv("RUN_NEO4J_TESTS") != "1", reason="Opt-in Neo4j test")


@pytest.fixture
def dictionary_case(neo4j_driver, tmp_path):
    prefix = "grls-dict-test-" + uuid.uuid4().hex
    yield neo4j_driver, prefix, tmp_path
    with neo4j_driver.session() as session:
        session.run("""
            MATCH (n) WHERE
                (n:GRLSDictionary AND n.name = $prefix)
                OR (n:GRLSDrug AND n.trade_name_norm STARTS WITH $prefix)
                OR (n:GRLSRegistration AND n.lp_id STARTS WITH $prefix)
                OR (n:GRLSRecord AND n.source_snapshot = $prefix)
                OR (n:TradeName AND n.name_norm STARTS WITH $prefix)
                OR (n:INN AND n.name_norm STARTS WITH $prefix)
                OR (n:ATC AND n.code STARTS WITH $atc_prefix)
            DETACH DELETE n
        """, prefix=prefix, atc_prefix=prefix.upper()).consume()


def test_seventeen_source_records_produce_one_search_result(dictionary_case):
    driver, prefix, folder = dictionary_case
    dictionary = folder / "normalized.parquet"
    records = folder / "records.parquet"
    inn, code = prefix + " вещество", prefix.upper() + "-ATX"
    pl.DataFrame({"trade_name_norm": [prefix], "inn_norm": [inn], "atx_code": [[code]]}).write_parquet(dictionary)
    rows = [build_normalized_row(
        {"lpId": prefix + str(i), "commercialName": prefix, "mnnName": inn,
         "atx": [code], "conditionId": "Д" if i < 3 else "ИЗМ"},
        fix_mojibake=False, source_snapshot=prefix, source_page=f"page_{i}.json",
    ) for i in range(17)]
    pl.DataFrame(rows, schema_overrides={"mnn_codes": pl.List(pl.String), "atx_codes": pl.List(pl.String)}).write_parquet(records)
    import_grls(driver, records)
    import_dictionary(driver, dictionary, dictionary_name=prefix)
    repeated = import_dictionary(driver, dictionary, dictionary_name=prefix)
    assert repeated.nodes_created == repeated.relationships_created == 0
    assert link_dictionary_sources(driver, dictionary, records) == 17
    resolver = GRLSGraphResolver(driver, dictionary_name=prefix)
    assert len(resolver.search(prefix).matches) == 1
    assert len(resolver.search_records(prefix).matches) == 17
    with driver.session() as session:
        count = session.run("""
            MATCH (:GRLSDictionary {name: $name})-[:CONTAINS]->(drug:GRLSDrug)-[:SUPPORTED_BY]->(record)
            RETURN count(DISTINCT drug) AS drugs, count(DISTINCT record) AS records
        """, name=prefix).single()
        assert count["drugs"] == 1 and count["records"] == 17


def test_dictionary_replacement_preserves_other_dictionaries_and_old_nodes(dictionary_case):
    driver, prefix, folder = dictionary_case
    path = folder / "normalized.parquet"
    rows = [
        {"trade_name_norm": prefix, "inn_norm": prefix + " вещество", "atx_code": [prefix.upper() + "-A"]},
        {"trade_name_norm": prefix, "inn_norm": prefix + " вещество", "atx_code": [prefix.upper() + "-B"]},
    ]
    with driver.session() as session:
        before = session.run("MATCH (:GRLSDictionary {name: 'main'})-[:CONTAINS]->(d) RETURN count(d) AS n").single()["n"]
    pl.DataFrame(rows).write_parquet(path)
    import_dictionary(driver, path, dictionary_name=prefix)
    resolver = GRLSGraphResolver(driver, dictionary_name=prefix)
    assert len(resolver.search(prefix).matches) == 2
    pl.DataFrame(rows[:1]).write_parquet(path)
    import_dictionary(driver, path, dictionary_name=prefix)
    assert len(resolver.search(prefix).matches) == 1
    with driver.session() as session:
        assert session.run("MATCH (:GRLSDictionary {name: 'main'})-[:CONTAINS]->(d) RETURN count(d) AS n").single()["n"] == before
        assert session.run("MATCH (d:GRLSDrug {trade_name_norm: $name}) RETURN count(d) AS n", name=prefix).single()["n"] == 2
