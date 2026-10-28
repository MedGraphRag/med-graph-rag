"""Opt-in integration checks against Neo4j 5; only synthetic GRLS nodes are removed.

Run with RUN_NEO4J_TESTS=1 and the existing NEO4J_* local configuration.
"""
import os
import uuid

import polars as pl
import pytest
from neo4j.exceptions import ConstraintError

from src.graph.grls_loader import import_grls
from src.graph.grls_search import GRLSGraphResolver
from src.nlp.grls_etl import build_normalized_row

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_NEO4J_TESTS") != "1", reason="Set RUN_NEO4J_TESTS=1 for Neo4j integration tests",
)


@pytest.fixture
def graph_case(neo4j_driver, tmp_path):
    prefix = "grls-test-" + uuid.uuid4().hex
    rows = [
        build_normalized_row(
            {"lpId": prefix + "-1", "commercialName": prefix,
             "mnnName": "Вещество Б + Вещество В", "mnnCode": [prefix + "-m1", prefix + "-m2"],
             "atx": [prefix.upper() + "-A1", prefix.upper() + "-A2"], "conditionId": "Д"},
            fix_mojibake=False, source_snapshot=prefix, source_page="page_1.json",
        ),
        build_normalized_row(
            {"lpId": prefix + "-2", "commercialName": prefix,
             "mnnName": "Вещество Г", "mnnCode": [prefix + "-m3"],
             "atx": [prefix.upper() + "-A3"], "conditionId": "П"},
            fix_mojibake=False, source_snapshot=prefix, source_page="page_2.json",
        ),
    ]
    path = tmp_path / "records.parquet"

    def write(values):
        pl.DataFrame(values, schema_overrides={"mnn_codes": pl.List(pl.String), "atx_codes": pl.List(pl.String)}).write_parquet(path)
        return path

    write(rows)
    yield neo4j_driver, prefix, rows, write
    with neo4j_driver.session() as session:
        # Keys use a random namespace. Never clear a database or touch ICD10.
        session.run("""
            MATCH (n)
            WHERE (n:GRLSRegistration AND n.lp_id STARTS WITH $prefix)
               OR (n:GRLSRecord AND n.source_snapshot STARTS WITH $prefix)
               OR (n:TradeName AND n.name_norm STARTS WITH $prefix)
               OR (n:INN AND (n.code STARTS WITH $prefix OR n.name_norm STARTS WITH $prefix))
               OR (n:ATC AND n.code STARTS WITH $atc_prefix)
            DETACH DELETE n
        """, prefix=prefix, atc_prefix=prefix.upper()).consume()


def test_repeat_import_and_search_preserve_registration_evidence(graph_case):
    driver, prefix, rows, write = graph_case
    duplicate = dict(rows[0], source_page="page_3.json")
    path = write([*rows, duplicate])
    first = import_grls(driver, path, batch_size=1)
    second = import_grls(driver, path, batch_size=2)
    assert first.imported_records == second.imported_records == 2
    assert first.duplicate_input_records == second.duplicate_input_records == 1
    assert first.nodes_created > 0 and first.relationships_created > 0
    assert second.nodes_created == second.relationships_created == 0
    result = GRLSGraphResolver(driver).search_records(prefix.upper())
    assert result.ambiguous
    assert len(result.matches) == 2
    matches = {match.lp_id: match for match in result.matches}
    assert matches[prefix + "-1"].mnn_codes == sorted(rows[0]["mnn_codes"])
    assert matches[prefix + "-1"].atx_codes == sorted(rows[0]["atx_codes"])
    assert matches[prefix + "-2"].atx_codes == rows[1]["atx_codes"]
    assert matches[prefix + "-1"].source_pages == ["page_1.json", "page_3.json"]
    assert matches[prefix + "-1"].source_snapshot == prefix
    with driver.session() as session:
        counts = session.run("""
            MATCH (r:GRLSRecord {source_snapshot: $snapshot})
            RETURN count(r) AS records, count(DISTINCT r.lp_id) AS registrations
        """, snapshot=prefix).single()
        assert counts["records"] == counts["registrations"] == 2
        pairs = session.run("""
            MATCH (inn:INN)-[edge]-(atc:ATC)
            WHERE inn.code STARTS WITH $prefix
            RETURN count(edge) AS count
        """, prefix=prefix).single()
        assert pairs["count"] == 0


def test_new_snapshot_does_not_mix_old_and_new_code_sets(graph_case):
    driver, prefix, rows, write = graph_case
    import_grls(driver, write([rows[0]]))
    changed = dict(rows[0], source_snapshot=prefix + "-next", atx_codes=[prefix.upper() + "-A4"])
    import_grls(driver, write([changed]))
    resolver = GRLSGraphResolver(driver)
    all_matches = resolver.search_records(prefix).matches
    assert len(all_matches) == 2
    assert len({match.lp_id for match in all_matches}) == 1
    latest = resolver.search_records(prefix, source_snapshot=prefix + "-next").matches
    assert len(latest) == 1
    assert latest[0].atx_codes == changed["atx_codes"]
    original = resolver.search_records(prefix, source_snapshot=prefix).matches
    assert original[0].atx_codes == sorted(rows[0]["atx_codes"])


def test_trade_dosage_and_complete_inn_search_use_existing_normalization(graph_case):
    driver, prefix, rows, write = graph_case
    rows[0]["commercial_name"] = prefix + "® 10 мг"
    rows[1]["commercial_name"] = prefix + " 20 мг"
    rows[0]["mnn_name"] = prefix + " Вещество Б + Вещество В"
    rows[1]["mnn_name"] = prefix + " Вещество Г"
    import_grls(driver, write(rows))
    resolver = GRLSGraphResolver(driver)
    variants = resolver.search_records(prefix)
    assert {match.trade_name for match in variants.matches} == {prefix + " 10", prefix + " 20"}
    assert variants.ambiguous
    exact = resolver.search_records(prefix.upper() + " 10 мг")
    assert len(exact.matches) == 1
    assert exact.matches[0].lp_id == rows[0]["lp_id"]
    assert exact.matches[0].atx_codes == sorted(rows[0]["atx_codes"])
    by_inn = resolver.search_records(rows[0]["mnn_name"].upper())
    assert len(by_inn.matches) == 1
    assert by_inn.matches[0].lp_id == rows[0]["lp_id"]
    assert resolver.search_records(prefix + " неизвестное название").matches == []


def test_conflict_rolls_back_whole_batch_and_preserves_existing_evidence(graph_case):
    driver, prefix, rows, write = graph_case
    import_grls(driver, write([rows[0]]))
    changed = dict(rows[0], atx_codes=[prefix.upper() + "-A4"])
    with pytest.raises(ConstraintError):
        import_grls(driver, write([rows[1], changed]), batch_size=2)
    result = GRLSGraphResolver(driver).search_records(prefix)
    assert len(result.matches) == 1
    assert result.matches[0].atx_codes == sorted(rows[0]["atx_codes"])
    with driver.session() as session:
        count = session.run("MATCH (r:GRLSRegistration {lp_id: $id}) RETURN count(r) AS count", id=rows[1]["lp_id"]).single()
        assert count["count"] == 0


def test_sparse_record_and_uncoded_combination_are_preserved(graph_case):
    driver, prefix, rows, write = graph_case
    combination = prefix + " Вещество Б + Вещество В"
    rows[0].update(mnn_codes=[], mnn_name=combination, atx_codes=[])
    rows[1].update(mnn_codes=[], mnn_name="~", atx_codes=[])
    import_grls(driver, write(rows))
    result = GRLSGraphResolver(driver).search_records(prefix)
    assert len(result.matches) == 2
    matches = {match.lp_id: match for match in result.matches}
    assert matches[prefix + "-1"].inn == combination.lower()
    assert matches[prefix + "-1"].mnn_codes == []
    assert matches[prefix + "-2"].inn is None
    assert all(match.atx_codes == [] for match in matches.values())
    with driver.session() as session:
        names = session.run("""
            MATCH (r:GRLSRecord {source_snapshot: $snapshot})-[:HAS_INN]->(inn:INN)
            RETURN inn.name_norm AS name, inn.code AS code
        """, snapshot=prefix).data()
        assert names == [{"name": combination.lower(), "code": None}]


def test_icd10_nodes_properties_and_hierarchy_are_unchanged(graph_case):
    driver, _, rows, write = graph_case

    def hierarchy():
        with driver.session() as session:
            nodes = session.run("MATCH (n:ICD10) RETURN n.code AS code, properties(n) AS properties ORDER BY code").data()
            edges = session.run("""
                MATCH (p:ICD10)-[r:PARENT_OF]->(c:ICD10)
                RETURN p.code AS parent, c.code AS child, properties(r) AS properties
                ORDER BY parent, child
            """).data()
        return nodes, edges

    before = hierarchy()
    import_grls(driver, write(rows), batch_size=1)
    assert hierarchy() == before
