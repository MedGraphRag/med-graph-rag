"""Import unique drug mappings directly from the ready normalized Parquet."""
from pathlib import Path

import polars as pl
from neo4j import Driver
from pydantic import ValidationError

from src.graph.grls_models import GRLSDictionaryEntry, GRLSImportReport, _digest
from src.graph.grls_schema import create_grls_constraints
from src.nlp.grls_etl import normalized_name


DEFAULT_DICTIONARY = Path("data/processed/grls_normalized.parquet")
IMPORT_DICTIONARY_QUERY = """
UNWIND $rows AS row
MERGE (drug:GRLSDrug {drug_id: row.drug_id})
ON CREATE SET drug.trade_name_norm = row.trade_name_norm,
              drug.inn_norm = row.inn_norm, drug.atx_code = row.atx_code,
              drug.source = 'GRLS_normalized'
MERGE (dictionary:GRLSDictionary {name: $dictionary_name})
MERGE (dictionary)-[:CONTAINS]->(drug)
MERGE (trade:TradeName {name_norm: row.trade_name_norm})
MERGE (drug)-[:HAS_TRADE_NAME]->(trade)
MERGE (inn:INN {key: 'name:' + row.inn_norm})
ON CREATE SET inn.name_norm = row.inn_norm
MERGE (drug)-[:HAS_INN]->(inn)
FOREACH (code IN row.atx_code |
    MERGE (atc:ATC {code: code})
    MERGE (drug)-[:HAS_ATC]->(atc)
)
"""


def prepare_dictionary(path: str | Path) -> tuple[list[dict], GRLSImportReport]:
    frame = pl.read_parquet(path)
    columns = list(GRLSDictionaryEntry.model_fields)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError("Normalized dictionary is missing columns: " + ", ".join(sorted(missing)))
    seen: dict[str, dict] = {}
    for index, raw in enumerate(frame.select(columns).iter_rows(named=True), start=1):
        try:
            entry = GRLSDictionaryEntry.model_validate(raw)
        except ValidationError:
            raise ValueError(f"Invalid normalized drug at row {index}") from None
        seen.setdefault(entry.drug_id, entry.graph_row())
    return list(seen.values()), GRLSImportReport(
        input_records=frame.height, imported_records=len(seen),
        duplicate_input_records=frame.height - len(seen),
    )


def import_dictionary(
    driver: Driver, path: str | Path = DEFAULT_DICTIONARY, *,
    batch_size: int = 1000, database: str | None = None,
    dictionary_name: str = "main",
) -> GRLSImportReport:
    """Atomically activate a validated dictionary; preserve old source records."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows, report = prepare_dictionary(path)
    if not rows:
        raise ValueError("Normalized dictionary is empty; current dictionary was not changed")
    create_grls_constraints(driver, database=database)

    def write(tx):
        nodes = relationships = batches = 0
        for start in range(0, len(rows), batch_size):
            summary = tx.run(
                IMPORT_DICTIONARY_QUERY, rows=rows[start:start + batch_size],
                dictionary_name=dictionary_name,
            ).consume()
            nodes += summary.counters.nodes_created
            relationships += summary.counters.relationships_created
            batches += 1
        # Replace membership atomically; preserve observations and clinical edges.
        tx.run("""
            MATCH (:GRLSDictionary {name: $dictionary_name})-[membership:CONTAINS]->(drug:GRLSDrug)
            WHERE NOT drug.drug_id IN $ids
            DELETE membership
        """, ids=[row["drug_id"] for row in rows], dictionary_name=dictionary_name).consume()
        return nodes, relationships, batches

    with driver.session(database=database) as session:
        report.nodes_created, report.relationships_created, report.batches = session.execute_write(write)
    return report


def link_dictionary_sources(
    driver: Driver, dictionary_path: str | Path, records_path: str | Path, *,
    batch_size: int = 1000, database: str | None = None,
) -> int:
    """Attach existing observations only to exact, whole dictionary mappings."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    drugs, _ = prepare_dictionary(dictionary_path)
    lookup = {
        (row["trade_name_norm"], row["inn_norm"], tuple(row["atx_code"])): row["drug_id"]
        for row in drugs
    }
    records = pl.read_parquet(records_path).with_columns(
        normalized_name("commercial_name").alias("_trade"),
        normalized_name("mnn_name").alias("_inn"),
    )
    links = set()
    for row in records.iter_rows(named=True):
        codes = tuple(sorted({code.strip() for code in row["atx_codes"] if code.strip()}))
        drug_id = lookup.get((row["_trade"], row["_inn"], codes))
        if drug_id and row["lp_id"] and row["source_snapshot"]:
            links.add((drug_id, _digest([row["source_snapshot"], row["lp_id"]])))
    rows = [{"drug_id": drug_id, "record_id": record_id} for drug_id, record_id in sorted(links)]
    query = """
        UNWIND $rows AS row
        MATCH (drug:GRLSDrug {drug_id: row.drug_id})
        MATCH (record:GRLSRecord {record_id: row.record_id})
        MERGE (drug)-[:SUPPORTED_BY]->(record)
        RETURN count(*) AS linked
    """
    with driver.session(database=database) as session:
        total = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            total += session.execute_write(lambda tx: tx.run(query, rows=batch).single()["linked"])
    return total
