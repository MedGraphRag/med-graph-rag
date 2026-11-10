"""Import the normalized drug dictionary or its individual source records."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl
from neo4j import Driver
from pydantic import ValidationError

from src.graph.grls_models import GRLSImportReport, GRLSRecord
from src.graph.grls_schema import create_grls_constraints
from src.nlp.grls_etl import NormalizedGRLSRecord

LOG = logging.getLogger(__name__)
DEFAULT_INPUT = Path("data/processed/grls_normalized_records.parquet")

# Independent FOREACH blocks preserve sparse records and avoid a Cartesian
# product between INN and ATC arrays. All edges are backed by the source record.
IMPORT_QUERY = """
UNWIND $rows AS row
MERGE (registration:GRLSRegistration {lp_id: row.lp_id})
MERGE (record:GRLSRecord {
    record_id: row.record_id, content_hash: row.content_hash
})
ON CREATE SET record += row.properties,
              record.source = 'GRLS',
              record.source_pages = []
SET record.source_pages = CASE
    WHEN row.source_page IN record.source_pages THEN record.source_pages
    ELSE record.source_pages + row.source_page END
MERGE (registration)-[:HAS_RECORD]->(record)
FOREACH (trade IN row.trades |
    MERGE (name:TradeName {name_norm: trade.name_norm})
    MERGE (record)-[:HAS_TRADE_NAME]->(name)
)
FOREACH (inn IN row.inns |
    MERGE (substance:INN {key: inn.key})
    ON CREATE SET substance.code = inn.code, substance.name_norm = inn.name_norm
    MERGE (record)-[:HAS_INN]->(substance)
)
FOREACH (atc IN row.atcs |
    MERGE (classification:ATC {code: atc.code})
    MERGE (record)-[:HAS_ATC]->(classification)
)
"""


def prepare_records(path: Path) -> tuple[list[dict], GRLSImportReport]:
    """Validate the complete input before the first database write.

    Rows without IDs and rows outside Д/П/ИЗМ are counted and skipped. Invalid
    structure or conflicting observations abort the import before schema writes.
    """
    frame = pl.read_parquet(path)
    missing = set(NormalizedGRLSRecord.model_fields) - set(frame.columns)
    if missing:
        raise ValueError("GRLS records Parquet is missing columns: " + ", ".join(sorted(missing)))
    report = GRLSImportReport(input_records=frame.height)
    rows: list[dict] = []
    seen: dict[str, dict] = {}
    for index, raw in enumerate(frame.iter_rows(named=True), start=1):
        if raw["lp_id"] is None or not str(raw["lp_id"]).strip():
            report.skipped_missing_id += 1
            continue
        if raw["condition_id"] not in ("Д", "П", "ИЗМ"):
            report.skipped_status += 1
            continue
        try:
            record = GRLSRecord.model_validate(raw)
            row = record.graph_row()
        except ValidationError as exc:
            # Do not print source values or bulky payloads in validation errors.
            fields = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())
            raise ValueError(f"Invalid GRLS record at row {index}; fields: {fields}") from None
        previous = seen.get(record.record_id)
        if previous is not None:
            if previous["content_hash"] != row["content_hash"]:
                raise ValueError(f"Conflicting GRLS observations in the same snapshot at row {index}")
            report.duplicate_input_records += 1
        else:
            seen[record.record_id] = row
        # Keep page locators even when the same observation appears on two pages.
        rows.append(row)
    report.imported_records = len(seen)
    return rows, report


def import_grls(
    driver: Driver,
    path: str | Path = DEFAULT_INPUT,
    *,
    batch_size: int = 1000,
    database: str | None = None,
) -> GRLSImportReport:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows, report = prepare_records(Path(path))
    if not rows:
        return report
    create_grls_constraints(driver, database=database)
    with driver.session(database=database) as session:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            summary = session.execute_write(_import_batch, batch)
            report.batches += 1
            report.nodes_created += summary.counters.nodes_created
            report.relationships_created += summary.counters.relationships_created
            LOG.info("Committed GRLS batch %d (%d rows)", report.batches, len(batch))
    return report


def _import_batch(tx, rows: list[dict]):
    return tx.run(IMPORT_QUERY, rows=rows).consume()


def main() -> None:
    from src.graph.grls_dictionary import (
        DEFAULT_DICTIONARY, import_dictionary, link_dictionary_sources, prepare_dictionary,
    )

    parser = argparse.ArgumentParser(description="Import the normalized drug dictionary; optionally import individual source records.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--records", action="store_true", help="Import individual source records instead of the unique dictionary")
    parser.add_argument("--link-records", type=Path, default=None, help="Attach already imported source records to exact dictionary mappings")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--database", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without connecting to Neo4j")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.records and args.link_records:
        parser.error("--link-records applies to the dictionary import, not --records")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    input_path = args.input or (DEFAULT_INPUT if args.records else DEFAULT_DICTIONARY)
    if args.dry_run:
        prepare = prepare_records if args.records else prepare_dictionary
        _, report = prepare(input_path)
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        return
    # Import local settings only when a connection is actually needed.
    from src.graph.connection import graph_db

    try:
        graph_db.driver.verify_connectivity()
        importer = import_grls if args.records else import_dictionary
        report = importer(graph_db.driver, input_path, batch_size=args.batch_size, database=args.database)
        result = report.model_dump()
        if args.link_records:
            result["linked_sources"] = link_dictionary_sources(
                graph_db.driver, input_path, args.link_records,
                batch_size=args.batch_size, database=args.database,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        graph_db.close()


if __name__ == "__main__":
    main()
