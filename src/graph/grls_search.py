"""Resolve trade names and INNs without mixing evidence across registrations."""
from __future__ import annotations

import re

from neo4j import Driver

from src.graph.grls_models import (
    GRLSDictionaryMatch, GRLSDictionarySearchResult,
    GRLSDrugMatch, GRLSDrugSearchResult,
)
from src.nlp.grls_resolver import normalize_query, normalize_trade_query


RECORD_SEARCH_QUERY = """
MATCH (registration:GRLSRegistration)-[:HAS_RECORD]->(record:GRLSRecord)
MATCH (record)-[:HAS_TRADE_NAME]->(trade:TradeName)
WHERE ($source_snapshot IS NULL OR record.source_snapshot = $source_snapshot)
  AND (trade.name_norm = $trade_query
       OR ($match_dosages AND trade.name_norm =~ $dosage_pattern)
       OR record.inn_norm = $inn_query)
RETURN record.record_id AS record_id, registration.lp_id AS lp_id,
       record.ru_number AS ru_number, trade.name_norm AS trade_name,
       record.mnn_name AS inn, record.source_snapshot AS source_snapshot,
       record.source_pages AS source_pages,
       [(record)-[:HAS_INN]->(inn:INN) WHERE inn.code IS NOT NULL | inn.code] AS mnn_codes,
       [(record)-[:HAS_ATC]->(atc:ATC) | atc.code] AS atx_codes
ORDER BY trade_name, lp_id, source_snapshot
"""

SEARCH_QUERY = """
MATCH (:GRLSDictionary {name: $dictionary_name})-[:CONTAINS]->(drug:GRLSDrug)
WHERE (drug.trade_name_norm = $trade_query
       OR drug.trade_name_norm = $inn_query
       OR ($match_dosages AND drug.trade_name_norm =~ $dosage_pattern)
       OR drug.inn_norm = $inn_query)
RETURN drug.drug_id AS drug_id, drug.trade_name_norm AS trade_name,
       drug.inn_norm AS inn, drug.atx_code AS atx_codes
ORDER BY trade_name, inn, drug_id
"""


class GRLSGraphResolver:
    def __init__(self, driver: Driver, *, database: str | None = None, dictionary_name: str = "main"):
        self.driver = driver
        self.database = database
        self.dictionary_name = dictionary_name

    def search(self, query: str) -> GRLSDictionarySearchResult:
        """Search unique normalized mappings, independently of registration count."""
        trade_query = normalize_trade_query(query)
        inn_query = normalize_query(query)
        if not trade_query and not inn_query:
            return GRLSDictionarySearchResult(query=query)
        has_dosage = bool(re.search(r"\d+(?:[.,]\d+)?\s*(?:мкг|мг|мл|г|%|ме)", query.lower()))
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(
                lambda tx: tx.run(
                    SEARCH_QUERY, trade_query=trade_query, inn_query=inn_query,
                    dictionary_name=self.dictionary_name,
                    match_dosages=not has_dosage,
                    dosage_pattern=re.escape(trade_query) + r"\s+\d.*",
                ).data()
            )
        matches = [GRLSDictionaryMatch.model_validate(row) for row in rows]
        return GRLSDictionarySearchResult(
            query=query, matches=matches,
            ambiguous=len({(match.trade_name, match.inn) for match in matches}) > 1,
        )

    def search_records(self, query: str, *, source_snapshot: str | None = None) -> GRLSDrugSearchResult:
        """Inspect individual source observations, including historical versions."""
        trade_query = normalize_trade_query(query)
        inn_query = normalize_query(query)
        if not trade_query and not inn_query:
            return GRLSDrugSearchResult(query=query)
        has_dosage = bool(re.search(r"\d+(?:[.,]\d+)?\s*(?:мкг|мг|мл|г|%|ме)", query.lower()))
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(
                lambda tx: tx.run(
                    RECORD_SEARCH_QUERY,
                    trade_query=trade_query,
                    inn_query=inn_query,
                    match_dosages=not has_dosage,
                    dosage_pattern=re.escape(trade_query) + r"\s+\d.*",
                    source_snapshot=source_snapshot,
                ).data()
            )
        matches = []
        for row in rows:
            row["inn"] = normalize_query(row["inn"]) if row["inn"] else None
            for key in ("mnn_codes", "atx_codes", "source_pages"):
                row[key] = sorted(set(row[key]))
            matches.append(GRLSDrugMatch.model_validate(row))
        variants = {(match.trade_name, match.inn) for match in matches}
        return GRLSDrugSearchResult(query=query, matches=matches, ambiguous=len(variants) > 1)
