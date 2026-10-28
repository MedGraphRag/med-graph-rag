"""Additive Neo4j 5 schema; ICD10/PARENT_OF are deliberately independent."""
from neo4j import Driver


CONSTRAINTS = (
    "CREATE CONSTRAINT grls_dictionary_name IF NOT EXISTS "
    "FOR (n:GRLSDictionary) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT grls_drug_id IF NOT EXISTS "
    "FOR (n:GRLSDrug) REQUIRE n.drug_id IS UNIQUE",
    "CREATE CONSTRAINT grls_registration_id IF NOT EXISTS "
    "FOR (n:GRLSRegistration) REQUIRE n.lp_id IS UNIQUE",
    "CREATE CONSTRAINT grls_record_id IF NOT EXISTS "
    "FOR (n:GRLSRecord) REQUIRE n.record_id IS UNIQUE",
    "CREATE CONSTRAINT grls_trade_name IF NOT EXISTS "
    "FOR (n:TradeName) REQUIRE n.name_norm IS UNIQUE",
    "CREATE CONSTRAINT grls_inn_key IF NOT EXISTS "
    "FOR (n:INN) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT grls_atc_code IF NOT EXISTS "
    "FOR (n:ATC) REQUIRE n.code IS UNIQUE",
)


def create_grls_constraints(driver: Driver, *, database: str | None = None) -> None:
    with driver.session(database=database) as session:
        for query in CONSTRAINTS:
            session.run(query).consume()
