"""Чтение подтверждённых клинических связей из Neo4j."""

from neo4j import Driver

from src.graph.prescription_checker import (
    ConfirmedContraindication,
    ConfirmedInteraction,
    DrugDiseaseNameCheckResult,
    DrugDrugNameCheckResult,
    DrugNameResolver,
    check_drug_disease,
    check_drug_drug,
)

CONTRAINDICATION_QUERY = """
MATCH (inn:INN {key: $inn_key})-[r:CONTRAINDICATED_FOR]->
      (disease:ICD10 {code: $icd10_code})
WHERE r.status = 'confirmed'
RETURN r.status AS status, {
    drug_inn: inn.name_norm,
    icd10_code: disease.code,
    conditions: r.conditions,
    source: {
        document_id: r.document_id, location: r.location, quote: r.quote
    }
} AS relation
"""

INTERACTION_QUERY = """
MATCH (first:INN)-[r:INTERACTS_WITH]->(second:INN)
WHERE r.status = 'confirmed'
  AND ((first.key = $first_key AND second.key = $second_key)
       OR (first.key = $second_key AND second.key = $first_key))
RETURN r.status AS status, {
    first_drug_inn: first.name_norm,
    second_drug_inn: second.name_norm,
    description: r.description,
    conditions: r.conditions,
    source: {
        document_id: r.document_id, location: r.location, quote: r.quote
    }
} AS relation
"""


class Neo4jPrescriptionChecker:
    """Соединяет поиск названий с чтением клинических связей.

    Идентификаторы МНН совместимы с импортом ГРЛС из PR #32.
    Подключение и ошибки валидации передаются вызывающему коду.
    """

    def __init__(
        self,
        driver: Driver,
        resolver: DrugNameResolver,
        *,
        database: str | None = None,
    ):
        self.driver = driver
        self.resolver = resolver
        self.database = database

    def _read(self, query: str, **parameters) -> list[dict]:
        # Результат полностью читается внутри управляемой транзакции.
        with self.driver.session(
            database=self.database, default_access_mode="READ"
        ) as session:
            return session.execute_read(
                lambda tx: tx.run(query, **parameters).data()
            )

    def check_drug_disease(
        self, drug_name: str, icd10_code: str | None
    ) -> DrugDiseaseNameCheckResult:
        """Возвращает противопоказания для названия лекарства и кода МКБ-10."""
        search = self.resolver.search(drug_name)
        if (
            search.ambiguous
            or not search.matches
            or not search.matches[0].inn
            or not search.matches[0].inn.strip()
            or not icd10_code
            or not icd10_code.strip()
        ):
            return DrugDiseaseNameCheckResult(drug_search=search)

        inn = search.matches[0].inn.strip()
        code = icd10_code.strip()
        rows = self._read(
            CONTRAINDICATION_QUERY, inn_key="name:" + inn, icd10_code=code
        )
        relations = [
            ConfirmedContraindication.model_validate(row["relation"])
            for row in rows
            if row["status"] == "confirmed"
        ]
        result = check_drug_disease(inn, code, relations)
        return DrugDiseaseNameCheckResult(
            drug_search=search, matches=result.matches
        )

    def check_drug_drug(
        self, first_drug_name: str, second_drug_name: str
    ) -> DrugDrugNameCheckResult:
        """Возвращает взаимодействия двух лекарств с условиями и источниками."""
        first = self.resolver.search(first_drug_name)
        second = self.resolver.search(second_drug_name)
        if (
            first.ambiguous
            or second.ambiguous
            or not first.matches
            or not second.matches
            or not first.matches[0].inn
            or not first.matches[0].inn.strip()
            or not second.matches[0].inn
            or not second.matches[0].inn.strip()
        ):
            return DrugDrugNameCheckResult(
                first_drug_search=first, second_drug_search=second
            )

        first_inn = first.matches[0].inn.strip()
        second_inn = second.matches[0].inn.strip()
        rows = self._read(
            INTERACTION_QUERY,
            first_key="name:" + first_inn,
            second_key="name:" + second_inn,
        )
        relations = [
            ConfirmedInteraction.model_validate(row["relation"])
            for row in rows
            if row["status"] == "confirmed"
        ]
        result = check_drug_drug(first_inn, second_inn, relations)
        return DrugDrugNameCheckResult(
            first_drug_search=first, second_drug_search=second,
            matches=result.matches,
        )
