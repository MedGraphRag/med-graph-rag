"""Сопоставление на небольшом вымышленном справочнике ГРЛС."""

import polars as pl
import pytest

from src.nlp.clinical_relations import ClinicalRelation
from src.nlp.clinical_resolver import resolve_drug, resolve_relation_drugs
from src.nlp.grls_resolver import DrugResolver


@pytest.fixture
def resolver(tmp_path):
    data = pl.DataFrame({
        "trade_name_norm": [
            "препарат а", "препарат а", "препарат б", "общее название",
            "общее название", "без мнн", "комбинация а",
        ],
        "inn_norm": [
            "вещество а", "вещество а", "вещество б", "вещество в",
            "вещество г", "", "вещество а + вещество б",
        ],
        "atx_code": [["TEST01"], ["TEST02"], ["TEST03"], ["TEST04"],
                     ["TEST05"], [], ["TEST06"]],
    })
    path = tmp_path / "drugs.parquet"
    data.write_parquet(path)
    return DrugResolver(path)


@pytest.fixture
def relation():
    return ClinicalRelation.model_validate({
        "subject": {"name": "Препарат А", "kind": "drug"},
        "object": {"name": "заболевании Б", "kind": "disease"},
        "relation_type": "CONTRAINDICATED_FOR",
        "polarity": "negated",
        "conditions": ["при условии В"],
        "description": "Вымышленное утверждение для теста",
        "source": {
            "document_id": "synthetic-example",
            "location": "Страница 2, учебный раздел",
            "page_number": 2,
            "quote": "Препарат А не противопоказан при заболевании Б при условии В.",
        },
        "status": "unresolved",
    })


def test_resolves_name_and_preserves_all_atx_codes(resolver):
    result = resolve_drug("Препарат А", resolver)

    assert result.status == "resolved"
    assert result.inn == "вещество а"
    assert result.atx_codes == ["TEST01", "TEST02"]
    assert len(result.search.matches) == 2


@pytest.mark.parametrize("name, status", [
    ("неизвестный препарат", "not_found"),
    ("общее название", "ambiguous"),
    ("без мнн", "missing_inn"),
])
def test_does_not_choose_an_unresolved_name(resolver, name, status):
    result = resolve_drug(name, resolver)

    assert result.status == status
    assert result.inn is None
    assert result.atx_codes == []
    if status == "ambiguous":
        assert len(result.search.matches) == 2


def test_keeps_combination_as_one_inn(resolver):
    result = resolve_drug("комбинация а", resolver)

    assert result.inn == "вещество а + вещество б"


def test_preserves_clinical_statement_and_does_not_change_input(resolver, relation):
    original = relation.model_dump()
    result = resolve_relation_drugs(relation, resolver)

    assert relation.model_dump() == original
    assert result.relation.status == "auto_checked"
    assert result.relation.model_dump(exclude={"status"}) == relation.model_dump(
        exclude={"status"},
    )
    assert result.subject_drug.inn == "вещество а"
    assert result.object_drug is None  # Заболевание не ищется в ГРЛС.


@pytest.mark.parametrize("second_name, expected_status", [
    ("Препарат Б", "auto_checked"),
    ("общее название", "unresolved"),
    ("неизвестный препарат", "unresolved"),
])
def test_resolves_both_drugs_in_interaction(resolver, relation, second_name, expected_status):
    relation.relation_type = "INTERACTS_WITH"
    relation.object.kind = "drug"
    relation.object.name = second_name

    result = resolve_relation_drugs(relation, resolver)

    assert result.relation.status == expected_status
    assert result.object_drug.search.query == second_name
    if expected_status == "auto_checked":
        assert result.object_drug.inn == "вещество б"
    else:
        assert result.object_drug.inn is None


@pytest.mark.parametrize("status", ["disputed", "rejected"])
def test_resolution_does_not_promote_disputed_or_rejected_statement(resolver, relation, status):
    relation.status = status

    result = resolve_relation_drugs(relation, resolver)

    assert result.subject_drug.status == "resolved"
    assert result.relation.status == status


def test_unresolved_first_drug_keeps_relation_unresolved(resolver, relation):
    relation.subject.name = "общее название"

    result = resolve_relation_drugs(relation, resolver)

    assert result.relation.status == "unresolved"
    assert result.subject_drug.status == "ambiguous"


def test_uncertain_statement_stays_disputed_after_name_resolution(resolver, relation):
    relation.polarity = "uncertain"

    result = resolve_relation_drugs(relation, resolver)

    assert result.relation.status == "disputed"
