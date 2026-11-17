"""Проверки схемы на вымышленных примерах, без медицинских утверждений."""

import pytest
from pydantic import ValidationError

from src.nlp.clinical_relations import ClinicalRelation, ExtractionResult


def make_relation():
    return {
        "subject": {"name": "Препарат А", "kind": "drug"},
        "object": {"name": "Заболевание Б", "kind": "disease"},
        "relation_type": "CONTRAINDICATED_FOR",
        "polarity": "negated",
        "conditions": ["при условии В"],
        "description": "Учебный пример отрицания противопоказания",
        "source": {
            "document_id": "synthetic-example",
            "location": "Страница 2, учебный раздел",
            "page_number": 2,
            "quote": (
                "  Препарат А не противопоказан при заболевании Б\n"
                "при условии В.  "
            ),
        },
    }


def test_preserves_negation_conditions_and_exact_quote():
    data = make_relation()
    relation = ClinicalRelation.model_validate(data)
    restored = ClinicalRelation.model_validate_json(relation.model_dump_json())

    assert restored.polarity == "negated"
    assert restored.conditions == ["при условии В"]
    assert restored.source.quote == data["source"]["quote"]
    assert restored.status == "unresolved"


def test_interaction_and_uncertain_statement():
    data = make_relation()
    data["object"] = {"name": "Препарат Б", "kind": "drug"}
    data["relation_type"] = "INTERACTS_WITH"
    data["polarity"] = "uncertain"
    data["conditions"] = []
    data["status"] = "disputed"

    relation = ClinicalRelation.model_validate(data)

    assert relation.object.kind == "drug"
    assert relation.polarity == "uncertain"
    assert relation.conditions == []
    assert relation.status == "disputed"


@pytest.mark.parametrize("status", ["confirmed", "confirmed_match"])
def test_extraction_cannot_be_confirmed(status):
    data = make_relation()
    data["status"] = status

    with pytest.raises(ValidationError):
        ClinicalRelation.model_validate(data)


@pytest.mark.parametrize("field", ["document_id", "location", "quote"])
def test_rejects_blank_source_fields(field):
    data = make_relation()
    data["source"][field] = " \n "

    with pytest.raises(ValidationError):
        ClinicalRelation.model_validate(data)


def test_rejects_grls_as_clinical_source():
    data = make_relation()
    data["source"]["source_type"] = "grls"

    with pytest.raises(ValidationError):
        ClinicalRelation.model_validate(data)


@pytest.mark.parametrize(
    "relation_type, subject_kind, object_kind",
    [
        ("CONTRAINDICATED_FOR", "disease", "disease"),
        ("CONTRAINDICATED_FOR", "drug", "drug"),
        ("INTERACTS_WITH", "drug", "disease"),
    ],
)
def test_rejects_wrong_participant_types(relation_type, subject_kind, object_kind):
    data = make_relation()
    data["relation_type"] = relation_type
    data["subject"]["kind"] = subject_kind
    data["object"]["kind"] = object_kind

    with pytest.raises(ValidationError):
        ClinicalRelation.model_validate(data)


@pytest.mark.parametrize("field", ["polarity", "conditions"])
def test_requires_explicit_polarity_and_conditions(field):
    data = make_relation()
    del data[field]

    with pytest.raises(ValidationError):
        ClinicalRelation.model_validate(data)


def test_empty_result_requires_explicit_list():
    assert ExtractionResult(relations=[]).relations == []

    with pytest.raises(ValidationError):
        ExtractionResult.model_validate({})
