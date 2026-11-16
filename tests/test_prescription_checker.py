import polars as pl
import pytest
from pydantic import ValidationError

from src.graph.prescription_checker import (
    ConfirmedContraindication,
    ConfirmedInteraction,
    GuidelineSource,
    check_drug_disease,
    check_drug_disease_by_name,
    check_drug_drug,
    check_drug_drug_by_name,
)

from src.nlp.grls_resolver import DrugResolver


@pytest.fixture
def relation() -> ConfirmedContraindication:
    return ConfirmedContraindication(
        drug_inn="test-inn",
        icd10_code="Z99.9",
        conditions=("synthetic condition",),
        source=GuidelineSource(
            document_id="test-guideline",
            location="page 2",
            quote="  Synthetic test quote.  ",
        ),
    )


def test_exact_pair_returns_conditions_and_source(relation):
    result = check_drug_disease("test-inn", "Z99.9", [relation])

    assert result.model_dump() == {
        "status": "confirmed_match",
        "matches": (
            {
                "drug_inn": "test-inn",
                "icd10_code": "Z99.9",
                "conditions": ("synthetic condition",),
                "source": {
                    "document_id": "test-guideline",
                    "location": "page 2",
                    "quote": "  Synthetic test quote.  ",
                },
            },
        ),
    }


@pytest.mark.parametrize(
    ("inn", "code"),
    [
        ("another-inn", "Z99.9"),
        ("test-inn", "Z99"),
        ("test-inn", None),
        (None, "Z99.9"),
        (" ", "Z99.9"),
    ],
)
def test_unmatched_or_missing_pair_is_insufficient_data(inn, code, relation):
    result = check_drug_disease(inn, code, [relation])

    assert result.status == "insufficient_data"
    assert result.matches == ()


def test_empty_relation_store_is_insufficient_data():
    result = check_drug_disease("test-inn", "Z99.9", [])

    assert result.status == "insufficient_data"


def test_source_requires_a_nonblank_quote():
    with pytest.raises(ValidationError, match="quote"):
        GuidelineSource(document_id="test-guideline", location="page 2", quote=" ")


@pytest.fixture
def interaction() -> ConfirmedInteraction:
    return ConfirmedInteraction(
        first_drug_inn="test-inn-a",
        second_drug_inn="test-inn-b",
        description="Тестовое МНН A изменяет действие тестового МНН B.",
        conditions=("Синтетическое условие применимости.",),
        source=GuidelineSource(
            document_id="test-guideline",
            location="Тестовая страница 3",
            quote="  Синтетическая цитата о взаимодействии A и B.  ",
        ),
    )


@pytest.mark.parametrize(
    ("first_inn", "second_inn"),
    [
        ("test-inn-a", "test-inn-b"),
        ("test-inn-b", "test-inn-a"),
        (" test-inn-a ", " test-inn-b "),
    ],
)
def test_interaction_preserves_participants_description_and_source(
    first_inn, second_inn, interaction
):
    result = check_drug_drug(first_inn, second_inn, [interaction])

    assert result.model_dump() == {
        "status": "confirmed_match",
        "matches": (interaction.model_dump(),),
    }


@pytest.mark.parametrize(
    ("first_inn", "second_inn"),
    [
        ("test-inn-a", "another-inn"),
        ("another-inn", "test-inn-b"),
        ("test-inn-a", "test-inn-a"),
        (None, "test-inn-b"),
        ("test-inn-a", None),
        (" ", "test-inn-b"),
        ("test-inn-a", " "),
    ],
)
def test_unmatched_or_missing_drug_pair_is_insufficient_data(
    first_inn, second_inn, interaction
):
    result = check_drug_drug(first_inn, second_inn, [interaction])

    assert result.status == "insufficient_data"
    assert result.matches == ()


def test_empty_interaction_store_is_insufficient_data():
    result = check_drug_drug("test-inn-a", "test-inn-b", [])

    assert result.status == "insufficient_data"
    assert result.matches == ()


def test_all_matching_interaction_sources_are_preserved(interaction):
    second_relation = ConfirmedInteraction(
        first_drug_inn="test-inn-b",
        second_drug_inn="test-inn-a",
        description="Другое синтетическое описание взаимодействия.",
        source=GuidelineSource(
            document_id="test-guideline-2",
            location="Тестовая страница 7",
            quote="Другая синтетическая цитата.",
        ),
    )
    unrelated = ConfirmedInteraction(
        first_drug_inn="test-inn-a",
        second_drug_inn="test-inn-c",
        description="Синтетическая связь с другим препаратом.",
        source=interaction.source,
    )
    result = check_drug_drug(
        "test-inn-a", "test-inn-b", iter([interaction, unrelated, second_relation])
    )

    assert result.matches == (interaction, second_relation)


def test_interaction_requires_a_nonblank_description(interaction):
    data = interaction.model_dump()
    data["description"] = " "

    with pytest.raises(ValidationError, match="description"):
        ConfirmedInteraction.model_validate(data)


@pytest.fixture
def drug_resolver(tmp_path) -> DrugResolver:
    # Синтетический справочник содержит разные МНН с одинаковым кодом АТХ.
    data = pl.DataFrame(
        {
            "trade_name_norm": [
                "однозначный препарат",
                "однозначный препарат",
                "тестпрепарат 10",
                "тестпрепарат 20",
                "другой препарат",
                "без мнн",
            ],
            "inn_norm": [
                "test-inn", "test-inn", "test-inn", "test-inn", "another-inn", ""
            ],
            "atx_code": [
                ["TEST01"], ["TEST02"], ["TEST01"],
                ["TEST01"], ["TEST01"], ["TEST01"],
            ],
        }
    )
    path = tmp_path / "synthetic_drugs.parquet"
    data.write_parquet(path)
    return DrugResolver(path)


def test_unambiguous_drug_name_uses_resolved_inn(drug_resolver, relation):
    result = check_drug_disease_by_name(
        "ОДНОЗНАЧНЫЙ ПРЕПАРАТ", "Z99.9", drug_resolver, [relation]
    )

    assert result.status == "confirmed_match"
    assert result.matches == (relation,)
    assert len(result.drug_search.matches) == 2
    assert result.drug_search.ambiguous is False


def test_ambiguous_name_returns_candidates_without_confirmed_matches(
    drug_resolver, relation
):
    result = check_drug_disease_by_name(
        "тестпрепарат", "Z99.9", drug_resolver, [relation]
    )

    assert result.status == "ambiguous"
    assert result.matches == ()
    assert {
        match.trade_name for match in result.drug_search.matches
    } == {"тестпрепарат 10", "тестпрепарат 20"}
    assert result.model_dump()["status"] == "ambiguous"


@pytest.mark.parametrize("name", ["неизвестный препарат", " ", "без мнн"])
def test_unknown_or_missing_inn_is_insufficient_data(
    name, drug_resolver, relation
):
    result = check_drug_disease_by_name(
        name, "Z99.9", drug_resolver, [relation]
    )

    assert result.status == "insufficient_data"
    assert result.matches == ()


def test_atx_match_alone_does_not_confirm_contraindication(drug_resolver, relation):
    result = check_drug_disease_by_name(
        "другой препарат", "Z99.9", drug_resolver, [relation]
    )

    assert result.drug_search.matches[0].atx_codes == ["TEST01"]
    assert result.drug_search.matches[0].inn == "another-inn"
    assert result.status == "insufficient_data"
    assert result.matches == ()


def test_resolved_drug_with_missing_disease_is_insufficient_data(
    drug_resolver, relation
):
    result = check_drug_disease_by_name(
        "однозначный препарат", None, drug_resolver, [relation]
    )

    assert result.status == "insufficient_data"
    assert result.matches == ()


@pytest.fixture
def named_interaction(interaction) -> ConfirmedInteraction:
    return ConfirmedInteraction(
        first_drug_inn="test-inn",
        second_drug_inn="another-inn",
        description=interaction.description,
        conditions=interaction.conditions,
        source=interaction.source,
    )


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("однозначный препарат", "другой препарат"),
        ("другой препарат", "ОДНОЗНАЧНЫЙ ПРЕПАРАТ"),
    ],
)
def test_drug_names_resolve_to_sourced_interaction(
    first_name, second_name, drug_resolver, named_interaction
):
    result = check_drug_drug_by_name(
        first_name, second_name, drug_resolver, [named_interaction]
    )

    assert result.status == "confirmed_match"
    assert result.matches == (named_interaction,)
    assert result.model_dump()["matches"] == (named_interaction.model_dump(),)


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("тестпрепарат", "другой препарат"),
        ("другой препарат", "тестпрепарат"),
        ("тестпрепарат", "тестпрепарат"),
    ],
)
def test_ambiguous_drug_pair_returns_search_candidates(
    first_name, second_name, drug_resolver, named_interaction
):
    result = check_drug_drug_by_name(
        first_name, second_name, drug_resolver, [named_interaction]
    )

    assert result.model_dump()["status"] == "ambiguous"
    assert result.matches == ()
    assert result.first_drug_search == drug_resolver.search(first_name)
    assert result.second_drug_search == drug_resolver.search(second_name)


@pytest.mark.parametrize("missing_name", ["неизвестный препарат", " ", "без мнн"])
@pytest.mark.parametrize("missing_first", [True, False])
def test_unresolved_drug_pair_is_insufficient_data(
    missing_name, missing_first, drug_resolver, named_interaction
):
    first_name, second_name = (
        (missing_name, "однозначный препарат")
        if missing_first
        else ("однозначный препарат", missing_name)
    )
    result = check_drug_drug_by_name(
        first_name, second_name, drug_resolver, [named_interaction]
    )

    assert result.status == "insufficient_data"
    assert result.matches == ()


def test_drug_names_without_stored_interaction_are_insufficient_data(drug_resolver):
    result = check_drug_drug_by_name(
        "однозначный препарат", "другой препарат", drug_resolver, []
    )

    assert result.status == "insufficient_data"
    assert result.matches == ()


def test_shared_atx_does_not_replace_second_inn(drug_resolver, named_interaction):
    result = check_drug_drug_by_name(
        "однозначный препарат", "однозначный препарат",
        drug_resolver, [named_interaction],
    )

    assert "TEST01" in {
        code
        for match in result.second_drug_search.matches
        for code in match.atx_codes
    }
    assert result.status == "insufficient_data"
    assert result.matches == ()


def test_disease_result_is_reproducible_for_different_relation_order(relation):
    other = ConfirmedContraindication(
        drug_inn=relation.drug_inn,
        icd10_code=relation.icd10_code,
        source=GuidelineSource(
            document_id="another-test-guideline",
            location="Тестовая страница 9",
            quote="  Дополнительная синтетическая цитата.  ",
        ),
        conditions=("Другое синтетическое условие.",),
    )

    forward = check_drug_disease("test-inn", "Z99.9", [relation, other])
    backward = check_drug_disease("test-inn", "Z99.9", iter([other, relation]))

    assert forward.model_dump_json() == backward.model_dump_json()
    assert len(forward.matches) == 2
    assert relation in forward.matches
    assert other in forward.matches


def test_interaction_result_is_reproducible_for_different_relation_order(interaction):
    other = ConfirmedInteraction(
        first_drug_inn=interaction.first_drug_inn,
        second_drug_inn=interaction.second_drug_inn,
        description="Дополнительное синтетическое описание взаимодействия.",
        source=GuidelineSource(
            document_id="another-test-guideline",
            location="Тестовая страница 9",
            quote="  Дополнительная синтетическая цитата.  ",
        ),
        conditions=("Другое синтетическое условие.",),
    )

    forward = check_drug_drug("test-inn-a", "test-inn-b", [interaction, other])
    backward = check_drug_drug(
        "test-inn-b", "test-inn-a", iter([other, interaction])
    )

    assert forward.model_dump_json() == backward.model_dump_json()
    assert len(forward.matches) == 2
    assert interaction in forward.matches
    assert other in forward.matches
