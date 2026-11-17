"""Проверки цитат и ограничений на вымышленных утверждениях."""

import hashlib

import pytest

from src.nlp.clinical_relations import ClinicalFragment, ClinicalRelation
from src.nlp.clinical_validation import check_relation, find_quote_span


@pytest.fixture
def sample():
    text = "Препарат А не противопоказан при болезни Б при условии В."
    fragment = ClinicalFragment(
        document_id="test", page_number=1, location="Страница 1", text=text,
    )
    relation = ClinicalRelation.model_validate({
        "subject": {"name": "Препарат А", "kind": "drug"},
        "object": {"name": "болезни Б", "kind": "disease"},
        "relation_type": "CONTRAINDICATED_FOR", "polarity": "negated",
        "conditions": ["при условии В"], "description": "Учебный пример",
        "source": {
            "document_id": "test", "page_number": 1,
            "location": "Страница 1", "quote": text,
        },
    })
    return fragment, relation


def test_keeps_negation_conditions_and_exact_offsets(sample):
    fragment, relation = sample
    original = relation.model_dump()
    checked = check_relation(fragment, relation)

    assert checked.issues == []
    assert checked.relation.polarity == "negated"
    assert checked.relation.conditions == relation.conditions
    assert checked.quote_start == 0
    assert checked.quote_end == len(fragment.text)
    assert checked.text_sha256 == hashlib.sha256(fragment.text.encode()).hexdigest()
    assert relation.model_dump() == original


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\t", "\u00a0", "   "])
def test_restores_whitespace_from_source_with_exact_offsets(sample, separator):
    fragment, relation = sample
    model_quote = relation.source.quote
    source_quote = model_quote.replace(" ", separator)
    prefix = "Введение.\n"
    fragment.text = prefix + source_quote + "\nКонец."
    original = relation.model_dump()

    checked = check_relation(fragment, relation)

    assert checked.issues == []
    assert checked.relation.status == "unresolved"
    assert checked.relation.source.quote == source_quote
    assert checked.quote_start == len(prefix)
    assert checked.quote_end == len(prefix) + len(source_quote)
    assert fragment.text[checked.quote_start:checked.quote_end] == source_quote
    assert checked.relation.polarity == "negated"
    assert checked.relation.conditions == relation.conditions
    assert relation.model_dump() == original


@pytest.mark.parametrize("original, changed", [
    ("не противопоказан", "противопоказан"),
    ("Препарат", "препарат"),
    ("В.", "В!"),
])
def test_whitespace_search_does_not_accept_changed_content(sample, original, changed):
    fragment, relation = sample
    fragment.text = fragment.text.replace(" ", "\n")
    relation.source.quote = relation.source.quote.replace(original, changed)

    checked = check_relation(fragment, relation)

    assert "quote_not_found" in checked.issues
    assert checked.relation.status == "disputed"
    assert checked.quote_start is None
    assert checked.quote_end is None


def test_whitespace_search_does_not_accept_changed_numbers():
    assert find_quote_span("АД выше 145/90", "АД выше 140/90")[2] == "quote_not_found"


def test_whitespace_search_does_not_guess_between_repeated_quotes(sample):
    fragment, relation = sample
    fragment.text = fragment.text.replace(" ", "\n")
    fragment.text += "\n" + fragment.text

    checked = check_relation(fragment, relation)

    assert "ambiguous_quote" in checked.issues
    assert checked.quote_start is None
    assert checked.quote_end is None
    assert checked.relation.source.quote == relation.source.quote


def test_whitespace_search_detects_overlapping_matches():
    assert find_quote_span("А\nА\nА", "А А") == (None, None, "ambiguous_quote")


@pytest.mark.parametrize("change, issue", [
    ("quote", "quote_not_found"),
    ("condition", "condition_not_in_quote"),
    ("subject", "subject_not_in_quote"),
    ("object", "object_not_in_quote"),
    ("uncertain", "uncertain_statement"),
    ("document", "source_mismatch"),
    ("page", "source_mismatch"),
])
def test_marks_unverifiable_claim_disputed(sample, change, issue):
    fragment, relation = sample
    if change == "quote":
        relation.source.quote = "Вымышленная цитата."
    elif change == "condition":
        relation.conditions.append("при дозе 999 мг")
    elif change == "subject":
        relation.subject.name = "Неизвестное лекарство"
    elif change == "object":
        relation.object.name = "Другое заболевание"
    elif change == "uncertain":
        relation.polarity = "uncertain"
    elif change == "document":
        relation.source.document_id = "wrong"
    else:
        relation.source.page_number = 99

    checked = check_relation(fragment, relation)

    assert issue in checked.issues
    assert checked.relation.status == "disputed"
    if issue == "source_mismatch":
        assert checked.original_source is not None
        assert checked.relation.source.document_id == fragment.document_id
        assert checked.relation.source.page_number == fragment.page_number


def test_repeated_quote_has_no_guessed_location(sample):
    fragment, relation = sample
    fragment.text += "\n" + fragment.text

    checked = check_relation(fragment, relation)

    assert "ambiguous_quote" in checked.issues
    assert checked.quote_start is None
    assert checked.quote_end is None


def test_part_of_word_is_not_an_entity_match(sample):
    fragment, relation = sample
    fragment.text = fragment.text.replace("Препарат А", "Препарат АБ")
    relation.source.quote = fragment.text

    assert "subject_not_in_quote" in check_relation(fragment, relation).issues


def test_rejected_claim_stays_rejected(sample):
    fragment, relation = sample
    relation.status = "rejected"

    assert check_relation(fragment, relation).relation.status == "rejected"
