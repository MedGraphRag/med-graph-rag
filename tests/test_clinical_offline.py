"""Весь путь от сохранённого ответа до допуска к экспорту, без API и Neo4j."""

from datetime import datetime
import json

import polars as pl
import pytest
from pydantic import ValidationError

from src.nlp.clinical_offline import (
    ClinicalSample, ReviewReport, compare_reference, main, prepare_report,
    review_draft, write_new_json,
)
from src.nlp.clinical_review import ReviewDecision, ReviewDecisions, export_reviewed
from src.nlp.grls_resolver import DrugResolver


@pytest.fixture
def resolver(tmp_path):
    path = tmp_path / "drugs.parquet"
    pl.DataFrame({
        "trade_name_norm": ["препарат а", "препарат б", "двойное", "двойное"],
        "inn_norm": ["вещество а", "вещество б", "вещество а", "вещество б"],
        "atx_code": [["TEST01"], ["TEST02"], ["TEST01"], ["TEST02"]],
    }).write_parquet(path)
    return DrugResolver(path)


@pytest.fixture
def sample():
    text = "Препарат А противопоказан при болезни Б при условии В."
    return ClinicalSample.model_validate({
        "fragment": {
            "document_id": "test_1", "page_number": 1,
            "location": "Страница 1", "text": text,
        },
        "relations": [{
            "subject": {"name": "Препарат А", "kind": "drug"},
            "object": {"name": "болезни Б", "kind": "disease"},
            "relation_type": "CONTRAINDICATED_FOR", "polarity": "affirmed",
            "conditions": ["при условии В"], "description": "Вымышленный пример",
            "source": {
                "document_id": "test_1", "page_number": 1,
                "location": "Страница 1", "quote": text,
            },
        }],
    })


def make_pages(sample):
    return pl.DataFrame({
        "code_version": [sample.fragment.document_id],
        "page_number": [sample.fragment.page_number],
        "text": [sample.fragment.text],
    })


def make_report(sample, resolver):
    return prepare_report([sample], resolver, make_pages(sample), "saved_response")


def approve(report):
    return ReviewDecisions(reviews=[ReviewDecision(
        claim_id=report.entries[0].id, decision="confirmed", reviewer="test reviewer",
        reviewed_at=datetime.fromisoformat("2026-11-17T12:00:00+07:00"),
        pdf_checked=True, participants_checked=True,
        polarity_checked=True, conditions_checked=True, icd10_code="A00",
    )])


def test_report_has_stable_identity_and_checked_evidence(sample, resolver):
    report = make_report(sample, resolver)
    again = make_report(sample, resolver)

    assert report.entries[0].id == again.entries[0].id
    assert report.entries[0].claim.relation.status == "auto_checked"
    assert report.entries[0].claim.issues == []
    assert report.entries[0].claim.quote_end == len(sample.fragment.text)
    assert report.manual_review == "not_requested"


@pytest.mark.parametrize("separator", [" ", "\n", "\u00a0"])
def test_quote_has_exact_offsets_in_full_source_page(sample, resolver, separator):
    sample.fragment.text = sample.fragment.text.replace(" ", separator)
    prefix = "Введение.\n"
    page_text = prefix + sample.fragment.text
    pages = make_pages(sample).with_columns(pl.lit(page_text).alias("text"))

    report = prepare_report([sample], resolver, pages, "saved_response")

    claim = report.entries[0].claim
    assert claim.issues == []
    assert claim.relation.status == "auto_checked"
    assert claim.quote_start == 0
    assert claim.page_quote_start == len(prefix)
    assert page_text[claim.page_quote_start:claim.page_quote_end] == claim.relation.source.quote


@pytest.mark.parametrize("change", ["missing_page", "changed_text", "duplicate_page", "repeated_fragment"])
def test_unverified_source_blocks_preparation(sample, resolver, change):
    pages = make_pages(sample)
    if change == "missing_page":
        pages = pages.with_columns(pl.lit(99).alias("page_number"))
    elif change == "changed_text":
        pages = pages.with_columns(pl.lit("Другой текст.").alias("text"))
    elif change == "repeated_fragment":
        pages = pages.with_columns(pl.lit(sample.fragment.text * 2).alias("text"))
    else:
        pages = pl.concat([pages, pages])

    with pytest.raises(ValueError):
        prepare_report([sample], resolver, pages, "saved_response")


def test_duplicates_are_not_extra_review_entries(sample, resolver):
    report = prepare_report([sample, sample], resolver, make_pages(sample), "saved_response")

    assert len(report.entries) == 1


def test_opposite_polarities_with_same_conditions_are_disputed(sample, resolver):
    other = sample.model_copy(deep=True)
    other.relations[0].polarity = "negated"
    report = prepare_report([sample, other], resolver, make_pages(sample), "saved_response")

    assert len(report.entries) == 2
    assert all(entry.claim.relation.status == "disputed" for entry in report.entries)
    assert all("contradictory_polarity" in entry.claim.issues for entry in report.entries)


def test_different_conditions_are_not_automatically_a_contradiction(sample, resolver):
    other = sample.model_copy(deep=True)
    other.relations[0].polarity = "negated"
    other.relations[0].conditions = []

    report = prepare_report([sample, other], resolver, make_pages(sample), "saved_response")

    assert all("contradictory_polarity" not in entry.claim.issues for entry in report.entries)


@pytest.mark.parametrize("name, issue", [
    ("Неизвестное лекарство", "subject_not_found"),
    ("двойное", "subject_ambiguous"),
])
def test_unknown_or_ambiguous_name_keeps_variants_and_blocks_export(sample, resolver, name, issue):
    sample.fragment.text = sample.fragment.text.replace("Препарат А", name)
    sample.relations[0].source.quote = sample.fragment.text
    sample.relations[0].subject.name = name
    report = make_report(sample, resolver)

    assert report.entries[0].claim.relation.status == "unresolved"
    assert issue in report.entries[0].claim.issues
    with pytest.raises(ValueError, match="неразрешённое"):
        export_reviewed(report, approve(report), resolver, make_pages(sample), {"A00"})


def test_empty_review_draft_exports_no_confirmed_relations(sample, resolver):
    report = make_report(sample, resolver)
    decisions = ReviewDecisions.model_validate(review_draft(report))
    result = export_reviewed(report, decisions, resolver, make_pages(sample), {"A00"})

    assert result["contraindications"] == []
    assert result["interactions"] == []
    assert result["deferred"][0]["status"] == "pending"


def test_complete_manual_review_exports_contract_with_original_conditions(sample, resolver):
    report = make_report(sample, resolver)
    result = export_reviewed(report, approve(report), resolver, make_pages(sample), {"A00"})

    confirmed = result["contraindications"][0]
    assert confirmed["status"] == "confirmed"
    assert confirmed["drug_inn"] == "вещество а"
    assert confirmed["icd10_code"] == "A00"
    assert confirmed["conditions"] == ["при условии В"]
    assert confirmed["source"]["quote"] == sample.fragment.text
    assert confirmed["review"]["reviewer"] == "test reviewer"


@pytest.mark.parametrize("flag", [
    "pdf_checked", "participants_checked", "polarity_checked", "conditions_checked",
])
def test_each_manual_check_is_required(sample, resolver, flag):
    report = make_report(sample, resolver)
    decisions = approve(report)
    setattr(decisions.reviews[0], flag, False)

    with pytest.raises(ValueError, match="не завершена"):
        export_reviewed(report, decisions, resolver, make_pages(sample), {"A00"})


@pytest.mark.parametrize("field, value", [("reviewer", " "), ("reviewed_at", None)])
def test_confirmation_requires_author_and_date(sample, resolver, field, value):
    report = make_report(sample, resolver)
    decisions = approve(report)
    setattr(decisions.reviews[0], field, value)

    with pytest.raises(ValueError, match="проверяющий и дата"):
        export_reviewed(report, decisions, resolver, make_pages(sample), {"A00"})


def test_negated_relation_cannot_be_exported_as_existing(sample, resolver):
    sample.relations[0].polarity = "negated"
    report = make_report(sample, resolver)

    with pytest.raises(ValueError, match="Отрицаемая"):
        export_reviewed(report, approve(report), resolver, make_pages(sample), {"A00"})


def test_invented_condition_blocks_confirmation(sample, resolver):
    sample.relations[0].conditions.append("при дозе 999 мг")
    report = make_report(sample, resolver)

    with pytest.raises(ValueError, match="Спорное"):
        export_reviewed(report, approve(report), resolver, make_pages(sample), {"A00"})


def test_confirmation_requires_known_icd10_code(sample, resolver):
    report = make_report(sample, resolver)

    with pytest.raises(ValueError, match="МКБ-10"):
        export_reviewed(report, approve(report), resolver, make_pages(sample), set())


def test_edited_report_invalidates_previous_approval(sample, resolver):
    report = make_report(sample, resolver)
    decisions = approve(report)
    report.entries[0].claim.relation.conditions = []

    with pytest.raises(ValueError, match="Отчёт изменён"):
        export_reviewed(report, decisions, resolver, make_pages(sample), {"A00"})


def test_changed_registry_requires_new_review(sample, resolver):
    report = make_report(sample, resolver)
    decisions = approve(report)
    resolver.df = resolver.df.with_columns(
        pl.when(pl.col("trade_name_norm") == "препарат а")
        .then(pl.lit("другое вещество")).otherwise(pl.col("inn_norm")).alias("inn_norm")
    )

    with pytest.raises(ValueError, match="нужна новая проверка"):
        export_reviewed(report, decisions, resolver, make_pages(sample), {"A00"})


def test_changed_page_invalidates_approval_even_when_quote_is_unchanged(sample, resolver):
    report = make_report(sample, resolver)
    decisions = approve(report)
    pages = make_pages(sample).with_columns(
        pl.lit(sample.fragment.text + "\nДополнение к странице.").alias("text"),
    )

    with pytest.raises(ValueError, match="нужна новая проверка"):
        export_reviewed(report, decisions, resolver, pages, {"A00"})


def test_review_requires_real_boolean_and_timezone(sample, resolver):
    report = make_report(sample, resolver)
    payload = approve(report).model_dump(mode="json")
    payload["reviews"][0]["pdf_checked"] = "true"
    with pytest.raises(ValidationError):
        ReviewDecisions.model_validate(payload)
    payload["reviews"][0]["pdf_checked"] = True
    payload["reviews"][0]["reviewed_at"] = "2026-11-17T12:00:00"
    with pytest.raises(ValidationError):
        ReviewDecisions.model_validate(payload)


def test_unknown_and_duplicate_review_ids_are_rejected(sample, resolver):
    report = make_report(sample, resolver)
    decisions = approve(report)
    decisions.reviews.append(decisions.reviews[0].model_copy())
    with pytest.raises(ValueError, match="несколько решений"):
        export_reviewed(report, decisions, resolver, make_pages(sample), {"A00"})
    decisions.reviews = [decisions.reviews[0]]
    decisions.reviews[0].claim_id = "wrong-id"
    with pytest.raises(ValueError, match="неизвестному"):
        export_reviewed(report, decisions, resolver, make_pages(sample), {"A00"})


def test_interaction_exports_both_drugs_without_icd10(sample, resolver):
    relation = sample.relations[0]
    relation.relation_type = "INTERACTS_WITH"
    relation.object.kind = "drug"
    relation.object.name = "Препарат Б"
    sample.fragment.text = "Препарат А взаимодействует с Препарат Б при условии В."
    relation.source.quote = sample.fragment.text
    report = make_report(sample, resolver)

    result = export_reviewed(report, approve(report), resolver, make_pages(sample), set())

    assert result["interactions"][0]["first_drug_inn"] == "вещество а"
    assert result["interactions"][0]["second_drug_inn"] == "вещество б"


@pytest.mark.parametrize("change", ["polarity", "conditions"])
def test_reference_comparison_detects_lost_negation_or_conditions(sample, resolver, tmp_path, change):
    reference = tmp_path / "reference.json"
    write_new_json(reference, {"examples": [{
        "id": "example", "expected": sample.relations[0].model_dump(),
    }]})
    if change == "polarity":
        sample.relations[0].polarity = "negated"
    else:
        sample.relations[0].conditions = []

    result = compare_reference(reference, make_report(sample, resolver))

    assert result["examples"][0]["status"] == "different"


@pytest.mark.parametrize("make_review", [False, True])
def test_prepare_command_works_with_saved_response_without_model_config(sample, resolver, tmp_path, monkeypatch, make_review):
    for name in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    input_path = tmp_path / "input.json"
    write_new_json(input_path, {
        "fragment": sample.fragment.model_dump(),
        "relations": [{"relation": sample.relations[0].model_dump()}],
    })
    pages_path = tmp_path / "pages.parquet"
    make_pages(sample).write_parquet(pages_path)
    output_dir = tmp_path / "review"

    arguments = [
        "prepare", "--input", str(input_path), "--pages", str(pages_path),
        "--dictionary", str(resolver.parquet_path), "--output-dir", str(output_dir),
    ]
    if make_review:
        arguments.append("--review-draft")
    assert main(arguments) == 0
    report = ReviewReport.model_validate_json((output_dir / "claims.json").read_text(encoding="utf-8"))
    assert len(report.entries) == 1
    assert report.entries[0].claim.relation.status == "auto_checked"
    assert (output_dir / "reviews.json").exists() == make_review
    assert report.manual_review == ("not_reviewed" if make_review else "not_requested")
    if make_review:
        reviews = ReviewDecisions.model_validate_json((output_dir / "reviews.json").read_text(encoding="utf-8"))
        assert reviews.reviews[0].decision == "pending"


def test_existing_results_are_never_overwritten(tmp_path):
    path = tmp_path / "result.json"
    write_new_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"original": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"original": True}
