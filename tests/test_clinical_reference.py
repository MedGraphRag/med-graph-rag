"""Проверки разметки настоящих КР без обращений к языковой модели.

Ответ модели подменяется эталонным утверждением. Это проверка сохранности
данных в нашем коде, а не оценка качества извлечения самой моделью.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest

from src.nlp.clinical_extractor import extract_relations
from src.nlp.clinical_relations import ClinicalFragment, ClinicalRelation, ExtractionResult

BASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_PATH = Path(__file__).with_name("clinical_reference_28_3.json")
REFERENCE = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
EXAMPLES = REFERENCE["examples"]


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda item: item["id"])
def test_reference_annotations_have_evidence_and_conditions(example):
    fragment = ClinicalFragment.model_validate(example["fragment"])
    relation = ClinicalRelation.model_validate(example["expected"])

    assert relation.source.document_id == fragment.document_id
    assert relation.source.page_number == fragment.page_number
    assert relation.source.location == fragment.location
    assert relation.source.quote in fragment.text
    assert relation.subject.name.casefold() in relation.source.quote.casefold()
    assert relation.object.name.casefold() in relation.source.quote.casefold()
    assert all(condition in relation.source.quote for condition in relation.conditions)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda item: item["id"])
def test_extractor_preserves_reference_statement(example):
    fragment = ClinicalFragment.model_validate(example["fragment"])
    expected = ClinicalRelation.model_validate(example["expected"])
    client = Mock()
    client.chat.completions.create.return_value = ExtractionResult(
        relations=[expected.model_copy(deep=True)],
    )

    actual = extract_relations(fragment, client=client, model="offline-test")

    # Включая отрицание, точные пробелы, дозу и временные ограничения.
    assert actual.relations[0].model_dump() == expected.model_dump()
    assert actual.relations[0].status == "unresolved"


@pytest.fixture(scope="module")
def source_pages():
    path = BASE_DIR / REFERENCE["prepared_pages"]
    if not path.exists():
        pytest.skip("Для сверки цитат нужен локальный cr_pages.parquet")
    return pl.read_parquet(path)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda item: item["id"])
def test_reference_quotes_match_original_prepared_pages(source_pages, example):
    source = example["expected"]["source"]
    rows = source_pages.filter(
        (pl.col("code_version") == source["document_id"])
        & (pl.col("page_number") == source["page_number"])
    )

    assert rows.height == 1
    text = rows["text"][0]
    assert example["fragment"]["text"] in text
    assert text[example["quote_start"]:example["quote_end"]] == source["quote"]


def test_reference_pdf_version_has_not_changed():
    path = BASE_DIR / REFERENCE["source_pdf"]
    if not path.exists():
        pytest.skip("Для сверки версии нужен локальный PDF 28_3")

    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE["source_pdf_sha256"]
