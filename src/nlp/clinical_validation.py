"""Проверка доказательств утверждения без обращения к языковой модели."""

import hashlib
import re

from pydantic import Field

from src.nlp.clinical_relations import (
    ClinicalFragment,
    ClinicalRelation,
    ExtractionModel,
    RelationSource,
)


class EvidenceCheck(ExtractionModel):
    fragment: ClinicalFragment
    relation: ClinicalRelation
    issues: list[str] = Field(default_factory=list)
    quote_start: int | None = None
    quote_end: int | None = None
    text_sha256: str
    original_source: RelationSource | None = None


def normalize_text(text: str) -> str:
    """Только пробелы и регистр; без раскрытия групп и изменения названий."""
    return " ".join(text.casefold().split())


def name_in_quote(name: str, quote: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(normalize_text(name)) + r"(?!\w)"
    return re.search(pattern, normalize_text(quote)) is not None


def find_quote_span(text: str, quote: str) -> tuple[int | None, int | None, str | None]:
    """Ищет цитату точно, затем с допуском только на пробельные символы."""
    start = text.find(quote)
    if start >= 0:
        if text.find(quote, start + 1) >= 0:
            return None, None, "ambiguous_quote"
        return start, start + len(quote), None

    words = quote.split()
    if not words:
        return None, None, "quote_not_found"
    pattern = re.compile(r"\s+".join(re.escape(word) for word in words))
    match = pattern.search(text)
    if match is None:
        return None, None, "quote_not_found"
    if pattern.search(text, match.start() + 1) is not None:
        return None, None, "ambiguous_quote"
    return match.start(), match.end(), None


def check_relation(fragment: ClinicalFragment, relation: ClinicalRelation) -> EvidenceCheck:
    """Проверяет буквальное доказательство, сохраняя отрицание и условия.

    Проверка не определяет медицинский смысл и не доказывает полноту условий.
    Пропущенное моделью отрицание должен выявить проверяющий или сравнение
    с эталонной разметкой.
    """
    relation = relation.model_copy(deep=True)
    issues = []
    original_source = None
    source = relation.source
    if (
        source.document_id != fragment.document_id
        or source.page_number != fragment.page_number
        or source.location != fragment.location
    ):
        issues.append("source_mismatch")
        original_source = source.model_copy(deep=True)

    quote = source.quote
    start, end, quote_issue = find_quote_span(fragment.text, quote)
    if quote_issue:
        issues.append(quote_issue)
    else:
        # Сохраняем дословный фрагмент источника, включая его переносы строк.
        quote = fragment.text[start:end]

    if not name_in_quote(relation.subject.name, quote):
        issues.append("subject_not_in_quote")
    if not name_in_quote(relation.object.name, quote):
        issues.append("object_not_in_quote")
    normalized_quote = " ".join(quote.split())
    if any(" ".join(condition.split()) not in normalized_quote
           for condition in relation.conditions):
        issues.append("condition_not_in_quote")
    if relation.polarity == "uncertain":
        issues.append("uncertain_statement")
    if relation.status == "disputed" and not issues:
        issues.append("previously_disputed")

    # Метаданные документа задаёт код. Исходная ошибка остаётся в original_source.
    relation.source = RelationSource(
        document_id=fragment.document_id,
        page_number=fragment.page_number,
        location=fragment.location,
        quote=quote,
    )
    if relation.status != "rejected":
        relation.status = "disputed" if issues else "unresolved"

    return EvidenceCheck(
        fragment=fragment.model_copy(deep=True),
        relation=relation,
        issues=issues,
        quote_start=start,
        quote_end=end,
        text_sha256=hashlib.sha256(fragment.text.encode("utf-8")).hexdigest(),
        original_source=original_source,
    )
