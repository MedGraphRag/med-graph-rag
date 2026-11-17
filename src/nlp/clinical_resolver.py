"""Сопоставление участников клинических утверждений с лекарствами ГРЛС."""

from typing import Literal

from pydantic import Field

from src.nlp.clinical_relations import (
    ClinicalRelation,
    ExtractionModel,
    NonBlankText,
)
from src.nlp.grls_resolver import DrugResolver, DrugSearchResult


class DrugResolution(ExtractionModel):
    """Результат поиска с сохранением всех вариантов названия."""

    search: DrugSearchResult
    status: Literal["resolved", "not_found", "ambiguous", "missing_inn"]
    inn: NonBlankText | None = None
    atx_codes: list[str] = Field(default_factory=list)


class ResolvedClinicalRelation(ExtractionModel):
    """Исходное утверждение и отдельно найденные сведения о лекарствах."""

    relation: ClinicalRelation
    subject_drug: DrugResolution
    object_drug: DrugResolution | None = None


def resolve_drug(name: str, resolver: DrugResolver) -> DrugResolution:
    search = resolver.search(name)
    if not search.matches:
        return DrugResolution(search=search, status="not_found")
    if search.ambiguous:
        return DrugResolution(search=search, status="ambiguous")
    if any(not match.inn or not match.inn.strip() for match in search.matches):
        return DrugResolution(search=search, status="missing_inn")

    inns = {match.inn.strip() for match in search.matches}
    if len(inns) != 1:
        return DrugResolution(search=search, status="ambiguous")

    codes = sorted({code for match in search.matches for code in match.atx_codes})
    return DrugResolution(
        search=search, status="resolved", inn=inns.pop(), atx_codes=codes,
    )


def resolve_relation_drugs(
    relation: ClinicalRelation,
    resolver: DrugResolver,
) -> ResolvedClinicalRelation:
    """Сопоставляет только лекарства, не создавая клинических связей из ГРЛС.

    Утверждение копируется: исходные названия, цитата, отрицание и условия
    не меняются. Заболевания не сопоставляются через лекарственный реестр.
    """
    relation = relation.model_copy(deep=True)
    subject = resolve_drug(relation.subject.name, resolver)
    target = None
    if relation.object.kind == "drug":
        target = resolve_drug(relation.object.name, resolver)

    # Сопоставление названий не снимает спорность и не отменяет отклонение.
    if relation.status not in ("disputed", "rejected"):
        if relation.polarity == "uncertain":
            relation.status = "disputed"
        elif subject.status != "resolved" or (
            target is not None and target.status != "resolved"
        ):
            relation.status = "unresolved"
        else:
            relation.status = "auto_checked"

    return ResolvedClinicalRelation(
        relation=relation, subject_drug=subject, object_drug=target,
    )
