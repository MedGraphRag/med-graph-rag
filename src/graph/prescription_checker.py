from collections.abc import Iterable
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, StringConstraints, computed_field

from src.nlp.grls_resolver import DrugSearchResult

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonBlankText = Annotated[str, StringConstraints(pattern=r"\S")]


class DrugNameResolver(Protocol):
    """Общий интерфейс существующих поисков лекарств в Parquet и Neo4j."""

    def search(self, query: str) -> DrugSearchResult: ...


class GuidelineSource(BaseModel):
    document_id: Identifier
    location: NonBlankText
    quote: NonBlankText


class ConfirmedContraindication(BaseModel):
    drug_inn: Identifier
    icd10_code: Identifier
    source: GuidelineSource
    conditions: tuple[NonBlankText, ...] = ()


class ConfirmedInteraction(BaseModel):
    """Хранит участников и описание взаимодействия в порядке из источника."""

    first_drug_inn: Identifier
    second_drug_inn: Identifier
    description: NonBlankText
    source: GuidelineSource
    conditions: tuple[NonBlankText, ...] = ()


class PrescriptionCheckResult(BaseModel):
    matches: tuple[ConfirmedContraindication | ConfirmedInteraction, ...] = ()

    @computed_field
    @property
    def status(self) -> Literal["confirmed_match", "insufficient_data"]:
        return "confirmed_match" if self.matches else "insufficient_data"


class DrugDiseaseNameCheckResult(BaseModel):
    """Сохраняет варианты поиска названия и найденные клинические связи."""

    drug_search: DrugSearchResult
    matches: tuple[ConfirmedContraindication, ...] = ()

    @computed_field
    @property
    def status(self) -> Literal["confirmed_match", "insufficient_data", "ambiguous"]:
        if self.drug_search.ambiguous:
            return "ambiguous"
        return "confirmed_match" if self.matches else "insufficient_data"


class DrugDrugNameCheckResult(BaseModel):
    """Сохраняет результаты поиска обоих препаратов и их взаимодействия."""

    first_drug_search: DrugSearchResult
    second_drug_search: DrugSearchResult
    matches: tuple[ConfirmedInteraction, ...] = ()

    @computed_field
    @property
    def status(self) -> Literal["confirmed_match", "insufficient_data", "ambiguous"]:
        if self.first_drug_search.ambiguous or self.second_drug_search.ambiguous:
            return "ambiguous"
        return "confirmed_match" if self.matches else "insufficient_data"


def check_drug_disease(
    drug_inn: str | None,
    icd10_code: str | None,
    confirmed_relations: Iterable[ConfirmedContraindication],
) -> PrescriptionCheckResult:
    """Ищет подтверждённые связи по известному МНН и точному коду МКБ-10.

    Совпадение означает, что связь есть в источнике. Возвращённые условия
    применимости нужно отдельно проверить для конкретного случая.
    """
    if drug_inn is None or icd10_code is None:
        return PrescriptionCheckResult()

    inn = drug_inn.strip()
    code = icd10_code.strip()
    matches = [
        relation
        for relation in confirmed_relations
        if relation.drug_inn == inn and relation.icd10_code == code
    ]
    # Порядок связей во входных данных не должен менять результат проверки.
    matches.sort(key=lambda relation: relation.model_dump_json())
    return PrescriptionCheckResult(matches=tuple(matches))


def check_drug_drug(
    first_drug_inn: str | None,
    second_drug_inn: str | None,
    confirmed_relations: Iterable[ConfirmedInteraction],
) -> PrescriptionCheckResult:
    """Ищет подтверждённые взаимодействия по двум уже определённым МНН.

    Порядок ввода лекарств не влияет на поиск. Порядок участников,
    описание, условия и источник найденной связи сохраняются.
    Условия применимости нужно отдельно проверить для конкретного случая.
    """
    if first_drug_inn is None or second_drug_inn is None:
        return PrescriptionCheckResult()

    first_inn = first_drug_inn.strip()
    second_inn = second_drug_inn.strip()
    pairs = {(first_inn, second_inn), (second_inn, first_inn)}
    matches = [
        relation
        for relation in confirmed_relations
        if (relation.first_drug_inn, relation.second_drug_inn) in pairs
    ]
    # Учитываем все поля связи, включая источник, цитату и условия.
    matches.sort(key=lambda relation: relation.model_dump_json())
    return PrescriptionCheckResult(matches=tuple(matches))


def check_drug_disease_by_name(
    drug_name: str,
    icd10_code: str | None,
    resolver: DrugNameResolver,
    confirmed_relations: Iterable[ConfirmedContraindication],
) -> DrugDiseaseNameCheckResult:
    """Проверяет пару «название лекарства — код МКБ-10» через справочник.

    При неоднозначном названии возвращает варианты для уточнения.
    Для однозначного результата использует МНН из справочника.
    """
    drug_search = resolver.search(drug_name)
    if drug_search.ambiguous or not drug_search.matches:
        return DrugDiseaseNameCheckResult(drug_search=drug_search)

    # Несколько записей одного варианта препарата имеют одинаковое МНН.
    result = check_drug_disease(
        drug_search.matches[0].inn, icd10_code, confirmed_relations
    )
    return DrugDiseaseNameCheckResult(
        drug_search=drug_search, matches=result.matches
    )


def check_drug_drug_by_name(
    first_drug_name: str,
    second_drug_name: str,
    resolver: DrugNameResolver,
    confirmed_relations: Iterable[ConfirmedInteraction],
) -> DrugDrugNameCheckResult:
    """Проверяет взаимодействия двух лекарств по названиям через справочник.

    Если хотя бы одно название неоднозначно, возвращает варианты для уточнения.
    Найденные связи сохраняют описание, условия и порядок участников источника.
    """
    first_search = resolver.search(first_drug_name)
    second_search = resolver.search(second_drug_name)
    if (
        first_search.ambiguous
        or second_search.ambiguous
        or not first_search.matches
        or not second_search.matches
    ):
        return DrugDrugNameCheckResult(
            first_drug_search=first_search, second_drug_search=second_search
        )

    result = check_drug_drug(
        first_search.matches[0].inn,
        second_search.matches[0].inn,
        confirmed_relations,
    )
    return DrugDrugNameCheckResult(
        first_drug_search=first_search,
        second_drug_search=second_search,
        matches=result.matches,
    )
