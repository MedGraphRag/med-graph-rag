"""Модели утверждений, извлечённых из клинических рекомендаций."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonBlankText = Annotated[str, StringConstraints(pattern=r"\S")]


class ExtractionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClinicalFragment(ExtractionModel):
    """Подготовленный текст одного фрагмента на одной странице КР."""

    source_type: Literal["clinical_guideline"] = "clinical_guideline"
    document_id: NonBlankText
    location: NonBlankText
    page_number: int = Field(ge=1, strict=True)
    text: NonBlankText


class RelationParticipant(ExtractionModel):
    """Название из текста. Сопоставление со справочником выполняется отдельно."""

    name: NonBlankText
    kind: Literal["drug", "disease"]


class RelationSource(ExtractionModel):
    """Источник утверждения. ГРЛС здесь использовать нельзя."""

    source_type: Literal["clinical_guideline"] = "clinical_guideline"
    document_id: NonBlankText
    location: NonBlankText
    page_number: int = Field(ge=1, strict=True)
    quote: NonBlankText


class ClinicalRelation(ExtractionModel):
    """Извлечённое утверждение, которое ещё нельзя публиковать в графе.

    affirmed — связь утверждается, negated — отрицается,
    uncertain — сформулирована с сомнением. Условия сохраняются отдельно.
    Статус confirmed здесь намеренно недоступен.
    """

    subject: RelationParticipant
    object: RelationParticipant
    relation_type: Literal["CONTRAINDICATED_FOR", "INTERACTS_WITH"]
    polarity: Literal["affirmed", "negated", "uncertain"]
    conditions: list[NonBlankText]
    description: NonBlankText
    source: RelationSource
    # auto_checked означает прохождение автоматических проверок, а не
    # медицинское подтверждение. pending_review читается для старых файлов.
    status: Literal["unresolved", "auto_checked", "disputed", "rejected", "pending_review"] = "unresolved"

    @model_validator(mode="after")
    def check_participants(self):
        if self.subject.kind != "drug":
            raise ValueError("Первый участник связи должен быть лекарством")

        if self.relation_type == "CONTRAINDICATED_FOR":
            if self.object.kind != "disease":
                raise ValueError("Противопоказание должно относиться к заболеванию")
        elif self.object.kind != "drug":
            raise ValueError("Взаимодействие должно связывать два лекарства")

        return self


class ExtractionResult(ExtractionModel):
    # Пустой список означает, что в переданном фрагменте связи не найдены.
    relations: list[ClinicalRelation]
