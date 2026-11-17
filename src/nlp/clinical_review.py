"""Необязательный ручной допуск выбранных связей, без записи в Neo4j."""

import argparse
import json
from pathlib import Path
import re
from typing import Literal

import polars as pl
from pydantic import AwareDatetime, Field

from src.graph.prescription_checker import (
    ConfirmedContraindication,
    ConfirmedInteraction,
    GuidelineSource,
)
from src.nlp.clinical_offline import (
    ClinicalSample,
    DATA_DIR,
    ReviewReport,
    claim_id,
    prepare_report,
    write_new_json,
)
from src.nlp.clinical_relations import ExtractionModel, NonBlankText
from src.nlp.grls_resolver import DrugResolver


class ReviewDecision(ExtractionModel):
    claim_id: NonBlankText
    decision: Literal["pending", "confirmed", "disputed", "rejected"] = "pending"
    reviewer: str = ""
    reviewed_at: AwareDatetime | None = None
    pdf_checked: bool = Field(default=False, strict=True)
    participants_checked: bool = Field(default=False, strict=True)
    polarity_checked: bool = Field(default=False, strict=True)
    conditions_checked: bool = Field(default=False, strict=True)
    icd10_code: str | None = None
    notes: str = ""


class ReviewDecisions(ExtractionModel):
    reviews: list[ReviewDecision]


def load_icd10_codes(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        node["code"] for node in data["nodes"]
        if re.fullmatch(r"[A-Z]\d{2}(?:\.\d{1,2})?", node["code"])
    }


def export_reviewed(
    report: ReviewReport,
    decisions: ReviewDecisions,
    resolver: DrugResolver,
    pages: pl.DataFrame,
    icd10_codes: set[str],
) -> dict:
    entries = {entry.id: entry for entry in report.entries}
    if len(entries) != len(report.entries):
        raise ValueError("В отчёте повторяются идентификаторы утверждений")
    if any(entry.id != claim_id(entry.claim) for entry in report.entries):
        raise ValueError("Отчёт изменён; заново выполните prepare и ручную проверку")

    reviews = {review.claim_id: review for review in decisions.reviews}
    if len(reviews) != len(decisions.reviews):
        raise ValueError("Для одного утверждения указано несколько решений")
    if set(reviews) - set(entries):
        raise ValueError("Решение относится к неизвестному или устаревшему утверждению")

    # Проверяем источник и справочник заново, не доверяя сохранённым флагам.
    samples = [ClinicalSample(
        fragment=entry.claim.fragment, relations=[entry.claim.relation],
    ) for entry in report.entries]
    fresh = prepare_report(samples, resolver, pages, report.source_kind)
    fresh_entries = {entry.id: entry for entry in fresh.entries}
    result = {
        "source_kind": report.source_kind,
        "contraindications": [], "interactions": [], "deferred": [],
    }
    for identifier, entry in entries.items():
        review = reviews.get(identifier)
        if review is None or review.decision != "confirmed":
            result["deferred"].append({
                "claim_id": identifier,
                "status": review.decision if review else "pending",
                "relation_status": entry.claim.relation.status,
                "issues": entry.claim.issues,
            })
            continue

        if not review.reviewer.strip() or review.reviewed_at is None:
            raise ValueError("Для подтверждения нужны проверяющий и дата с часовым поясом")
        if not all((review.pdf_checked, review.participants_checked,
                    review.polarity_checked, review.conditions_checked)):
            raise ValueError("Проверка PDF, участников, отрицания и условий не завершена")
        if identifier not in fresh_entries:
            raise ValueError("Источник или результат проверки изменился; нужна новая проверка")
        claim = fresh_entries[identifier].claim
        relation = claim.relation
        if claim.issues or relation.status != "auto_checked":
            raise ValueError("Спорное, отклонённое или неразрешённое утверждение нельзя подтвердить")
        if relation.polarity != "affirmed":
            raise ValueError("Отрицаемая связь не публикуется как существующая")

        source = GuidelineSource.model_validate(relation.source.model_dump(
            include={"document_id", "location", "quote"},
        ))
        if relation.relation_type == "CONTRAINDICATED_FOR":
            code = (review.icd10_code or "").strip()
            if code not in icd10_codes:
                raise ValueError("Укажите проверенный код заболевания из справочника МКБ-10")
            confirmed = ConfirmedContraindication(
                drug_inn=claim.subject_drug.inn, icd10_code=code,
                source=source, conditions=tuple(relation.conditions),
            )
            target = result["contraindications"]
        else:
            if claim.object_drug is None or claim.object_drug.status != "resolved":
                raise ValueError("Не определено второе лекарство")
            confirmed = ConfirmedInteraction(
                first_drug_inn=claim.subject_drug.inn,
                second_drug_inn=claim.object_drug.inn,
                description=relation.description,
                source=source, conditions=tuple(relation.conditions),
            )
            target = result["interactions"]
        target.append({
            **confirmed.model_dump(mode="json"), "status": "confirmed",
            "claim_id": identifier, "review": review.model_dump(mode="json"),
        })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--pages", type=Path, default=DATA_DIR / "cr_pages.parquet")
    parser.add_argument("--dictionary", type=Path, default=DATA_DIR / "grls_normalized.parquet")
    parser.add_argument("--icd10", type=Path, default=DATA_DIR / "icd10.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Файл экспорта уже существует; укажите новый --output")
    report = ReviewReport.model_validate_json(args.report.read_text(encoding="utf-8"))
    reviews = ReviewDecisions.model_validate_json(args.reviews.read_text(encoding="utf-8"))
    codes = load_icd10_codes(args.icd10) if args.icd10.exists() else set()
    result = export_reviewed(
        report, reviews, DrugResolver(args.dictionary), pl.read_parquet(args.pages), codes,
    )
    write_new_json(args.output, result)
    print(f"Противопоказаний: {len(result['contraindications'])}")
    print(f"Взаимодействий: {len(result['interactions'])}")
    print(f"Отложено: {len(result['deferred'])}; результат: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
