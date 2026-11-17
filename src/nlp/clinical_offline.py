"""Автоматическая проверка сохранённых утверждений без LLM."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

import polars as pl

from src.nlp.clinical_relations import ClinicalFragment, ClinicalRelation, ExtractionModel
from src.nlp.clinical_resolver import DrugResolution, resolve_relation_drugs
from src.nlp.clinical_validation import EvidenceCheck, check_relation, normalize_text
from src.nlp.grls_resolver import DrugResolver

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data/processed"
REFERENCE_PATH = BASE_DIR / "tests/clinical_reference_28_3.json"


class ClinicalSample(ExtractionModel):
    fragment: ClinicalFragment
    relations: list[ClinicalRelation]


class PreparedClaim(EvidenceCheck):
    subject_drug: DrugResolution
    object_drug: DrugResolution | None = None
    page_sha256: str
    page_quote_start: int | None = None
    page_quote_end: int | None = None


class ReviewEntry(ExtractionModel):
    id: str
    claim: PreparedClaim


class ReviewReport(ExtractionModel):
    schema_version: Literal[1] = 1
    source_kind: Literal["reference_annotations", "saved_response"]
    manual_review: Literal["not_requested", "not_reviewed"] = "not_requested"
    entries: list[ReviewEntry]


def claim_id(claim: PreparedClaim) -> str:
    # Правка цитаты, условий, контекста или разрешения имени меняет идентификатор.
    content = json.dumps(claim.model_dump(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_samples(path: Path) -> tuple[str, list[ClinicalSample]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("examples"), list):
        if not data["examples"]:
            raise ValueError("Эталонная выборка пуста")
        return "reference_annotations", [
            ClinicalSample(fragment=item["fragment"], relations=[item["expected"]])
            for item in data["examples"]
        ]
    if isinstance(data, dict) and "fragment" in data and "relations" in data:
        # Формат результата clinical_trial. Сведения ГРЛС из файла не считаем
        # проверенными: они будут заново получены из текущего справочника.
        relations = [
            item["relation"] if "relation" in item else item
            for item in data["relations"]
        ]
        return "saved_response", [ClinicalSample(
            fragment=data["fragment"], relations=relations,
        )]
    raise ValueError("Нужна эталонная выборка или JSON результата clinical_trial")


def check_fragment_source(fragment: ClinicalFragment, pages: pl.DataFrame) -> str:
    required = {"code_version", "page_number", "text"}
    if not required.issubset(pages.columns):
        raise ValueError("В таблице страниц нужны code_version, page_number и text")
    rows = pages.filter(
        (pl.col("code_version") == fragment.document_id)
        & (pl.col("page_number") == fragment.page_number)
    )
    if rows.height != 1:
        raise ValueError("Исходная страница отсутствует или указана неоднозначно")
    text = rows["text"][0]
    if not isinstance(text, str) or fragment.text not in text:
        raise ValueError("Фрагмент не совпадает с исходной страницей")
    start = text.find(fragment.text)
    if text.find(fragment.text, start + 1) >= 0:
        raise ValueError("Фрагмент встречается на странице несколько раз; добавьте контекст")
    return text


def mark_conflicts(claims: list[PreparedClaim]) -> None:
    """Помечает явные противоречия только для одинаковых участников и условий."""
    groups = {}
    for claim in claims:
        relation = claim.relation
        if claim.issues or relation.status == "rejected":
            continue
        first = normalize_text(claim.subject_drug.inn or relation.subject.name)
        second = normalize_text(
            (claim.object_drug.inn if claim.object_drug else None) or relation.object.name
        )
        pair = (first, second)
        if relation.relation_type == "INTERACTS_WITH":
            pair = tuple(sorted(pair))
        conditions = tuple(sorted({normalize_text(text) for text in relation.conditions}))
        key = (relation.relation_type, pair, conditions)
        groups.setdefault(key, []).append(claim)

    for group in groups.values():
        polarities = {claim.relation.polarity for claim in group}
        if "affirmed" in polarities and "negated" in polarities:
            for claim in group:
                claim.relation.status = "disputed"
                claim.issues.append("contradictory_polarity")


def prepare_report(
    samples: list[ClinicalSample],
    resolver: DrugResolver,
    pages: pl.DataFrame,
    source_kind: str,
) -> ReviewReport:
    claims = []
    for sample in samples:
        page_text = check_fragment_source(sample.fragment, pages)
        page_offset = page_text.find(sample.fragment.text)
        for relation in sample.relations:
            check = check_relation(sample.fragment, relation)
            resolved = resolve_relation_drugs(check.relation, resolver)
            check.relation = resolved.relation
            if resolved.subject_drug.status != "resolved":
                check.issues.append("subject_" + resolved.subject_drug.status)
            if resolved.object_drug and resolved.object_drug.status != "resolved":
                check.issues.append("object_" + resolved.object_drug.status)
            claims.append(PreparedClaim(
                **check.model_dump(),
                subject_drug=resolved.subject_drug,
                object_drug=resolved.object_drug,
                page_sha256=hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
                page_quote_start=(page_offset + check.quote_start
                                  if check.quote_start is not None else None),
                page_quote_end=(page_offset + check.quote_end
                                if check.quote_end is not None else None),
            ))

    mark_conflicts(claims)
    # Повтор одного и того же утверждения не создаёт ещё одну запись проверки.
    entries = {claim_id(claim): ReviewEntry(id=claim_id(claim), claim=claim) for claim in claims}
    return ReviewReport(source_kind=source_kind, entries=list(entries.values()))


def write_new_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def review_draft(report: ReviewReport) -> dict:
    return {"reviews": [{
        "claim_id": entry.id, "decision": "pending", "reviewer": "",
        "reviewed_at": None, "pdf_checked": False, "participants_checked": False,
        "polarity_checked": False, "conditions_checked": False,
        "icd10_code": None, "notes": "",
    } for entry in report.entries]}


def compare_reference(reference_path: Path, report: ReviewReport) -> dict:
    """Сравнение выбранных утверждений; выборка не размечает все связи документа."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    comparisons = []
    for example in reference["examples"]:
        expected = ClinicalRelation.model_validate(example["expected"])
        candidates = []
        for entry in report.entries:
            actual = entry.claim.relation
            if (
                actual.source.document_id == expected.source.document_id
                and actual.source.page_number == expected.source.page_number
                and actual.relation_type == expected.relation_type
                and normalize_text(actual.subject.name) == normalize_text(expected.subject.name)
                and normalize_text(actual.object.name) == normalize_text(expected.object.name)
            ):
                candidates.append(entry.claim)
        matched = any(
            claim.relation.status not in ("disputed", "rejected")
            and claim.relation.polarity == expected.polarity
            and set(claim.relation.conditions) == set(expected.conditions)
            for claim in candidates
        )
        comparisons.append({
            "id": example["id"],
            "status": "matched" if matched else "different" if candidates else "missing",
        })
    return {
        "source_kind": report.source_kind,
        "partial_reference": True,
        "note": "Сравнение участников, отрицания и условий; прочие связи не оцениваются.",
        "examples": comparisons,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Проверить сохранённые утверждения")
    prepare.add_argument("--input", type=Path, default=REFERENCE_PATH)
    prepare.add_argument("--pages", type=Path, default=DATA_DIR / "cr_pages.parquet")
    prepare.add_argument("--dictionary", type=Path, default=DATA_DIR / "grls_normalized.parquet")
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--review-draft", action="store_true",
                         help="Дополнительно создать черновик необязательной ручной проверки")
    compare = commands.add_parser("compare", help="Сравнить выбранные утверждения с эталоном")
    compare.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    compare.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "compare":
        report = ReviewReport.model_validate_json(args.report.read_text(encoding="utf-8"))
        print(json.dumps(compare_reference(args.reference, report), ensure_ascii=False, indent=2))
        return 0

    kind, samples = load_samples(args.input)
    report = prepare_report(samples, DrugResolver(args.dictionary), pl.read_parquet(args.pages), kind)
    if args.review_draft:
        report.manual_review = "not_reviewed"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_dir or DATA_DIR / "clinical_trials" / f"offline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new_json(output_dir / "claims.json", report.model_dump())
    if args.review_draft:
        write_new_json(output_dir / "reviews.json", review_draft(report))
    counts = Counter(entry.claim.relation.status for entry in report.entries)
    print(f"Источник: {kind}; утверждений: {len(report.entries)}; статусы: {dict(counts)}")
    quotes_found = sum(entry.claim.quote_start is not None for entry in report.entries)
    print(f"Цитаты найдены однозначно: {quotes_found}/{len(report.entries)}")
    print(f"Результаты автоматической проверки: {output_dir}")
    if kind == "reference_annotations":
        print("Использована готовая разметка. Извлечение языковой моделью не выполнялось.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
