"""Оценка извлечения LLM на выбранных утверждениях из эталонной разметки."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter

from src.nlp.clinical_extractor import extract_relations
from src.nlp.clinical_llm import extraction_client
from src.nlp.clinical_relations import ClinicalFragment, ClinicalRelation
from src.nlp.clinical_validation import name_in_quote

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = BASE_DIR / "tests" / "clinical_reference_28_3.json"


def normalize(text: str) -> str:
    return " ".join(text.split())


def compare_statement(expected, relations, fragment) -> dict:
    """Сравнивает одну размеченную связь; остальные связи не штрафуются.

    Названия сравниваются без учёта регистра и пробелов, без синонимов.
    Условия сравниваются целиком, без учёта порядка и пробелов.
    Цитата может быть другой длины, но обязана находиться в исходном тексте.
    """
    candidates = [
        relation for relation in relations
        if relation.relation_type == expected.relation_type
        and relation.subject.kind == expected.subject.kind
        and relation.object.kind == expected.object.kind
        and normalize(relation.subject.name).casefold() == normalize(expected.subject.name).casefold()
        and normalize(relation.object.name).casefold() == normalize(expected.object.name).casefold()
    ]

    def checks(relation):
        return {
            "relation_found": True,
            "polarity_correct": relation.polarity == expected.polarity,
            "conditions_correct": (
                {normalize(item) for item in relation.conditions}
                == {normalize(item) for item in expected.conditions}
            ),
            "quote_grounded": (
                relation.source.quote in fragment.text
                and name_in_quote(relation.subject.name, relation.source.quote)
                and name_in_quote(relation.object.name, relation.source.quote)
                and all(normalize(item) in normalize(relation.source.quote)
                        for item in relation.conditions)
            ),
        }

    if not candidates:
        return {
            "relation_found": False,
            "polarity_correct": False,
            "conditions_correct": False,
            "quote_grounded": False,
        }
    # Все проверки относятся к одной связи, даже если модель выдала дубликаты.
    return max((checks(item) for item in candidates), key=lambda item: sum(item.values()))


def evaluate_reference(reference: dict, *, client, model: str) -> dict:
    examples = reference["examples"]
    if not examples:
        raise ValueError("Эталонная выборка пуста")
    cache = {}
    cases = []
    for example in examples:
        fragment = ClinicalFragment.model_validate(example["fragment"])
        expected = ClinicalRelation.model_validate(example["expected"])
        key = fragment.model_dump_json()
        if key not in cache:
            print(
                f"Фрагмент {len(cache) + 1}: {fragment.document_id}, "
                f"страница {fragment.page_number}, {len(fragment.text)} символов; "
                "ожидаю ответ модели...",
                flush=True,
            )
            started = perf_counter()
            try:
                cache[key] = extract_relations(fragment, client=client, model=model)
            except Exception as error:
                print(
                    f"Запрос завершился ошибкой {type(error).__name__} "
                    f"через {perf_counter() - started:.1f} с.",
                    flush=True,
                )
                raise
            print(
                f"Ответ получен за {perf_counter() - started:.1f} с; "
                f"утверждений: {len(cache[key].relations)}.",
                flush=True,
            )
        actual = cache[key]
        checks = compare_statement(expected, actual.relations, fragment)
        cases.append({
            "id": example["id"],
            "checks": checks,
            "passed": all(checks.values()),
            "fragment": fragment.model_dump(),
            "expected": expected.model_dump(),
            "actual": actual.model_dump(),
        })

    total = len(cases)
    metrics = {
        name: sum(case["checks"][field] for case in cases) / total
        for name, field in {
            "relation_recall": "relation_found",
            "polarity_accuracy": "polarity_correct",
            "conditions_accuracy": "conditions_correct",
            "quote_grounding_accuracy": "quote_grounded",
        }.items()
    }
    metrics["strict_accuracy"] = sum(case["passed"] for case in cases) / total
    return {
        "model": model,
        "human_review": reference.get("human_review", "unknown"),
        "pdf_visual_review": reference.get("pdf_visual_review", "unknown"),
        "scope": "Выбранные утверждения; лишние связи не оцениваются, precision и F1 не вычисляются",
        "total": total,
        "metrics": metrics,
        "cases": cases,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-accuracy", type=float, help="Минимальная strict_accuracy от 0 до 1")
    args = parser.parse_args(argv)
    if args.min_accuracy is not None and not 0 <= args.min_accuracy <= 1:
        parser.error("--min-accuracy должна быть от 0 до 1")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or BASE_DIR / "data/processed/clinical_trials" / f"evaluation_{timestamp}.json"
    if output.exists():
        raise FileExistsError("Файл результата уже существует; укажите другой --output")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    print("Сравниваем ответы модели с эталонной выборкой...", flush=True)
    with extraction_client() as (client, model):
        print(f"Модель: {model}", flush=True)
        reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip() or None
        print(f"Режим reasoning: {reasoning_effort or 'настройка сервера'}", flush=True)
        report = evaluate_reference(reference, client=client, model=model)
        report["reasoning_effort"] = reasoning_effort
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Модель: {model}; примеров: {report['total']}")
    for name, value in report["metrics"].items():
        print(f"{name}: {value:.1%}")
    print(f"Проверка разметки человеком: {report['human_review']}")
    print(f"Сохранено: {output}")
    if args.min_accuracy is not None and report["metrics"]["strict_accuracy"] < args.min_accuracy:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
