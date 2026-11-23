"""Проверки расчёта качества; реальная модель в этих тестах не вызывается."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.nlp import clinical_evaluation as evaluation
from src.nlp.clinical_relations import ClinicalFragment, ClinicalRelation, ExtractionResult

REFERENCE = json.loads(
    Path(__file__).with_name("clinical_reference_28_3.json").read_text(encoding="utf-8")
)


def test_reference_evaluation_preserves_all_annotations_and_reuses_fragments():
    client = Mock()

    def answer(**kwargs):
        fragment = json.loads(kwargs["messages"][-1]["content"])
        relations = [
            ClinicalRelation.model_validate(example["expected"])
            for example in REFERENCE["examples"]
            if ClinicalFragment.model_validate(example["fragment"]).model_dump() == fragment
        ]
        return ExtractionResult(relations=relations)

    client.chat.completions.create.side_effect = answer
    report = evaluation.evaluate_reference(REFERENCE, client=client, model="gemma4:12b")

    assert report["total"] == 5
    assert all(value == 1 for value in report["metrics"].values())
    assert report["human_review"] == "pending"
    assert client.chat.completions.create.call_count == len({
        ClinicalFragment.model_validate(item["fragment"]).model_dump_json()
        for item in REFERENCE["examples"]
    })


@pytest.mark.parametrize("change,failed_check", [
    ("missing", "relation_found"),
    ("participant", "relation_found"),
    ("polarity", "polarity_correct"),
    ("condition_missing", "conditions_correct"),
    ("condition_extra", "conditions_correct"),
    ("quote", "quote_grounded"),
    ("irrelevant_quote", "quote_grounded"),
])
def test_detects_extraction_errors(change, failed_check):
    example = REFERENCE["examples"][0]
    fragment = ClinicalFragment.model_validate(example["fragment"])
    expected = ClinicalRelation.model_validate(example["expected"])
    actual = expected.model_copy(deep=True)
    if change == "participant":
        actual.object.name = "Другое заболевание"
    elif change == "polarity":
        actual.polarity = "negated"
    elif change == "condition_missing":
        actual.conditions = []
    elif change == "condition_extra":
        actual.conditions.append("Дополнительное условие")
    elif change == "quote":
        actual.source.quote = "Выдуманная цитата"
    elif change == "irrelevant_quote":
        actual.source.quote = fragment.text[:20]
    relations = [] if change == "missing" else [actual]

    checks = evaluation.compare_statement(expected, relations, fragment)

    assert checks[failed_check] is False
    assert not all(checks.values())


def test_does_not_combine_correct_fields_from_different_predictions():
    example = REFERENCE["examples"][0]
    fragment = ClinicalFragment.model_validate(example["fragment"])
    expected = ClinicalRelation.model_validate(example["expected"])
    wrong_polarity = expected.model_copy(deep=True, update={"polarity": "negated"})
    wrong_conditions = expected.model_copy(deep=True, update={"conditions": []})

    checks = evaluation.compare_statement(expected, [wrong_polarity, wrong_conditions], fragment)

    assert not all(checks.values())


def test_empty_response_scores_zero():
    client = Mock()
    client.chat.completions.create.return_value = ExtractionResult(relations=[])

    report = evaluation.evaluate_reference(REFERENCE, client=client, model="test")

    assert all(value == 0 for value in report["metrics"].values())


def test_api_errors_are_not_scored_as_missing_relations():
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("API unavailable")

    with pytest.raises(RuntimeError, match="API unavailable"):
        evaluation.evaluate_reference(REFERENCE, client=client, model="test")


def test_accuracy_threshold_saves_report_and_returns_failure(tmp_path, monkeypatch):
    client = Mock()
    client.chat.completions.create.return_value = ExtractionResult(relations=[])

    @contextmanager
    def fake_client():
        yield client, "gemma4:12b"

    monkeypatch.setattr(evaluation, "extraction_client", fake_client)
    output = tmp_path / "evaluation.json"

    assert evaluation.main(["--output", str(output), "--min-accuracy", "0.8"]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["metrics"]["strict_accuracy"] == 0
    with pytest.raises(FileExistsError):
        evaluation.main(["--output", str(output)])


def test_empty_reference_is_rejected():
    reference = deepcopy(REFERENCE)
    reference["examples"] = []
    with pytest.raises(ValueError, match="пуста"):
        evaluation.evaluate_reference(reference, client=Mock(), model="test")
