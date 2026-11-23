"""Батч: контрольные точки и ошибки без настоящих запросов LLM."""

from contextlib import contextmanager
import json

import polars as pl
import pytest

from src.nlp import clinical_batch as batch
from src.nlp.clinical_relations import ClinicalRelation, ExtractionResult


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    pages = tmp_path / "pages.parquet"
    pl.DataFrame({
        "code_version": ["200_1", "100_1", "100_1"],
        "page_number": [1, 2, 1],
        "text": ["   ", "Введение без ключевых слов.",
                 "  Препарат А не противопоказан при заболевании Б\nпри условии В.  "],
        "section_title": [None, None, "Учебный раздел"],
    }).write_parquet(pages)
    dictionary = tmp_path / "dictionary.parquet"
    pl.DataFrame({"trade_name_norm": ["Препарат А"], "inn_norm": ["вещество а"],
                  "atx_code": [["A01"]]}).write_parquet(dictionary)
    calls = []

    @contextmanager
    def client():
        yield object(), "test-model"

    def extract(fragment, **kwargs):
        calls.append(fragment)
        return ExtractionResult(relations=[])

    monkeypatch.setattr(batch, "extraction_client", client)
    monkeypatch.setattr(batch, "extract_relations", extract)
    return pages, dictionary, tmp_path / "result", calls


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_processes_all_pages_preserving_text_and_resumes_without_client(inputs, monkeypatch):
    path, dictionary, output, calls = inputs
    pages = batch.read_pages(path)
    assert batch.run_batch(pages, dictionary, output) == 0
    assert [fragment.page_number for fragment in calls] == [1, 2]
    assert calls[0].text == pages[0].text
    assert calls[0].location == "Страница 1, Учебный раздел"
    summary = load(output / "summary.json")
    assert (summary["completed_pages"], summary["empty_pages"], summary["error_pages"],
            summary["pending_pages"]) == (2, 1, 0, 0)
    record = load(output / "pages" / batch.page_filename(pages[0]))
    assert record["result"]["relations"] == []
    assert record["result"]["manual_review"] == "not_requested"

    def forbidden():
        pytest.fail("Готовый батч не должен открывать клиент")

    monkeypatch.setattr(batch, "extraction_client", forbidden)
    assert batch.run_batch(pages, dictionary, output, resume=True) == 0
    assert len(calls) == 2


def test_errors_are_not_empty_results_and_can_be_retried(inputs, monkeypatch):
    path, dictionary, output, calls = inputs
    pages = batch.read_pages(path)

    def extract(fragment, **kwargs):
        calls.append(fragment)
        if fragment.page_number == 1:
            raise RuntimeError("secret-api-key must not be saved")
        return ExtractionResult(relations=[])

    monkeypatch.setattr(batch, "extract_relations", extract)
    assert batch.run_batch(pages, dictionary, output) == 1
    record = load(output / "pages" / batch.page_filename(pages[0]))
    assert record["status"] == "error"
    assert "result" not in record
    assert "secret-api-key" not in json.dumps(record)
    assert load(output / "summary.json")["completed_pages"] == 1
    assert batch.run_batch(pages, dictionary, output, resume=True) == 1
    assert len(calls) == 2
    monkeypatch.setattr(batch, "extract_relations", lambda *a, **k: ExtractionResult(relations=[]))
    assert batch.run_batch(pages, dictionary, output, resume=True, retry_errors=True) == 0
    assert load(output / "summary.json")["error_pages"] == 0


def test_interrupt_preserves_checkpoint_and_continues(inputs, monkeypatch):
    path, dictionary, output, calls = inputs
    pages = batch.read_pages(path)

    def extract(fragment, **kwargs):
        calls.append(fragment)
        if fragment.page_number == 2:
            raise KeyboardInterrupt
        return ExtractionResult(relations=[])

    monkeypatch.setattr(batch, "extract_relations", extract)
    assert batch.run_batch(pages, dictionary, output) == 130
    summary = load(output / "summary.json")
    assert summary["interrupted"] is True
    assert summary["completed_pages"] == 1
    assert summary["pending_pages"] == 1
    calls.clear()

    def finish(fragment, **kwargs):
        calls.append(fragment)
        return ExtractionResult(relations=[])

    monkeypatch.setattr(batch, "extract_relations", finish)
    assert batch.run_batch(pages, dictionary, output, resume=True) == 0
    assert [item.page_number for item in calls] == [2]
    assert load(output / "summary.json")["interrupted"] is False


def test_resolves_drug_and_preserves_evidence(inputs, monkeypatch):
    path, dictionary, output, _ = inputs
    pages = batch.read_pages(path, ["100_1"], 1)

    def extract(fragment, **kwargs):
        return ExtractionResult(relations=[ClinicalRelation(
            subject={"name": "Препарат А", "kind": "drug"},
            object={"name": "заболевании Б", "kind": "disease"},
            relation_type="CONTRAINDICATED_FOR", polarity="negated",
            conditions=["при условии В"], description="Учебный пример",
            source={**fragment.model_dump(exclude={"text"}), "quote": fragment.text},
        )])

    monkeypatch.setattr(batch, "extract_relations", extract)
    assert batch.run_batch(pages, dictionary, output) == 0
    record = load(output / "pages" / batch.page_filename(pages[0]))
    item = record["result"]["relations"][0]
    assert item["subject_drug"]["inn"] == "вещество а"
    assert item["relation"]["source"]["quote"] == pages[0].text
    assert item["relation"]["conditions"] == ["при условии В"]
    assert item["relation"]["polarity"] == "negated"
    assert load(output / "summary.json")["relation_statuses"] == {"auto_checked": 1}


@pytest.mark.parametrize("change", ["text", "dictionary", "selection", "pipeline"])
def test_resume_rejects_changed_inputs(inputs, monkeypatch, change):
    path, dictionary, output, _ = inputs
    pages = batch.read_pages(path)
    batch.run_batch(pages, dictionary, output)
    if change == "text":
        pages[0] = pages[0].model_copy(update={"text": "Изменено"})
    elif change == "dictionary":
        pl.DataFrame({"other": [1]}).write_parquet(dictionary)
    elif change == "selection":
        pages = pages[:1]
    else:
        manifest = load(output / "manifest.json")
        manifest["input"]["pipeline_sha256"] = {}
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="изменились"):
        batch.run_batch(pages, dictionary, output, resume=True)


def test_model_change_is_rejected_before_request(inputs, monkeypatch):
    path, dictionary, output, calls = inputs
    pages = batch.read_pages(path)
    batch.run_batch(pages, dictionary, output)
    (output / "pages" / batch.page_filename(pages[0])).unlink()

    @contextmanager
    def changed_client():
        yield object(), "different-model"

    monkeypatch.setattr(batch, "extraction_client", changed_client)
    with pytest.raises(ValueError, match="Модель изменилась"):
        batch.run_batch(pages, dictionary, output, resume=True)
    assert len(calls) == 2


def test_reasoning_change_requires_new_batch(inputs, monkeypatch):
    path, dictionary, output, calls = inputs
    pages = batch.read_pages(path)
    batch.run_batch(pages, dictionary, output)
    (output / "pages" / batch.page_filename(pages[0])).unlink()
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    with pytest.raises(ValueError, match="Режим reasoning изменился"):
        batch.run_batch(pages, dictionary, output, resume=True)
    assert len(calls) == 2


def test_existing_output_and_corrupt_checkpoint_are_rejected(inputs):
    path, dictionary, output, _ = inputs
    pages = batch.read_pages(path)
    batch.run_batch(pages, dictionary, output)
    with pytest.raises(FileExistsError):
        batch.run_batch(pages, dictionary, output)
    (output / "pages" / batch.page_filename(pages[0])).write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        batch.run_batch(pages, dictionary, output, resume=True)


def test_summary_is_rebuilt_from_saved_pages(inputs):
    path, dictionary, output, calls = inputs
    pages = batch.read_pages(path)
    batch.run_batch(pages, dictionary, output)
    (output / "summary.json").write_text("{", encoding="utf-8")
    # Остаток недописанного файла не подменяет готовую контрольную точку.
    (output / "pages" / (batch.page_filename(pages[0]) + ".tmp")).write_text("{", encoding="utf-8")
    assert batch.run_batch(pages, dictionary, output, resume=True) == 0
    assert len(calls) == 2
    assert load(output / "summary.json")["completed_pages"] == 2


@pytest.mark.parametrize("row", [
    {"code_version": "a", "page_number": 0, "text": "x"},
    {"code_version": "a", "page_number": 1.5, "text": "x"},
    {"code_version": None, "page_number": 1, "text": "x"},
    {"code_version": "a", "page_number": 1, "text": 12},
    {"text": "x"},
])
def test_rejects_invalid_source_metadata(tmp_path, row):
    path = tmp_path / "invalid.parquet"
    pl.DataFrame([row]).write_parquet(path)
    with pytest.raises(ValueError):
        batch.read_pages(path)


def test_dry_run_needs_no_dictionary_or_client_and_writes_nothing(inputs, monkeypatch, capsys):
    path, _, output, calls = inputs
    assert batch.main(["--pages", str(path), "--dictionary", "missing.parquet",
                       "--output-dir", str(output), "--dry-run"]) == 0
    assert load_output(capsys)["total_pages"] == 3
    assert not calls
    assert not output.exists()


def load_output(capsys):
    return json.loads(capsys.readouterr().out)


def test_null_and_blank_pages_never_open_client(inputs, monkeypatch):
    _, dictionary, output, _ = inputs
    pages = [batch.PreparedPage(code_version="a", page_number=1, text=None)]
    monkeypatch.setattr(batch, "extraction_client", lambda: pytest.fail("LLM вызвана"))
    assert batch.run_batch(pages, dictionary, output) == 0
    assert load(output / "summary.json")["empty_pages"] == 1


def test_selection_validation_and_safe_paths(inputs):
    path, _, _, _ = inputs
    assert len(batch.read_pages(path, ["100_1"], 1)) == 1
    with pytest.raises(ValueError, match="не найдены"):
        batch.read_pages(path, ["missing"])
    table = pl.read_parquet(path)
    pl.concat([table, table]).write_parquet(path)
    with pytest.raises(ValueError, match="несколько записей"):
        batch.read_pages(path)
    page = batch.PreparedPage(code_version="../../outside", page_number=1, text="x")
    assert "/" not in batch.page_filename(page)


@pytest.mark.parametrize("args", [["--limit", "0"], ["--resume"], ["--retry-errors"]])
def test_invalid_arguments_fail_before_reading_files(args):
    with pytest.raises(SystemExit) as error:
        batch.main(args)
    assert error.value.code == 2


def test_write_failure_is_not_recorded_as_model_error(inputs, monkeypatch):
    path, dictionary, output, calls = inputs
    real_write = batch.write_json

    def write(path, value):
        if value.get("status") == "completed":
            raise OSError("disk full")
        real_write(path, value)

    monkeypatch.setattr(batch, "write_json", write)
    with pytest.raises(OSError, match="disk full"):
        batch.run_batch(batch.read_pages(path), dictionary, output)
    assert len(calls) == 1
    assert load(output / "summary.json")["error_pages"] == 0
    assert load(output / "summary.json")["pending_pages"] == 2
