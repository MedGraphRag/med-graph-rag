"""Пакетный запуск и продолжение после ошибок, без вызовов API."""

import argparse
import json

import polars as pl
import pytest

from src.nlp import clinical_batch


@pytest.fixture
def batch(tmp_path, monkeypatch):
    pages = tmp_path / "pages.parquet"
    pl.DataFrame({
        "code_version": ["200_1", "100_1", "100_1", "200_1"],
        "page_number": [2, 1, 2, 1],
        "text": ["Текст второго документа.", "Введение без ключевых слов.", " ", None],
    }).write_parquet(pages)
    output = tmp_path / "results"
    calls = []
    monkeypatch.setattr(clinical_batch, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(clinical_batch.time, "sleep", lambda seconds: None)
    for name, value in {
        "LLM_API_KEY": "test-key", "LLM_BASE_URL": "https://example.test/v1",
        "LLM_MODEL": "test-model",
    }.items():
        monkeypatch.setenv(name, value)

    def run_page(argv):
        parser = argparse.ArgumentParser()
        for name in ("pages", "dictionary", "document", "page", "output"):
            parser.add_argument("--" + name)
        args = parser.parse_args(argv)
        calls.append((args.document, int(args.page)))
        row = pl.read_parquet(pages).filter(
            (pl.col("code_version") == args.document)
            & (pl.col("page_number") == int(args.page))
        ).row(0, named=True)
        output.mkdir(exist_ok=True)
        (output / f"{args.document}_page_{args.page}.json").write_text(json.dumps({
            "model": "test-model", "fragment": {
                "document_id": args.document, "page_number": int(args.page),
                "location": f"Страница {args.page}", "text": row["text"],
            }, "relations": [],
        }), encoding="utf-8")
        return 0

    monkeypatch.setattr(clinical_batch, "run_page", run_page)
    return ["--pages", str(pages), "--output-dir", str(output)], calls, output


def test_processes_all_documents_and_skips_empty_pages(batch):
    arguments, calls, output = batch
    assert clinical_batch.main(arguments) == 0
    assert calls == [("100_1", 1), ("200_1", 2)]
    assert len(list(output.glob("*.json"))) == 2


def test_dry_run_needs_no_model_and_makes_no_requests(batch, monkeypatch):
    arguments, calls, output = batch
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert clinical_batch.main(arguments + ["--dry-run"]) == 0
    assert calls == []
    assert not output.exists()


def test_limit_and_resume_skip_successful_pages(batch):
    arguments, calls, output = batch
    assert clinical_batch.main(arguments + ["--limit", "1"]) == 0
    first = output / "100_1_page_1.json"
    original = first.read_bytes()
    assert calls == [("100_1", 1)]

    assert clinical_batch.main(arguments + ["--limit", "1"]) == 0
    assert calls == [("100_1", 1), ("200_1", 2)]
    assert first.read_bytes() == original


def test_api_error_stops_batch_and_can_be_resumed(batch, monkeypatch):
    arguments, calls, output = batch
    normal_run = clinical_batch.run_page

    def failing_run(argv):
        if argv[argv.index("--document") + 1] == "200_1":
            raise RuntimeError("429: upstream_provider_shared_pool")
        return normal_run(argv)

    monkeypatch.setattr(clinical_batch, "run_page", failing_run)
    assert clinical_batch.main(arguments) == 1
    assert (output / "100_1_page_1.json").exists()
    assert not (output / "200_1_page_2.json").exists()

    monkeypatch.setattr(clinical_batch, "run_page", normal_run)
    assert clinical_batch.main(arguments) == 0
    assert calls == [("100_1", 1), ("200_1", 2)]


@pytest.mark.parametrize("change", ["broken_json", "changed_text", "changed_model"])
def test_invalid_saved_result_is_not_silently_skipped(batch, change):
    arguments, calls, output = batch
    assert clinical_batch.main(arguments + ["--limit", "1"]) == 0
    saved_path = output / "100_1_page_1.json"
    if change == "broken_json":
        saved_path.write_text("{", encoding="utf-8")
    else:
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if change == "changed_text":
            saved["fragment"]["text"] = "Другой текст"
        else:
            saved["model"] = "another-model"
        saved_path.write_text(json.dumps(saved), encoding="utf-8")
    original = saved_path.read_bytes()

    assert clinical_batch.main(arguments) == 1
    assert calls == [("100_1", 1)]
    assert saved_path.read_bytes() == original


def test_duplicate_source_pages_block_requests(batch):
    arguments, calls, _ = batch
    pages_path = arguments[1]
    pages = pl.read_parquet(pages_path)
    pl.concat([pages, pages.head(1)]).write_parquet(pages_path)
    with pytest.raises(SystemExit) as error:
        clinical_batch.main(arguments)
    assert error.value.code == 2
    assert calls == []


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_rejects_nonpositive_limit(batch, limit):
    arguments, calls, _ = batch
    with pytest.raises(SystemExit) as error:
        clinical_batch.main(arguments + ["--limit", limit])
    assert error.value.code == 2
    assert calls == []
