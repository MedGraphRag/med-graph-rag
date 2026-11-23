"""Проверки выбора страницы для пробного запуска, без запросов к LLM."""

import polars as pl
import pytest

from src.nlp.clinical_trial import main, read_fragment


@pytest.fixture
def pages_path(tmp_path):
    path = tmp_path / "pages.parquet"
    pl.DataFrame({
        "document_id": [20, 10, 10, 10],
        "code_version": ["200_1", "100_1", "100_1", "100_1"],
        "page_number": [2, 1, 4, 6],
        "section_title": [None, None, "Учебный раздел", "Другой раздел"],
        "text": [
            "Вымышленное взаимодействие.",
            "Введение.",
            "  Вымышленные ПРОТИВОПОКАЗАНИЯ.\nУсловие.  ",
            "Другое вымышленное взаимодействие.",
        ],
    }).write_parquet(path)
    return path


def test_selects_one_candidate_and_preserves_exact_text(pages_path):
    fragment = read_fragment(pages_path)

    assert fragment.document_id == "100_1"
    assert fragment.page_number == 4
    assert fragment.text == "  Вымышленные ПРОТИВОПОКАЗАНИЯ.\nУсловие.  "
    assert fragment.location == "Страница 4, Учебный раздел"


def test_selects_document(pages_path):
    fragment = read_fragment(pages_path, document="200_1")

    assert fragment.document_id == "200_1"
    assert fragment.page_number == 2


def test_explicit_page_does_not_require_keywords(pages_path):
    fragment = read_fragment(pages_path, document="100_1", page=1)

    assert fragment.text == "Введение."
    assert fragment.location == "Страница 1"


def test_shortest_page_does_not_require_keywords(pages_path):
    fragment = read_fragment(pages_path, shortest=True)

    assert fragment.document_id == "100_1"
    assert fragment.page_number == 1
    assert fragment.text == "Введение."


def test_shortest_page_respects_document(pages_path):
    fragment = read_fragment(pages_path, document="200_1", shortest=True)

    assert fragment.document_id == "200_1"
    assert fragment.page_number == 2


def test_shortest_page_skips_blank_text_and_preserves_source(tmp_path):
    path = tmp_path / "short_pages.parquet"
    pl.DataFrame({
        "code_version": ["a"] * 4,
        "page_number": [1, 2, 3, 4],
        "text": [None, "  \n", " Короткий текст. ", "Длинный текст про взаимодействия."],
    }).write_parquet(path)

    fragment = read_fragment(path, shortest=True)

    assert fragment.page_number == 3
    assert fragment.text == " Короткий текст. "


def test_shortest_and_explicit_page_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        main(["--shortest", "--document", "a", "--page", "1"])
    assert error.value.code == 2


def test_rejects_unknown_document(pages_path):
    with pytest.raises(ValueError, match="страница не найдена"):
        read_fragment(pages_path, document="missing")


def test_rejects_ambiguous_page_source(pages_path):
    pages = pl.read_parquet(pages_path)
    pl.concat([pages, pages]).write_parquet(pages_path)

    with pytest.raises(ValueError, match="несколько записей"):
        read_fragment(pages_path)


def test_rejects_missing_source_columns(tmp_path):
    path = tmp_path / "wrong.parquet"
    pl.DataFrame({"text": ["Противопоказания."]}).write_parquet(path)

    with pytest.raises(ValueError, match="нет полей"):
        read_fragment(path)


def test_page_requires_document_before_loading_any_files():
    with pytest.raises(SystemExit) as error:
        main(["--page", "2"])

    assert error.value.code == 2
