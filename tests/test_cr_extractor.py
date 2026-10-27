
import pymupdf
import pytest

from src.nlp.cr_extractor import extract_pages


def create_pdf(path, page_texts):
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)


def test_extract_pages_preserves_text_and_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path, ["First page", "Second page"])

    pages = extract_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "First page" in pages[0]["text"]
    assert pages[1]["page"] == 2
    assert "Second page" in pages[1]["text"]


def test_extract_pages_preserves_empty_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path, ["Text", ""])

    pages = extract_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0]["text"]
    assert pages[1]["page"] == 2
    assert pages[1]["text"] == ""


def test_extract_pages_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pages(tmp_path / "missing.pdf")


def test_extract_pages_rejects_non_pdf_file(tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("Some text", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF"):
        extract_pages(path)


def test_extract_pages_rejects_empty_pdf(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    with pytest.raises(RuntimeError, match="Failed to extract text"):
        extract_pages(pdf_path)

