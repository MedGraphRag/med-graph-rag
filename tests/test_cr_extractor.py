
import pymupdf
import pytest

from src.nlp.cr_extractor import (
    extract_pages,
    extract_section_headings,
    extract_tables,
    is_valid_section_heading,
)


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



def test_extract_section_headings_keeps_valid_headings():
    headings = [
        "1. Краткая информация",
        "1.1 Определение заболевания",
        "2.5 Иные диагностические исследования",
        "3.3 Хирургическое лечение",
        "7. Дополнительная информация",
    ]

    formatted_text = {
        "blocks": [
            {
                "lines": [
                    {
                        "spans": [
                            {
                                "text": heading,
                                "flags": 16,
                                "size": 12,
                            }
                        ]
                    }
                ]
            }
            for heading in headings
        ]
    }

    assert extract_section_headings(formatted_text) == headings




def test_extract_section_headings_rejects_bibliography():
    text = (
        "496. Tiwari S, Løchen ML, Jacobsen BK, et al. "
        "CHA2DS2-VASc score. Open Heart. 2016;3(2);e000439. "
        "doi:10.1136/openhrt-2016-000439"
    )

    formatted_text = {
        "blocks": [
            {
                "lines": [
                    {
                        "spans": [
                            {
                                "text": text,
                                "flags": 16,
                                "size": 12,
                            }
                        ]
                    }
                ]
            }
        ]
    }

    assert extract_section_headings(formatted_text) == []


def test_is_valid_section_heading_rejects_long_text():
    assert not is_valid_section_heading("1. " + "слово " * 40)


def test_extract_tables_preserves_merged_cells(tmp_path):
    path = tmp_path / "table.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Таблица А1", fontname="china-s")
        page.draw_rect((72, 90, 272, 170))
        for y in (110, 130, 150):
            page.draw_line((72, y), (272, y))
        page.draw_line((172, 110), (172, 170))
        page.insert_text((80, 104), "Measurements")
        for y, left, right in (
            (124, "Sample", "Value"),
            (144, "A", "0,5 mg"),
            (164, "B", "1 mg"),
        ):
            page.insert_text((80, y), left)
            page.insert_text((180, y), right)
        document.save(path)

    (table,) = extract_tables(path, pages=extract_pages(path))
    assert table["caption"] == "Таблица А1"
    assert table["status"] == "extracted"
    header, *rows = table["rows"]
    assert header[0] == "Measurements"
    assert header[1:] and all(cell is None for cell in header[1:])
    assert [(row[0], row[-1]) for row in rows] == [
        ("Sample", "Value"),
        ("A", "0,5 mg"),
        ("B", "1 mg"),
    ]


def test_extract_tables_reports_unmatched_caption(tmp_path):
    path = tmp_path / "caption_only.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Таблица А1", fontname="china-s")
        document.save(path)

    (table,) = extract_tables(path)
    assert table["page"] == 1
    assert table["caption"] == "Таблица А1"
    assert table["status"] == "unprocessed"
    assert table["rows"] is None
    assert table["message"]
