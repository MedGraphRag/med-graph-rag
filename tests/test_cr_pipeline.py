import json

import polars as pl
import pymupdf
import pytest

from src.nlp.cr_pipeline import run_pipeline


@pytest.mark.parametrize(
    "native_text,ocr_result,status,attempts,errors",
    [
        (True, "fail", "partial", 1, 1),
        (False, "fail", "error", 1, 1),
        (False, "text", "ok", 1, 0),
        (False, "blank", "empty", 1, 0),
    ],
)
def test_pipeline_reports_ocr_and_preserves_pages(
    tmp_path, monkeypatch, native_text, ocr_result, status, attempts, errors,
):
    pdf = tmp_path / "pdf"
    pdf.mkdir()
    with pymupdf.open() as document:
        if native_text:
            document.new_page().insert_text((72, 72), "Native page")
        document.new_page()
        document.save(pdf / "1_1.pdf")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"CodeVersion": "1_1"}]))

    def recognize(page, **kwargs):
        if ocr_result == "fail":
            raise RuntimeError("OCR unavailable")
        if ocr_result == "text":
            page.insert_text((72, 72), "Recognized text")
        return page.get_textpage()

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", recognize)
    successful, failed = run_pipeline(
        manifest_path=manifest, pdf_dir=pdf,
        output_path=tmp_path / "pages.parquet",
        report_path=tmp_path / "report.parquet",
        table_output_path=tmp_path / "tables.parquet",
    )

    report = pl.read_parquet(tmp_path / "report.parquet").row(0, named=True)
    pages = pl.read_parquet(tmp_path / "pages.parquet")
    assert report["status"] == status
    assert report["ocr_pages"] == attempts
    assert report["ocr_failed_pages"] == errors
    assert len(report["ocr_messages"]) == errors
    assert pages.height == (2 if native_text else 1)
    assert pages["ocr_attempted"].sum() == attempts
    assert pages["ocr_error"].drop_nulls().len() == errors
    assert (successful, failed) == ((1, 0) if status == "ok" else (0, 1))
    if native_text:
        assert pages["text"][0] == "Native page"
    if errors:
        assert report["ocr_messages"] == [
            f"Page {pages.height}: OCR unavailable",
        ]


def test_parallel_pipeline_passes_no_ocr_to_workers(tmp_path):
    pdf = tmp_path / "pdf"
    pdf.mkdir()
    records = []
    for code in ("1_1", "2_1"):
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), "Native page")
            document.new_page()
            document.save(pdf / f"{code}.pdf")
        records.append({"CodeVersion": code})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))

    assert run_pipeline(
        manifest_path=manifest, pdf_dir=pdf,
        output_path=tmp_path / "pages.parquet",
        report_path=tmp_path / "report.parquet",
        table_output_path=tmp_path / "tables.parquet",
        workers=4, ocr=False, tessdata="/missing/models",
    ) == (2, 0)
    pages = pl.read_parquet(tmp_path / "pages.parquet")
    assert pages.height == 4
    assert not pages["ocr_attempted"].any()
    assert pages["ocr_error"].drop_nulls().len() == 0
