"""Регрессии PDF API: CodeVersion вместо Id и ошибки IIS под HTTP 200."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.nlp.downloader import Recommendation, download_records, fetch_pdf, is_pdf, main

PDF = b"%PDF-1.7\nexample\n%%EOF\n"
RECORD = {"Id": 2235, "Code": 1078, "Version": 1, "CodeVersion": "1078_1"}


def response(content=PDF, status=200):
    return SimpleNamespace(status_code=status, content=content, headers={"content-type": "application/pdf"})


def test_pdf_uses_code_version_not_internal_id():
    session = Mock()
    session.get.return_value = response()
    record = Recommendation.model_validate(RECORD)
    assert fetch_pdf(session, record.code_version) == PDF
    assert session.get.call_args.kwargs["params"] == {"op": "GetClinrecPdf", "id": "1078_1"}
    assert Recommendation.model_validate({"Code": 1078, "Version": 1}).code_version == "1078_1"
    with pytest.raises(ValueError):
        Recommendation.model_validate({"Id": 2235})


def test_iis_error_under_http_200_is_not_saved(tmp_path):
    session = Mock()
    session.get.return_value = response(b"<p>iisnode encountered an error</p>HTTP status: 500")
    with patch("src.nlp.downloader.requests.Session") as factory, patch("src.nlp.downloader.time.sleep"):
        factory.return_value.__enter__.return_value = session
        assert download_records([RECORD], tmp_path, delay=0) == (0, 0, 1)
    assert session.get.call_count == 3
    assert not list(tmp_path.iterdir())


def test_retry_temporary_error_then_save_and_resume(tmp_path):
    session = Mock()
    session.get.side_effect = [response(b"unavailable", 503), response()]
    with patch("src.nlp.downloader.requests.Session") as factory, patch("src.nlp.downloader.time.sleep"):
        factory.return_value.__enter__.return_value = session
        assert download_records([RECORD], tmp_path, delay=0) == (1, 0, 0)
        assert download_records([RECORD], tmp_path, delay=0) == (0, 1, 0)
    assert session.get.call_count == 2
    assert (tmp_path / "1078_1.pdf").read_bytes() == PDF
    assert not list(tmp_path.glob("*.part"))


def test_html_and_truncated_pdf_are_rejected():
    assert not is_pdf(b"<html>" + b"x" * 20000)
    assert not is_pdf(b"%PDF-1.7\ntruncated")
    session = Mock()
    session.get.return_value = response(b"<html>Access denied</html>")
    with pytest.raises(ValueError, match="Access denied"):
        fetch_pdf(session, "1078_1")
    assert session.get.call_count == 1


def test_existing_invalid_file_is_preserved(tmp_path):
    path = tmp_path / "1078_1.pdf"
    path.write_bytes(b"<html>previous error</html>")
    with patch("src.nlp.downloader.requests.Session") as factory:
        assert download_records([RECORD], tmp_path, delay=0) == (0, 0, 1)
    factory.return_value.__enter__.return_value.get.assert_not_called()
    assert path.read_bytes() == b"<html>previous error</html>"


def test_missing_manifest_exits_with_error(tmp_path):
    assert main(["--manifest", str(tmp_path / "missing.json")]) == 1
