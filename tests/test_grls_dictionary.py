"""The ready normalized Parquet defines the primary drug dictionary."""
from unittest.mock import MagicMock

import polars as pl
import pytest

from src.graph.grls_dictionary import import_dictionary, prepare_dictionary
from src.graph.grls_search import GRLSGraphResolver


def write_dictionary(tmp_path, rows):
    path = tmp_path / "normalized.parquet"
    pl.DataFrame(rows, schema_overrides={"atx_code": pl.List(pl.String)}).write_parquet(path)
    return path


def row(**changes):
    return {"trade_name_norm": "конкор", "inn_norm": "бисопролол",
            "atx_code": ["C07AB07"], **changes}


def test_repeated_mappings_and_code_order_produce_one_drug(tmp_path):
    path = write_dictionary(tmp_path, [row(atx_code=["B", "A", "A"]), row(atx_code=["A", "B"])])
    drugs, report = prepare_dictionary(path)
    assert report.input_records == 2
    assert report.imported_records == 1
    assert report.duplicate_input_records == 1
    assert drugs[0]["atx_code"] == ["A", "B"]
    single, _ = prepare_dictionary(write_dictionary(tmp_path, [row(atx_code=["A", "B"])]))
    assert drugs[0]["drug_id"] == single[0]["drug_id"]


def test_different_inns_and_code_sets_remain_separate(tmp_path):
    path = write_dictionary(tmp_path, [row(), row(inn_norm="другое вещество"), row(atx_code=["C01"])])
    drugs, _ = prepare_dictionary(path)
    assert len({drug["drug_id"] for drug in drugs}) == 3
    assert {tuple(drug["atx_code"]) for drug in drugs} == {("C07AB07",), ("C01",)}


@pytest.mark.parametrize("changes", [
    {"trade_name_norm": " "}, {"inn_norm": None}, {"inn_norm": "~"},
    {"atx_code": []}, {"atx_code": [""]},
])
def test_invalid_dictionary_is_rejected_before_writes(tmp_path, changes):
    driver = MagicMock()
    with pytest.raises(ValueError, match="Invalid normalized drug"):
        import_dictionary(driver, write_dictionary(tmp_path, [row(), row(**changes)]))
    driver.session.assert_not_called()


def test_default_search_returns_unique_dictionary_mappings():
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.execute_read.return_value = [{
        "drug_id": "canonical-id", "trade_name": "конкор", "inn": "бисопролол",
        "atx_codes": ["C07AB07"],
    }]
    result = GRLSGraphResolver(driver).search("Конкор®")
    assert len(result.matches) == 1 and not result.ambiguous
    assert result.matches[0].trade_name == "конкор"
    assert result.matches[0].atx_codes == ["C07AB07"]
    assert "lp_id" not in result.matches[0].model_dump()


def test_empty_dictionary_search_does_not_connect():
    driver = MagicMock()
    assert GRLSGraphResolver(driver).search(" ").matches == []
    driver.session.assert_not_called()
