"""Offline checks of graph identities and the ETL/import/search contract."""
from unittest.mock import MagicMock

import polars as pl
import pytest
from pydantic import ValidationError

from src.graph.grls_loader import import_grls, prepare_records
from src.graph.grls_models import ATC, INN, GRLSRecord, TradeName
from src.graph.grls_search import GRLSGraphResolver
from src.nlp.grls_etl import build_normalized_row, normalize_snapshot


def make_row(**overrides):
    row = build_normalized_row(
        {"lpId": "registration-1", "commercialName": "Препарат А®",
         "mnnName": "Вещество Б", "mnnCode": ["m1"],
         "atx": ["A01AA01"], "conditionId": "Д", "ruNumber": "РУ-1"},
        fix_mojibake=False, source_snapshot="snapshot-1", source_page="page_1.json",
    )
    row.update(overrides)
    return row


def write_records(tmp_path, rows):
    path = tmp_path / "records.parquet"
    pl.DataFrame(rows, schema_overrides={"mnn_codes": pl.List(pl.String), "atx_codes": pl.List(pl.String)}).write_parquet(path)
    return path


def test_registration_identity_and_snapshot_evidence_are_separate():
    first = GRLSRecord.model_validate(make_row())
    page_repeat = GRLSRecord.model_validate(make_row(source_page="page_2.json"))
    other_registration = GRLSRecord.model_validate(make_row(lp_id="registration-2"))
    new_snapshot = GRLSRecord.model_validate(make_row(source_snapshot="snapshot-2"))
    assert first.record_id == page_repeat.record_id
    assert first.content_hash == page_repeat.content_hash
    assert first.record_id != other_registration.record_id
    assert first.record_id != new_snapshot.record_id
    assert first.graph_row()["lp_id"] == new_snapshot.graph_row()["lp_id"]


def test_multiple_codes_do_not_infer_names_or_pairs():
    record = GRLSRecord.model_validate(make_row(
        mnn_codes=["m2", " m1 ", "m1", ""],
        mnn_name="Вещество Б + Вещество В", atx_codes=["a01aa01", "A01AA02", "A01AA01"],
    ))
    row = record.graph_row()
    assert row["properties"]["mnn_name"] == "Вещество Б + Вещество В"
    assert row["inns"] == [
        {"key": "code:m1", "code": "m1", "name_norm": None},
        {"key": "code:m2", "code": "m2", "name_norm": None},
    ]
    assert row["atcs"] == [{"code": "A01AA01"}, {"code": "A01AA02"}]


def test_uncoded_combination_is_not_split_or_matched_to_a_code():
    record = GRLSRecord.model_validate(make_row(mnn_codes=[], mnn_name="Вещество Б + Вещество В"))
    assert record.graph_row()["inns"] == [{
        "key": "name:вещество б + вещество в", "code": None,
        "name_norm": "вещество б + вещество в",
    }]


def test_placeholder_is_missing_and_sparse_record_is_preserved():
    row = GRLSRecord.model_validate(make_row(mnn_name=" ~ ", mnn_codes=[], atx_codes=[])).graph_row()
    assert row["inns"] == []
    assert row["atcs"] == []
    assert row["trades"] == [{"name_norm": "препарат а"}]


def test_entity_keys_keep_dosages_and_code_name_namespaces():
    assert TradeName(name_norm="Берлиприл® 10 мг").name_norm == "берлиприл 10"
    assert ATC(code=" a01aa01 ").code == "A01AA01"
    assert INN(code="123").key != INN(name_norm="123").key
    with pytest.raises(ValidationError):
        INN(code="123", name_norm="Неподтверждённое название")


def test_prepare_counts_skips_duplicates_and_preserves_pages(tmp_path):
    path = write_records(tmp_path, [
        make_row(), make_row(source_page="page_2.json"),
        make_row(lp_id=None), make_row(lp_id="expired", condition_id="И"),
        make_row(lp_id="unknown", condition_id=None),
    ])
    rows, report = prepare_records(path)
    assert report.input_records == 5
    assert report.imported_records == 1
    assert report.duplicate_input_records == 1
    assert report.skipped_missing_id == 1
    assert report.skipped_status == 2
    assert {row["source_page"] for row in rows} == {"page_1.json", "page_2.json"}


def test_prepare_keeps_changed_and_active_observations_separate(tmp_path):
    path = write_records(tmp_path, [
        make_row(condition_id="ИЗМ", source_snapshot="snapshot-old"),
        make_row(condition_id="Д", source_snapshot="snapshot-new"),
        make_row(lp_id="excluded", condition_id="ИСКЛ"),
    ])
    rows, report = prepare_records(path)
    assert report.imported_records == 2
    assert report.skipped_status == 1
    assert {row["properties"]["condition_id"] for row in rows} == {"Д", "ИЗМ"}
    assert rows[0]["lp_id"] == rows[1]["lp_id"]
    assert rows[0]["record_id"] != rows[1]["record_id"]


@pytest.mark.parametrize("changes", [
    {"source_snapshot": None}, {"source_page": " "},
    {"mnn_codes": None}, {"commercial_name": "®"},
])
def test_invalid_input_aborts_before_database_writes(tmp_path, changes):
    driver = MagicMock()
    path = write_records(tmp_path, [make_row(), make_row(lp_id="registration-2", **changes)])
    with pytest.raises(ValueError, match="Invalid GRLS record at row 2"):
        import_grls(driver, path, batch_size=1)
    driver.session.assert_not_called()


def test_conflicting_observation_aborts_before_database_writes(tmp_path):
    driver = MagicMock()
    path = write_records(tmp_path, [make_row(), make_row(atx_codes=["A01AA02"])])
    with pytest.raises(ValueError, match="Conflicting GRLS observations"):
        import_grls(driver, path)
    driver.session.assert_not_called()


def test_rejects_search_dictionary_instead_of_provenance_records(tmp_path):
    path = tmp_path / "dictionary.parquet"
    pl.DataFrame({"trade_name_norm": ["препарат"], "inn_norm": ["вещество"], "atx_code": [["A01AA01"]]}).write_parquet(path)
    with pytest.raises(ValueError, match="missing columns"):
        prepare_records(path)


def test_batch_size_must_be_positive(tmp_path):
    driver = MagicMock()
    with pytest.raises(ValueError, match="positive"):
        import_grls(driver, tmp_path / "unused.parquet", batch_size=0)
    driver.session.assert_not_called()


def test_batch_sizes_and_committed_counters(tmp_path):
    path = write_records(tmp_path, [make_row(lp_id=f"r-{i}") for i in range(5)])
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    summary = MagicMock()
    summary.counters.nodes_created = 3
    summary.counters.relationships_created = 2
    session.execute_write.return_value = summary
    report = import_grls(driver, path, batch_size=2, database="test")
    assert [len(call.args[1]) for call in session.execute_write.call_args_list] == [2, 2, 1]
    assert report.batches == 3
    assert report.nodes_created == 9
    assert report.relationships_created == 6


def test_etl_output_is_accepted_without_transforming_raw(tmp_path):
    import json

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    raw = [{"lpId": "1", "commercialName": "Препарат А", "mnnName": "Вещество Б",
            "mnnCode": ["1", "2"], "atx": ["A01AA01", "A01AA02"], "conditionId": "П"}]
    source = snapshot / "page_000001.json"
    source.write_text(json.dumps(raw, ensure_ascii=False))
    before = source.read_bytes()
    normalize_snapshot(snapshot, tmp_path / "grls.parquet", allow_incomplete=True)
    rows, report = prepare_records(tmp_path / "grls_records.parquet")
    assert report.imported_records == 1
    assert len(rows[0]["inns"]) == len(rows[0]["atcs"]) == 2
    assert source.read_bytes() == before


def test_search_preserves_registration_scopes_and_existing_result_contract():
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.execute_read.return_value = [
        {"record_id": "a", "lp_id": "r1", "trade_name": "препарат а", "inn": "Вещество Б",
         "source_snapshot": "snapshot-1", "source_pages": ["page_1.json"], "mnn_codes": ["m1"], "atx_codes": ["A01AA01"]},
        {"record_id": "b", "lp_id": "r2", "trade_name": "препарат а", "inn": "Вещество В",
         "source_snapshot": "snapshot-1", "source_pages": ["page_2.json"], "mnn_codes": ["m2"], "atx_codes": ["A01AA02"]},
    ]
    result = GRLSGraphResolver(driver).search_records("Препарат А®")
    assert result.ambiguous
    assert result.matches[0].inn == "вещество б"
    assert result.matches[0].atx_codes == ["A01AA01"]
    assert result.matches[1].atx_codes == ["A01AA02"]
    assert result.matches[0].lp_id != result.matches[1].lp_id


def test_empty_query_does_not_connect():
    driver = MagicMock()
    assert GRLSGraphResolver(driver).search_records("   ").matches == []
    driver.session.assert_not_called()
