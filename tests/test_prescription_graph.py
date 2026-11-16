"""Проверки пути от названия лекарства до ответа из графа."""

import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest
from neo4j.exceptions import ServiceUnavailable
from pydantic import ValidationError

from src.graph import prescription_cli
from src.graph.prescription_graph import (
    CONTRAINDICATION_QUERY,
    INTERACTION_QUERY,
    Neo4jPrescriptionChecker,
)
from src.nlp.grls_resolver import DrugResolver

SOURCE = {
    "document_id": "synthetic-guideline",
    "location": "Синтетическая страница 2",
    "quote": "  Синтетическая цитата с условием.  ",
}
CONTRAINDICATION = {
    "drug_inn": "test-a",
    "icd10_code": "Z99.9",
    "conditions": ["Синтетическое условие."],
    "source": SOURCE,
}
INTERACTION = {
    "first_drug_inn": "test-a",
    "second_drug_inn": "test-b",
    "description": "Синтетическое влияние A на B.",
    "conditions": ["Синтетическое условие."],
    "source": SOURCE,
}


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "drugs.parquet"
    pl.DataFrame(
        {
            "trade_name_norm": ["альфа", "бета", "вариант 10", "вариант 20", "без мнн"],
            "inn_norm": ["test-a", "test-b", "test-a", "test-a", ""],
            "atx_code": [["TEST01"]] * 5,
        }
    ).write_parquet(path)
    driver = MagicMock()
    driver.__enter__.return_value = driver
    session = driver.session.return_value.__enter__.return_value
    transaction = MagicMock()
    transaction.run.return_value.data.return_value = []
    session.execute_read.side_effect = lambda work: work(transaction)
    checker = Neo4jPrescriptionChecker(
        driver, DrugResolver(path), database="test-database"
    )
    return checker, driver, transaction


def set_rows(transaction, relation, status="confirmed"):
    transaction.run.return_value.data.return_value = [
        {"status": status, "relation": copy.deepcopy(relation)}
    ]


def test_graph_disease_check_returns_source_and_uses_parameters(graph):
    checker, driver, transaction = graph
    set_rows(transaction, CONTRAINDICATION)

    result = checker.check_drug_disease("АЛЬФА", " Z99.9 ")

    assert result.status == "confirmed_match"
    assert result.matches[0].model_dump(mode="json") == CONTRAINDICATION
    transaction.run.assert_called_once_with(
        CONTRAINDICATION_QUERY, inn_key="name:test-a", icd10_code="Z99.9"
    )
    driver.session.assert_called_once_with(
        database="test-database", default_access_mode="READ"
    )


@pytest.mark.parametrize(
    ("first", "second", "first_key", "second_key"),
    [
        ("альфа", "бета", "name:test-a", "name:test-b"),
        ("бета", "альфа", "name:test-b", "name:test-a"),
    ],
)
def test_graph_interaction_keeps_original_participant_roles(
    graph, first, second, first_key, second_key
):
    checker, _, transaction = graph
    set_rows(transaction, INTERACTION)

    result = checker.check_drug_drug(first, second)

    assert result.status == "confirmed_match"
    assert result.matches[0].model_dump(mode="json") == INTERACTION
    transaction.run.assert_called_once_with(
        INTERACTION_QUERY, first_key=first_key, second_key=second_key
    )


@pytest.mark.parametrize("status", ["disputed", "unresolved", "pending", None])
@pytest.mark.parametrize("kind", ["disease", "interaction"])
def test_unconfirmed_graph_rows_do_not_become_matches(graph, status, kind):
    checker, _, transaction = graph
    # Даже некорректное неподтверждённое утверждение не попадает в результат.
    set_rows(transaction, {}, status=status)

    if kind == "disease":
        result = checker.check_drug_disease("альфа", "Z99.9")
    else:
        result = checker.check_drug_drug("альфа", "бета")

    assert result.status == "insufficient_data"
    assert result.matches == ()
    query = transaction.run.call_args.args[0]
    assert "WHERE r.status = 'confirmed'" in query


@pytest.mark.parametrize(
    ("name", "code", "status"),
    [
        ("вариант", "Z99.9", "ambiguous"),
        ("неизвестный", "Z99.9", "insufficient_data"),
        ("без мнн", "Z99.9", "insufficient_data"),
        (" ", "Z99.9", "insufficient_data"),
        ("альфа", None, "insufficient_data"),
        ("альфа", " ", "insufficient_data"),
    ],
)
def test_disease_with_unresolved_input_does_not_query_graph(graph, name, code, status):
    checker, driver, _ = graph

    result = checker.check_drug_disease(name, code)

    assert result.status == status
    assert result.matches == ()
    driver.session.assert_not_called()


@pytest.mark.parametrize(
    ("first", "second", "status"),
    [
        ("вариант", "бета", "ambiguous"),
        ("альфа", "вариант", "ambiguous"),
        ("неизвестный", "бета", "insufficient_data"),
        ("альфа", "неизвестный", "insufficient_data"),
        ("без мнн", "бета", "insufficient_data"),
        ("альфа", "без мнн", "insufficient_data"),
    ],
)
def test_interaction_with_unresolved_input_does_not_query_graph(
    graph, first, second, status
):
    checker, driver, _ = graph

    result = checker.check_drug_drug(first, second)

    assert result.status == status
    assert result.matches == ()
    driver.session.assert_not_called()


def test_empty_graph_does_not_imply_safety(graph):
    checker, _, _ = graph

    assert checker.check_drug_disease("альфа", "Z99.9").status == "insufficient_data"
    assert checker.check_drug_drug("альфа", "бета").status == "insufficient_data"


@pytest.mark.parametrize("kind", ["disease", "interaction"])
def test_graph_connection_failure_is_reported_as_error(graph, kind):
    checker, driver, _ = graph
    session = driver.session.return_value.__enter__.return_value
    session.execute_read.side_effect = ServiceUnavailable("test failure")

    with pytest.raises(ServiceUnavailable):
        if kind == "disease":
            checker.check_drug_disease("альфа", "Z99.9")
        else:
            checker.check_drug_drug("альфа", "бета")


@pytest.mark.parametrize("kind", ["disease", "interaction"])
@pytest.mark.parametrize("invalid_field", ["quote", "conditions"])
def test_incomplete_confirmed_evidence_blocks_answer(graph, kind, invalid_field):
    checker, _, transaction = graph
    relation = copy.deepcopy(
        CONTRAINDICATION if kind == "disease" else INTERACTION
    )
    if invalid_field == "quote":
        relation["source"]["quote"] = " "
    else:
        relation["conditions"] = None
    set_rows(transaction, relation)

    with pytest.raises(ValidationError):
        if kind == "disease":
            checker.check_drug_disease("альфа", "Z99.9")
        else:
            checker.check_drug_drug("альфа", "бета")


def test_graph_answer_is_reproducible_with_multiple_sources(graph):
    checker, _, transaction = graph
    second = copy.deepcopy(CONTRAINDICATION)
    second["source"]["document_id"] = "another-synthetic-guideline"
    rows = [
        {"status": "confirmed", "relation": relation}
        for relation in [CONTRAINDICATION, second]
    ]
    transaction.run.return_value.data.return_value = rows
    forward = checker.check_drug_disease("альфа", "Z99.9")
    transaction.run.return_value.data.return_value = list(reversed(rows))
    backward = checker.check_drug_disease("альфа", "Z99.9")

    assert forward.model_dump_json() == backward.model_dump_json()
    assert len(forward.matches) == 2


def configure_cli(monkeypatch, driver):
    # Настройки тестовые, содержимое локального .env не используется.
    monkeypatch.setitem(
        sys.modules, "src.config",
        SimpleNamespace(settings=SimpleNamespace(
            neo4j_uri="bolt://test.invalid:7687",
            neo4j_user="test-user",
            neo4j_password="test-secret",
        )),
    )
    monkeypatch.setattr(
        prescription_cli.GraphDatabase, "driver", lambda *args, **kwargs: driver
    )


@pytest.mark.parametrize("kind", ["disease", "interaction"])
def test_cli_resolves_names_reads_graph_and_prints_json(
    graph, monkeypatch, capsys, kind
):
    checker, driver, transaction = graph
    configure_cli(monkeypatch, driver)
    if kind == "disease":
        set_rows(transaction, CONTRAINDICATION)
        arguments = ["drug-disease", "--drug", "альфа", "--icd10", "Z99.9"]
    else:
        set_rows(transaction, INTERACTION)
        arguments = ["drug-drug", "--first", "бета", "--second", "альфа"]

    exit_code = prescription_cli.main(
        ["--dictionary", str(checker.resolver.parquet_path)] + arguments
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "confirmed_match"
    assert output["matches"][0]["source"] == SOURCE


def test_cli_connection_error_has_nonzero_exit_and_keeps_secrets_private(
    graph, monkeypatch, capsys
):
    checker, driver, _ = graph
    configure_cli(monkeypatch, driver)
    driver.session.return_value.__enter__.return_value.execute_read.side_effect = (
        ServiceUnavailable("test-secret")
    )

    exit_code = prescription_cli.main([
        "--dictionary", str(checker.resolver.parquet_path),
        "drug-disease", "--drug", "альфа", "--icd10", "Z99.9",
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "ServiceUnavailable" in captured.err
    assert "test-secret" not in captured.err
