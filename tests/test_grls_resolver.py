
import polars as pl

from src.nlp.grls_resolver import DrugResolver


def make_resolver(tmp_path):
    data = pl.DataFrame(
        {
            "trade_name_norm": [
                "берлиприл 10",
                "берлиприл 20",
                "эналаприл",
                "эналаприл",
                "берлиприл 5",
                "берлиприл плюс",
                "аторвастатин",
                "аторвастатин авексима",
                "бисопролол",
            ],
            "inn_norm": [
                "эналаприл",
                "эналаприл",
                "эналаприл",
                "эналаприл",
                "эналаприл",
                "гидрохлоротиазид + эналаприл",
                "аторвастатин",
                "аторвастатин",
                "бисопролол",
            ],
            "atx_code": [
                ["C09AA02"],
                ["C09AA02"],
                ["C09AA02"],
                ["C09BA02"],
                ["C09AA02"],
                ["C09BA02"],
                ["C10AA05"],
                ["C10AA05"],
                ["C07AB07"],
            ],
        }
    )
    path = tmp_path / "drugs.parquet"
    data.write_parquet(path)
    return DrugResolver(path)


def test_search_trade_name_with_dosage(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("Берлиприл 10 мг")

    assert len(result.matches) == 1
    assert result.matches[0].trade_name == "берлиприл 10"
    assert result.matches[0].atx_codes == ["C09AA02"]
    assert result.ambiguous is False


def test_search_by_inn_returns_all_matches(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("эналаприл")

    assert len(result.matches) == 5
    assert result.ambiguous is True
    assert {
        code
        for match in result.matches
        for code in match.atx_codes
    } == {"C09AA02", "C09BA02"}


def test_unknown_drug_returns_no_matches(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("несуществующийпрепарат")

    assert result.matches == []
    assert result.ambiguous is False


def test_search_trade_name_without_dosage(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("Берлиприл")

    assert {
        match.trade_name for match in result.matches
    } == {"берлиприл 5", "берлиприл 10", "берлиприл 20"}
    assert result.ambiguous is True


def test_search_is_case_insensitive(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("ЭНАЛАПРИЛ")

    assert len(result.matches) == 5
    assert result.ambiguous is True


def test_search_empty_query(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("   ")

    assert result.matches == []
    assert result.ambiguous is False


def test_search_combination_by_trade_name(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("Берлиприл плюс")

    assert len(result.matches) == 1
    assert result.matches[0].trade_name == "берлиприл плюс"
    assert result.matches[0].inn == "гидрохлоротиазид + эналаприл"
    assert result.matches[0].atx_codes == ["C09BA02"]
    assert result.ambiguous is False


def test_search_preserves_multiple_atx_codes(tmp_path):
    resolver = make_resolver(tmp_path)

    data = pl.read_parquet(resolver.parquet_path)
    data = data.with_columns(
        pl.when(pl.col("trade_name_norm") == "берлиприл 10")
        .then(pl.lit(["C09AA02", "C09AA03"]))
        .otherwise(pl.col("atx_code"))
        .alias("atx_code")
    )
    data.write_parquet(resolver.parquet_path)
    resolver = DrugResolver(resolver.parquet_path)

    result = resolver.search("Берлиприл 10 мг")

    assert len(result.matches) == 1
    assert result.matches[0].atx_codes == ["C09AA02", "C09AA03"]
    assert result.ambiguous is False


def test_search_atorvastatin_by_inn(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("аторвастатин")

    assert len(result.matches) == 2
    assert result.ambiguous is True
    assert all(match.inn == "аторвастатин" for match in result.matches)
    assert all(match.atx_codes == ["C10AA05"] for match in result.matches)


def test_search_bisoprolol_by_inn(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("бисопролол")

    assert len(result.matches) == 1
    assert result.matches[0].inn == "бисопролол"
    assert result.matches[0].atx_codes == ["C07AB07"]
    assert result.ambiguous is False


def test_search_combination_by_inn(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("гидрохлоротиазид + эналаприл")

    assert len(result.matches) == 1
    assert result.matches[0].inn == "гидрохлоротиазид + эналаприл"
    assert result.matches[0].atx_codes == ["C09BA02"]
    assert result.ambiguous is False


def test_ambiguous_search_returns_all_variants(tmp_path):
    resolver = make_resolver(tmp_path)

    result = resolver.search("эналаприл")

    trade_names = {match.trade_name for match in result.matches}

    assert trade_names == {
        "берлиприл 10",
        "берлиприл 20",
        "эналаприл",
        "берлиприл 5",
    }
    assert result.ambiguous is True