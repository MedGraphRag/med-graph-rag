
import re
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class DrugMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_name: str
    inn: str | None
    atx_codes: list[str] = Field(default_factory=list)


class DrugSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    matches: list[DrugMatch] = Field(default_factory=list)
    ambiguous: bool = False


def normalize_query(name: str) -> str:
    """Normalize an INN or a general drug name."""
    name = name.lower()
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(
        r"\d+(?:[.,]\d+)?\s*(?:мкг|мг|мл|г|%|ме)(?:/\s*(?:мл|г))?",
        " ",
        name,
    )
    name = re.sub(r"[®™]", "", name)
    name = re.sub(r"[^\w\s+.-]", " ", name)
    return re.sub(r"\s+", " ", name).strip(" ,;")


def normalize_trade_query(name: str) -> str:
    """Normalize a trade name while preserving its dosage number."""
    name = name.lower()
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(
        r"\s*(?:мкг|мг|мл|г|%|ме)(?:/\s*(?:мл|г))?",
        "",
        name,
    )
    name = re.sub(r"[®™]", "", name)
    name = re.sub(r"[^\w\s+.-]", " ", name)
    return re.sub(r"\s+", " ", name).strip(" ,;")


class DrugResolver:
    def __init__(self, parquet_path: str | Path):
        self.parquet_path = Path(parquet_path)
        self.df = pl.read_parquet(self.parquet_path)

        required_columns = {"trade_name_norm", "inn_norm", "atx_code"}
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(
                "Parquet is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        self.df = self.df.with_columns(
            pl.col("trade_name_norm").fill_null("").cast(pl.String),
            pl.col("inn_norm").fill_null("").cast(pl.String),
            pl.col("atx_code").fill_null(pl.lit([], dtype=pl.List(pl.String))),
        ).with_columns(
            pl.col("trade_name_norm")
            .map_elements(normalize_trade_query, return_dtype=pl.String)
            .alias("_trade_search"),
            pl.col("inn_norm")
            .map_elements(normalize_query, return_dtype=pl.String)
            .alias("_inn_search"),
        )

    def search(self, query: str) -> DrugSearchResult:
        normalized = normalize_query(query)
        normalized_trade = normalize_trade_query(query)

        if not normalized and not normalized_trade:
            return DrugSearchResult(query=query)

        has_dosage = bool(
            re.search(
                r"\d+(?:[.,]\d+)?\s*(?:мкг|мг|мл|г|%|ме)",
                query.lower(),
            )
        )

        if has_dosage:
            trade_match = pl.col("_trade_search") == normalized_trade
        else:
            trade_match = (
                (pl.col("_trade_search") == normalized_trade)
                | pl.col("_trade_search").str.contains(
                    "^" + re.escape(normalized_trade) + r"\s+\d"
                )
            )

        matches = self.df.filter(
            (trade_match & (pl.col("_trade_search") != ""))
            | (
                (pl.col("_inn_search") == normalized)
                & (pl.col("_inn_search") != "")
            )
        ).unique(
            subset=["trade_name_norm", "inn_norm", "atx_code"]
        ).sort(["trade_name_norm", "inn_norm"])

        drugs = [
            DrugMatch(
                trade_name=row["trade_name_norm"],
                inn=row["inn_norm"] or None,
                atx_codes=row["atx_code"] or [],
            )
            for row in matches.iter_rows(named=True)
        ]

        # Multiple registry entries for the same trade name and INN can
        # carry different ATX codes. Keep those codes, but do not count
        # them as separate drug-name variants when determining ambiguity.
        variants = {
            (drug.trade_name, drug.inn)
            for drug in drugs
        }

        return DrugSearchResult(
            query=query,
            matches=drugs,
            ambiguous=len(variants) > 1,
        )
