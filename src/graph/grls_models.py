"""Validated GRLS entities and deterministic identities for graph import."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.nlp.grls_etl import NormalizedGRLSRecord
from src.nlp.grls_resolver import DrugMatch, DrugSearchResult, normalize_query, normalize_trade_query


class GraphEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GRLSRegistration(GraphEntity):
    lp_id: str = Field(min_length=1)


class TradeName(GraphEntity):
    name_norm: str = Field(min_length=1)

    @field_validator("name_norm")
    @classmethod
    def normalize(cls, value: str) -> str:
        value = normalize_trade_query(value)
        if not value:
            raise ValueError("Trade name is empty after normalization")
        return value


class INN(GraphEntity):
    """A code OR an unsplit source name; no inferred code/name mapping."""

    code: str | None = Field(default=None, min_length=1)
    name_norm: str | None = Field(default=None, min_length=1)

    @field_validator("name_norm")
    @classmethod
    def normalize(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = normalize_query(value)
        if not value or value == "~":
            raise ValueError("INN name is missing")
        return value

    @model_validator(mode="after")
    def check_identity(self) -> INN:
        if (self.code is None) == (self.name_norm is None):
            raise ValueError("Provide either an INN code or a complete source name")
        return self

    @property
    def key(self) -> str:
        return f"code:{self.code}" if self.code is not None else f"name:{self.name_norm}"


class ATC(GraphEntity):
    code: str = Field(min_length=1)

    @field_validator("code")
    @classmethod
    def normalize(cls, value: str) -> str:
        return value.upper()


class GRLSRecord(NormalizedGRLSRecord):
    """One immutable observation of a registration in a source snapshot.

    Source pages are locators, not identity: pagination can repeat a record.
    A changed payload under the same snapshot/lp_id is a conflict, not an update.
    """

    lp_id: str = Field(min_length=1)
    source_snapshot: str = Field(min_length=1)
    source_page: str = Field(min_length=1)
    condition_id: Literal["Д", "П", "ИЗМ"]

    @field_validator("lp_id", "source_snapshot", "source_page", "condition_id", mode="before")
    @classmethod
    def strip_required(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("commercial_name", "mnn_name", mode="before")
    @classmethod
    def clean_name(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return None if value in ("", "~") else value
        return value

    @field_validator("mnn_codes", "atx_codes")
    @classmethod
    def clean_codes(cls, values: list[str], info) -> list[str]:
        if info.field_name == "atx_codes":
            return sorted({value.strip().upper() for value in values if value.strip()})
        return sorted({value.strip() for value in values if value.strip()})

    @property
    def record_id(self) -> str:
        return _digest([self.source_snapshot, self.lp_id])

    @property
    def content_hash(self) -> str:
        return _digest(self.model_dump(exclude={"source_snapshot", "source_page"}))

    def graph_row(self) -> dict:
        registration = GRLSRegistration(lp_id=self.lp_id)
        trades = [TradeName(name_norm=self.commercial_name)] if self.commercial_name else []
        inns = [INN(code=code) for code in self.mnn_codes]
        if not inns and self.mnn_name:
            inns = [INN(name_norm=self.mnn_name)]
        return {
            "lp_id": registration.lp_id,
            "record_id": self.record_id,
            "content_hash": self.content_hash,
            "properties": {
                **self.model_dump(exclude={"source_page"}),
                "inn_norm": normalize_query(self.mnn_name) if self.mnn_name else None,
            },
            "source_page": self.source_page,
            "trades": [trade.model_dump() for trade in trades],
            "inns": [{"key": inn.key, **inn.model_dump()} for inn in inns],
            "atcs": [ATC(code=code).model_dump() for code in self.atx_codes],
        }


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class GRLSImportReport(GraphEntity):
    input_records: int = 0
    imported_records: int = 0
    duplicate_input_records: int = 0
    skipped_missing_id: int = 0
    skipped_status: int = 0
    batches: int = 0
    nodes_created: int = 0
    relationships_created: int = 0


class GRLSDrugMatch(DrugMatch):
    """Search contract extends DrugMatch with registration-scoped evidence."""

    record_id: str
    lp_id: str
    ru_number: str | None = None
    mnn_codes: list[str] = Field(default_factory=list)
    source_snapshot: str
    source_pages: list[str] = Field(default_factory=list)


class GRLSDrugSearchResult(DrugSearchResult):
    matches: list[GRLSDrugMatch] = Field(default_factory=list)


class GRLSDictionaryEntry(GraphEntity):
    """One unique mapping from the ready normalized search Parquet."""

    trade_name_norm: str = Field(min_length=1)
    inn_norm: str = Field(min_length=1)
    atx_code: list[str] = Field(min_length=1)

    @field_validator("inn_norm")
    @classmethod
    def require_inn(cls, value: str) -> str:
        if value == "~":
            raise ValueError("INN name is missing")
        return value

    @field_validator("atx_code")
    @classmethod
    def canonical_codes(cls, values: list[str]) -> list[str]:
        if any(not code.strip() for code in values):
            raise ValueError("ATX code is empty")
        return sorted({code.strip() for code in values})

    @property
    def drug_id(self) -> str:
        return _digest([self.trade_name_norm, self.inn_norm, self.atx_code])

    def graph_row(self) -> dict:
        return {"drug_id": self.drug_id, **self.model_dump()}


class GRLSDictionaryMatch(DrugMatch):
    drug_id: str


class GRLSDictionarySearchResult(DrugSearchResult):
    matches: list[GRLSDictionaryMatch] = Field(default_factory=list)
