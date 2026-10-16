"""Normalize raw GRLS API snapshots into a Polars Parquet table.

Raw page responses are never edited. Optional mojibake repair applies only to
selected normalized text columns and preserves the original value in *_raw
columns so that transformations remain auditable.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

LOG = logging.getLogger("grls_etl")
_MOJIBAKE_MARKERS = re.compile(r"(?:Р[А-Яа-яЁё]|С[А-Яа-яЁё]|вЂ|Ð.|Ã.)")


class NormalizedGRLSRecord(BaseModel):
    """Minimal provenance; bulky source fields stay in immutable raw JSON."""
    model_config = ConfigDict(extra="forbid")
    source_snapshot: str | None = None
    source_page: str | None = None
    lp_id: str | None = None
    ru_number: str | None = None
    commercial_name_raw: str | None = None
    commercial_name: str | None = None
    mnn_name_raw: str | None = None
    mnn_name: str | None = None
    mnn_codes: list[str]
    atx_codes: list[str]
    condition_id: str | None = None
    release_forms_json: str


def _as_optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def extract_records(payload: Any) -> list[dict[str, Any]] | None:
    """Extract records from the raw API JSON, matching the scraper's logic."""
    record_keys = {"lpId", "commercialName", "dossierId", "ruNumber"}
    container_keys = ("content", "items", "results", "records", "data", "rows", "medicaments")

    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(x, dict) for x in payload) and any(
            record_keys.intersection(x.keys()) for x in payload
        ):
            return payload
        return None
    if not isinstance(payload, dict):
        return None

    for key in container_keys:
        value = payload.get(key)
        if isinstance(value, list):
            if not value:
                return []
            if all(isinstance(x, dict) for x in value) and any(
                record_keys.intersection(x.keys()) for x in value
            ):
                return value

    for value in payload.values():
        if isinstance(value, (dict, list)):
            found = extract_records(value)
            if found is not None:
                return found
    return None


def repair_mojibake(value: Any) -> Any:
    """Try reversible CP1251-bytes -> UTF-8 repair only if text looks corrupted.

    The returned value may be repaired; callers should preserve the raw value.
    Other object types and normal-looking strings are returned unchanged.
    """
    if not isinstance(value, str) or not _MOJIBAKE_MARKERS.search(value):
        return value
    try:
        candidate = value.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    before = len(_MOJIBAKE_MARKERS.findall(value))
    after = len(_MOJIBAKE_MARKERS.findall(candidate))
    return candidate if after < before else value


def repair_nested(value: Any) -> Any:
    """Apply optional string repair recursively to nested source fields."""
    if isinstance(value, str):
        return repair_mojibake(value)
    if isinstance(value, list):
        return [repair_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: repair_nested(item) for key, item in value.items()}
    return value


def _list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_normalized_row(
    record: dict[str, Any], *, fix_mojibake: bool,
    source_snapshot: str | None = None, source_page: str | None = None,
) -> dict[str, Any]:
    def text(value: Any) -> str | None:
        value = _as_optional_text(value)
        return repair_mojibake(value) if fix_mojibake else value

    raw_name = _as_optional_text(record.get("commercialName"))
    raw_mnn = _as_optional_text(record.get("mnnName"))
    return NormalizedGRLSRecord(
        source_snapshot=source_snapshot, source_page=source_page,
        lp_id=_as_optional_text(record.get("lpId")),
        ru_number=_as_optional_text(record.get("ruNumber")),
        commercial_name_raw=raw_name, commercial_name=text(raw_name),
        mnn_name_raw=raw_mnn,
        mnn_name=None if raw_mnn is None or raw_mnn.strip() == "~" else text(raw_mnn),
        mnn_codes=_list_of_strings(record.get("mnnCode")),
        atx_codes=_list_of_strings(record.get("atx")),
        condition_id=_as_optional_text(record.get("conditionId")),
        release_forms_json=_json_compact(record.get("releaseForms") or []),
    ).model_dump()


def normalized_name(column: str):
    """Vectorized, conservative name cleaning; preserve combination separators."""
    import polars as pl
    return (pl.col(column).str.to_lowercase()
        .str.replace_all(r"\([^)]*\)", " ")
        .str.replace_all(r"\d+(?:[.,]\d+)?\s*(?:мкг|мг|мл|г|%|ме)(?:/\s*(?:мл|г))?", " ")
        .str.replace_all(r"(?i)\b(?:таблетки|таблетка|таблеток|капсулы|капсула|капсул|раствор|инъекций|покрытые|оболочкой|пленочной|мг|мл)\b", " ")
        .str.replace_all(r"[®™]", "")
        .str.replace_all(r"\s+", " ").str.strip_chars(" ,;"))


def find_latest_snapshot(raw_root: Path) -> Path:
    snapshots = sorted(
        (p for p in raw_root.glob("snapshot_*") if p.is_dir()),
        key=lambda p: p.name,
    )
    if not snapshots:
        # Also allow pointing directly at a folder containing page_*.json files.
        if list(raw_root.glob("page_*.json")) or list((raw_root / "pages").glob("page_*.json")):
            return raw_root
        raise FileNotFoundError(
            f"No snapshot folders with page_*.json found under {raw_root}. "
            "Run grls_scraper.py first or pass --input-dir to a snapshot."
        )
    return snapshots[-1]


def normalize_snapshot(
    input_dir: Path,
    output_path: Path,
    *,
    fix_mojibake: bool = False,
    allow_incomplete: bool = False,
    mapped_only: bool = False,
) -> int:
    import polars as pl

    if output_path.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("Output must be outside the raw snapshot")
    manifest: dict[str, Any] = {}
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists() and not allow_incomplete:
        raise RuntimeError("Missing manifest; use --allow-incomplete for diagnostic input")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete", False) and not allow_incomplete:
            raise RuntimeError(
                f"Snapshot is marked incomplete ({manifest.get('termination_reason')}). "
                "Inspect manifest.json; use --allow-incomplete only if you accept this."
            )

    page_files = sorted((input_dir / "pages").glob("page_*.json"))
    if not page_files:
        # Support older/manual layouts.
        page_files = sorted(input_dir.glob("page_*.json"))
    if not page_files:
        raise FileNotFoundError(f"No page_*.json files found in {input_dir}")

    rows_by_id: dict[str, dict[str, Any]] = {}
    rows_without_id: list[dict[str, Any]] = []
    bad_pages: list[str] = []
    input_records = 0
    sparse_pages = 0
    skipped_page_files: list[str] = []
    skipped_records = 0

    for page_file in page_files:
        try:
            payload = json.loads(page_file.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            bad_pages.append(f"{page_file.name}: invalid JSON/encoding: {exc}")
            continue
        records = extract_records(payload)
        if records is None:
            bad_pages.append(f"{page_file.name}: unrecognized response schema")
            continue
        input_records += len(records)
        # Only explicit mapped-only mode permits excluding sparse source pages.
        if records and all(
            not record.get("lpId") and not str(record.get("commercialName") or "").strip()
            for record in records
        ):
            sparse_pages += 1
            if not mapped_only:
                bad_pages.append(f"{page_file.name}: all records lack lpId and commercialName")
                continue
        if mapped_only and records and not any(
            str(record.get("commercialName") or "").strip()
            and str(record.get("mnnName") or "").strip() not in ("", "~")
            and any(code.strip() for code in _list_of_strings(record.get("atx")))
            for record in records
        ):
            skipped_page_files.append(page_file.name)
            skipped_records += len(records)
            continue
        for record in records:
            row = build_normalized_row(
                record, fix_mojibake=fix_mojibake,
                source_snapshot=str(input_dir.resolve()), source_page=page_file.name,
            )
            lp_id = row.get("lp_id")
            if lp_id:
                # Keep the first copy; raw pages preserve all originals for audit.
                rows_by_id.setdefault(str(lp_id), row)
            else:
                rows_without_id.append(row)

    if bad_pages:
        raise RuntimeError("Cannot normalize snapshot safely:\n" + "\n".join(bad_pages))

    rows = list(rows_by_id.values()) + rows_without_id
    if not rows:
        raise RuntimeError("No usable records remain in the snapshot; output was not written")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = {name: pl.String for name in NormalizedGRLSRecord.model_fields}
    schema.update(mnn_codes=pl.List(pl.String), atx_codes=pl.List(pl.String))
    df = pl.DataFrame(rows, schema_overrides=schema)
    provenance_path = output_path.with_name(output_path.stem + "_records.parquet")
    # No independent explosion of INN and ATX: that would invent pairwise relations.
    candidates = df.with_columns(
        normalized_name("commercial_name").alias("trade_name_norm"),
        normalized_name("mnn_name").alias("inn_norm"),
        pl.col("atx_codes").list.eval(pl.element().str.strip_chars())
          .list.eval(pl.element().filter(pl.element() != "")).alias("atx_code"),
    )
    has_trade = pl.col("trade_name_norm").is_not_null() & (pl.col("trade_name_norm") != "")
    has_inn = pl.col("inn_norm").is_not_null() & (pl.col("inn_norm") != "")
    has_atx = pl.col("atx_code").list.len() > 0
    eligible = candidates.filter(has_trade & has_inn & has_atx if mapped_only else has_trade)
    dictionary = (eligible.select("trade_name_norm", "inn_norm", "atx_code")
      .unique().sort(["trade_name_norm", "inn_norm", "atx_code"]))
    report = {
        "source_snapshot": str(input_dir.resolve()),
        "source_complete": bool(manifest.get("complete", False)),
        "coverage_verified": bool(manifest.get("coverage_verified", False)),
        "coverage_scope": manifest.get("coverage_scope"),
        "registry_complete": bool(manifest.get("registry_complete", False)),
        "source_termination_reason": manifest.get("termination_reason"),
        "source_condition_ids_filter": manifest.get("condition_ids_filter", []),
        "mapped_only": mapped_only,
        "input_records": input_records,
        "records_after_lp_id_dedup": df.height,
        "sparse_pages": sparse_pages,
        "skipped_pages": len(skipped_page_files),
        "skipped_page_files": skipped_page_files,
        "skipped_page_records": skipped_records,
        "duplicate_lp_id_records": input_records - skipped_records - df.height,
        # Missing-field counts overlap and are measured after lpId deduplication.
        "missing_trade_name": candidates.filter(~has_trade).height,
        "missing_inn": candidates.filter(~has_inn).height,
        "missing_atx": candidates.filter(~has_atx).height,
        "eligible_records_before_dictionary_dedup": eligible.height,
        "excluded_records": df.height - eligible.height,
        "total_excluded_records": skipped_records + df.height - eligible.height,
        "dictionary_rows": dictionary.height,
    }
    for frame, destination in ((df, provenance_path), (dictionary, output_path)):
        temporary = destination.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(destination)

    report_path = output_path.with_name(output_path.stem + "_report.json")
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_report.replace(report_path)
    LOG.info("Selection report: %s", report_path)
    LOG.info("Read %s page files", len(page_files))
    LOG.info("Normalized %s unique records to %s", len(rows), output_path)
    LOG.info("Rows with ATX: %s", df.filter(pl.col("atx_codes").list.len() > 0).height)
    LOG.info("Rows with MNN: %s", df.filter(
        pl.col("mnn_name").is_not_null() & (pl.col("mnn_name") != "~")
    ).height)
    return dictionary.height


def merge_snapshots(
    input_dirs: Sequence[Path], output_path: Path, *,
    fix_mojibake: bool = False, allow_incomplete: bool = False,
    mapped_only: bool = False,
) -> int:
    """Union mappings without losing older links or their source evidence.

    Deduplication by lpId happens within each snapshot; distinct observations of
    the same lpId in different snapshots remain separate provenance records.
    All inputs must validate before any existing output is replaced.
    """
    import polars as pl

    sources = list(dict.fromkeys(path.resolve() for path in input_dirs))
    if not sources:
        raise ValueError("At least one snapshot is required")
    if any(output_path.resolve().is_relative_to(source) for source in sources):
        raise ValueError("Output must be outside every raw snapshot")
    dictionaries = []
    provenance = []
    reports = []
    with tempfile.TemporaryDirectory(prefix="grls-merge-") as folder:
        for index, source in enumerate(sources):
            staged = Path(folder) / f"source_{index}.parquet"
            normalize_snapshot(source, staged, fix_mojibake=fix_mojibake,
                               allow_incomplete=allow_incomplete, mapped_only=mapped_only)
            dictionaries.append(pl.read_parquet(staged))
            provenance.append(pl.read_parquet(staged.with_name(staged.stem + "_records.parquet")))
            reports.append(json.loads(staged.with_name(staged.stem + "_report.json").read_text()))

    dictionary = (pl.concat(dictionaries).unique()
                  .sort(["trade_name_norm", "inn_norm", "atx_code"]))
    records = pl.concat(provenance)
    report = {
        "mode": "merge_snapshots",
        "source_complete": all(item["source_complete"] for item in reports),
        "mapped_only": mapped_only,
        "sources": reports,
        "input_records": sum(item["input_records"] for item in reports),
        "skipped_pages": sum(item["skipped_pages"] for item in reports),
        "skipped_page_records": sum(item["skipped_page_records"] for item in reports),
        "provenance_rows": records.height,
        "dictionary_rows_before_merge": dictionaries[0].height,
        "dictionary_rows": dictionary.height,
        "new_links_added": dictionary.height - dictionaries[0].height,
        "duplicate_dictionary_rows_removed": sum(frame.height for frame in dictionaries) - dictionary.height,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = output_path.with_name(output_path.stem + "_records.parquet")
    report_path = output_path.with_name(output_path.stem + "_report.json")
    pending = []
    for frame, destination in ((dictionary, output_path), (records, provenance_path)):
        staged = destination.with_suffix(".parquet.tmp")
        frame.write_parquet(staged, compression="zstd")
        pending.append((staged, destination))
    staged_report = report_path.with_suffix(".json.tmp")
    staged_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.append((staged_report, report_path))
    for staged, destination in pending:
        staged.replace(destination)
    LOG.info("Merged %s snapshots: %s links (%s new)",
             len(sources), dictionary.height, report["new_links_added"])
    return dictionary.height


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a raw GRLS API snapshot to Parquet.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/grls"))
    parser.add_argument(
        "--additional-input-dir", type=Path, action="append", default=[],
        help="Additional snapshot to merge; repeatable. Original links and source evidence are kept.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/grls_normalized.parquet"),
    )
    parser.add_argument(
        "--repair-mojibake", action="store_true",
        help="Attempt conservative CP1251->UTF-8 repair in selected text fields. "
             "Raw page responses remain unchanged and *_raw columns are preserved.",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Allow normalization of a snapshot whose manifest is marked incomplete.",
    )
    parser.add_argument(
        "--mapped-only", action="store_true",
        help="Build a partial dictionary only from nonempty trade name, INN and ATX. "
             "Skip pages without a complete source mapping; preserve raw responses.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    input_dir = find_latest_snapshot(args.input_dir)
    if args.additional_input_dir:
        count = merge_snapshots(
            [input_dir, *(find_latest_snapshot(path) for path in args.additional_input_dir)],
            args.output, fix_mojibake=args.repair_mojibake,
            allow_incomplete=args.allow_incomplete, mapped_only=args.mapped_only,
        )
    else:
        count = normalize_snapshot(
            input_dir, args.output, fix_mojibake=args.repair_mojibake,
            allow_incomplete=args.allow_incomplete, mapped_only=args.mapped_only,
        )
    print(f"Normalized {count} records to {args.output.resolve()}")


if __name__ == "__main__":
    main()
