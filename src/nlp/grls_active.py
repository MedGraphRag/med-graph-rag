"""Collect and verify the full public GRLS query for statuses Д/П.

Date partitions avoid the observed degradation of large API result sets.
Completeness applies only to this query, never to the entire registry.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx

from .grls_scraper import API_URL, _get_with_retries, _write_manifest, download_registry, extract_records

LOG = logging.getLogger("grls_active")
ACTIVE_STATUSES = ("Д", "П")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


async def download_active_registry(
    *, output_root: Path, delay_seconds: float = 1,
    split_dates: Sequence[str] = ("2024-01-01", "2025-01-01"),
) -> Path:
    """Publish a snapshot only when unique IDs match stable API totals.

    Date boundaries are validated through the final unique-ID count. If any
    partition still hits the server defect, fail without publishing a complete
    snapshot; the raw partition remains available for diagnosis.
    """
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be nonnegative")
    dates = list(split_dates)
    for value in dates:
        datetime.strptime(value, "%Y-%m-%d")
    if dates != sorted(set(dates)):
        raise ValueError("split_dates must be unique and increasing")
    collection = output_root / ("collection_active_" + _timestamp())
    (collection / "controls").mkdir(parents=True, exist_ok=False)
    report = {
        "source": API_URL, "condition_ids_filter": list(ACTIVE_STATUSES),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False, "registry_complete": False,
        "termination_reason": "running", "partitions": [], "errors": [],
    }
    _write_manifest(collection / "manifest.json", report)

    async def control(label: str) -> tuple[int, set[str]]:
        params = [("page", "1"), ("size", "1"),
                  *(("conditionIds", status) for status in ACTIVE_STATUSES)]
        async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=15),
                                     headers={"Accept": "application/json"}) as client:
            response = await _get_with_retries(client, params)
        path = collection / "controls" / (label + ".json")
        path.write_bytes(response.content)
        total = response.headers.get("x-total-count")
        if total is None or not total.isdigit():
            raise RuntimeError("API control response has no valid x-total-count")
        rows = extract_records(response.json())
        if rows is None:
            raise RuntimeError("Unrecognized control response schema")
        _write_manifest(path.with_name(label + "_metadata.json"), {
            "params": params, "x_total_count": int(total),
            "sha256": hashlib.sha256(response.content).hexdigest(),
        })
        return int(total), {str(row["lpId"]) for row in rows if row.get("lpId")}

    try:
        expected, control_ids_before = await control("before")
        report["api_total_count_initial"] = expected
        _write_manifest(collection / "manifest.json", report)
        bounds = [None, *dates, None]
        for start, end in zip(bounds, bounds[1:]):
            await asyncio.sleep(delay_seconds)
            source = await download_registry(
                output_root=collection / "parts", condition_ids=ACTIVE_STATUSES,
                delay_seconds=delay_seconds, entry_date_start=start, entry_date_end=end,
            )
            manifest = json.loads((source / "manifest.json").read_text())
            report["partitions"].append(str(source.resolve()))
            _write_manifest(collection / "manifest.json", report)
            if not manifest.get("query_complete") or manifest.get("errors"):
                raise RuntimeError(
                    f"Partition {start}..{end} is incomplete: {manifest['termination_reason']}; "
                    "use smaller date partitions"
                )

        seen_ids: set[str] = set()
        statuses: Counter[str] = Counter()
        page_origins = []
        atx_count = mnn_count = received = 0
        declared_partition_total = 0
        for source_path in report["partitions"]:
            source = Path(source_path)
            manifest = json.loads((source / "manifest.json").read_text())
            declared_partition_total += manifest["api_total_count_initial"]
            for page in sorted((source / "pages").glob("page_*.json")):
                rows = extract_records(json.loads(page.read_bytes()))
                if rows is None:
                    raise RuntimeError(f"Invalid raw page: {page}")
                received += len(rows)
                for row in rows:
                    lp_id = row.get("lpId")
                    if not lp_id or row.get("conditionId") not in ACTIVE_STATUSES:
                        raise RuntimeError(f"Missing identity or unexpected status in {page}")
                    identity = str(lp_id)
                    if identity in seen_ids:
                        continue
                    seen_ids.add(identity)
                    statuses[row["conditionId"]] += 1
                    atx_count += bool(row.get("atx"))
                    mnn_count += bool(row.get("mnnName") and row["mnnName"].strip() != "~")
                page_origins.append({
                    "source_snapshot": str(source.resolve()), "source_page": page.name,
                    "entry_date_start": manifest["entry_date_start"],
                    "entry_date_end": manifest["entry_date_end"],
                })
        await asyncio.sleep(delay_seconds)
        final_total, control_ids_after = await control("after")
        report.update(api_total_count_final=final_total, records_saved_unique=len(seen_ids),
                      declared_partition_total=declared_partition_total)
        if not (expected == final_total == len(seen_ids) == declared_partition_total):
            raise RuntimeError(
                f"Coverage mismatch: before={expected}, after={final_total}, "
                f"unique lpId={len(seen_ids)}, partition totals={declared_partition_total}"
            )
        if not (control_ids_before | control_ids_after) <= seen_ids:
            raise RuntimeError("Control records are missing from downloaded partitions")

        # Copy raw bodies byte-for-byte into a normal snapshot consumable by ETL.
        snapshot = output_root / ("snapshot_" + _timestamp())
        (snapshot / "pages").mkdir(parents=True, exist_ok=False)
        for index, origin in enumerate(page_origins, 1):
            original = Path(origin["source_snapshot"]) / "pages" / origin["source_page"]
            page_name = f"page_{index:06d}.json"
            destination = snapshot / "pages" / page_name
            shutil.copyfile(original, destination)
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Raw copy checksum mismatch")
            origin.update(page=page_name, sha256=digest)
        manifest = {
            "source": API_URL, "condition_ids_filter": list(ACTIVE_STATUSES),
            "started_at_utc": report["started_at_utc"],
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": True, "coverage_verified": True, "query_complete": True,
            "registry_complete": False, "coverage_scope": "conditionIds=Д,П",
            "pagination_complete": True, "page_size_requested": 100,
            "termination_reason": "verified_date_partitions",
            "api_total_count_initial": expected, "api_total_count_final": final_total,
            "pages_fetched": len(page_origins), "records_saved_unique": len(seen_ids),
            "records_received_including_duplicates": received,
            "condition_id_counts": dict(statuses), "records_with_atx": atx_count,
            "records_with_mnn": mnn_count, "cookies_or_auth_used": False,
            "source_collection": str(collection.resolve()), "page_origins": page_origins,
            "errors": [],
        }
        _write_manifest(snapshot / "manifest.json", manifest)
        report.update(complete=True, coverage_scope="conditionIds=Д,П",
                      termination_reason="verified_date_partitions", snapshot=str(snapshot.resolve()))
        LOG.info("Verified all %s Д/П records in %s", len(seen_ids), snapshot)
        return snapshot
    except (KeyboardInterrupt, asyncio.CancelledError):
        report["termination_reason"] = "interrupted"
        raise
    except Exception as exc:
        report["termination_reason"] = "error"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(collection / "manifest.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify all public GRLS records with statuses Д/П")
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/grls"))
    parser.add_argument("--delay", type=float, default=1)
    parser.add_argument("--split-date", action="append", help="Repeatable date boundaries, YYYY-MM-DD; default: 2024-01-01 and 2025-01-01")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    options = {"split_dates": args.split_date} if args.split_date is not None else {}
    snapshot = asyncio.run(download_active_registry(output_root=args.output_root, delay_seconds=args.delay, **options))
    print(f"Verified Д/П snapshot: {snapshot.resolve()}")


if __name__ == "__main__":
    main()
