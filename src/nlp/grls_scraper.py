"""Download raw pages from the public GRLS registry API.

The scraper intentionally sends no cookies, authorization headers, or browser
session state. Raw HTTP response bodies are saved unchanged for auditability.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx

API_URL = (
    "https://rlp.minzdrav.gov.ru/"
    "medicine/portal/api/v0.1/medicament-registry"
)
LOG = logging.getLogger("grls_scraper")
_RECORD_KEYS = {"lpId", "commercialName", "dossierId", "ruNumber"}
_CONTAINER_KEYS = (
    "content", "items", "results", "records", "data", "rows", "medicaments"
)


def extract_records(payload: Any) -> list[dict[str, Any]] | None:
    """Find a collection of GRLS records in common JSON response envelopes.

    Returns None for an unrecognized schema, [] for a recognized empty page.
    The live endpoint has been observed to return a list of objects, but the
    envelope handling makes a schema change fail more visibly and safely.
    """
    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(item, dict) for item in payload) and any(
            _RECORD_KEYS.intersection(item.keys()) for item in payload
        ):
            return payload  # preserve entire object, including unknown fields
        return None

    if not isinstance(payload, dict):
        return None

    for key in _CONTAINER_KEYS:
        if key in payload and isinstance(payload[key], list):
            value = payload[key]
            if not value:
                return []
            if all(isinstance(item, dict) for item in value) and any(
                _RECORD_KEYS.intersection(item.keys()) for item in value
            ):
                return value

    # Some APIs wrap the actual envelope in one or more nested objects.
    for value in payload.values():
        if isinstance(value, (dict, list)):
            found = extract_records(value)
            if found is not None:
                return found
    return None


def _valid_utf8(body: bytes) -> bool:
    try:
        body.decode("utf-8-sig")
        return True
    except UnicodeDecodeError:
        return False


def is_unusable_page(records: list[dict[str, Any]]) -> bool:
    """Detect the observed API degradation without treating it as end-of-data.

    Sparse registration objects with neither identity nor trade name cannot
    support the lookup dictionary. Keep their raw response and stop for audit.
    """
    return bool(records) and all(
        not record.get("lpId") and not str(record.get("commercialName") or "").strip()
        for record in records
    )


def _record_fingerprint(record: dict[str, Any]) -> str:
    """Stable identity fallback for records without lpId."""
    encoded = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _page_signature(records: list[dict[str, Any]]) -> tuple[str, ...]:
    """Detect an API that silently repeats a page at the end of pagination."""
    return tuple(sorted(
        str(record.get("lpId") or _record_fingerprint(record))
        for record in records
    ))


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


async def _get_with_retries(
    client: httpx.AsyncClient,
    params: list[tuple[str, str]],
    *,
    attempts: int = 5,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.get(API_URL, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < attempts:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = max(0.0, float(retry_after)) if retry_after else 0.75 * (2**attempt)
                    except ValueError:
                        delay = 0.75 * (2**attempt)
                    await asyncio.sleep(min(delay, 30.0))
                    continue
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(min(0.75 * (2**attempt), 30.0))
    if last_error is not None:
        raise last_error
    raise RuntimeError("HTTP request failed without an exception")


async def download_registry(
    *,
    output_root: Path,
    page_size: int = 100,
    condition_ids: Sequence[str] = (),
    delay_seconds: float = 0.25,
    max_pages: int = 100_000,
    entry_date_start: str | None = None,
    entry_date_end: str | None = None,
) -> Path:
    """Download pages into a timestamped raw snapshot directory.

    By default no conditionIds filter is sent. This is intentional: whether an
    unfiltered endpoint includes every registry status must be validated from
    the resulting manifest and compared with the UI's visible total.
    """
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be nonnegative")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")

    for value in (entry_date_start, entry_date_end):
        if value is not None:
            datetime.strptime(value, "%Y-%m-%d")
    if entry_date_start and entry_date_end and entry_date_start >= entry_date_end:
        raise ValueError("entry_date_start must be before entry_date_end")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_dir = output_root / f"snapshot_{timestamp}"
    pages_dir = snapshot_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "source": API_URL,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "page_size_requested": page_size,
        "condition_ids_filter": list(condition_ids),
        "entry_date_start": entry_date_start,
        "entry_date_end": entry_date_end,
        "query_complete": False,
        "cookies_or_auth_used": False,
        "pages_fetched": 0,
        "records_received_including_duplicates": 0,
        "records_saved_unique": 0,
        "records_with_atx": 0,
        "records_with_mnn": 0,
        "condition_id_counts": {},
        "termination_reason": "running",
        "api_total_count_initial": None,
        "quality_issue": None,
        "complete": False,
        "pagination_complete": False,
        "coverage_verified": False,
        "response_encodings": [],
        "errors": [],
    }
    seen_records: set[str] = set()
    seen_page_signatures: set[tuple[str, ...]] = set()
    status_counts: Counter[str] = Counter()
    atx_count = 0
    mnn_count = 0
    unique_count = 0
    last_page = 0

    def persist_progress() -> None:
        manifest["records_saved_unique"] = unique_count
        manifest["records_with_atx"] = atx_count
        manifest["records_with_mnn"] = mnn_count
        manifest["condition_id_counts"] = dict(status_counts)
        _write_manifest(snapshot_dir / "manifest.json", manifest)

    persist_progress()
    timeout = httpx.Timeout(45.0, connect=15.0)
    headers = {
        "Accept": "application/json",
        "User-Agent": "MedGraphRAG-GRLS-research-export/1.0",
    }

    try:
        async with httpx.AsyncClient(
            timeout=timeout, headers=headers, follow_redirects=True
        ) as client:
            for page in range(1, max_pages + 1):
                params = [("page", str(page)), ("size", str(page_size))]
                params.extend(("conditionIds", value) for value in condition_ids)
                if entry_date_start:
                    params.append(("entryDateStart", entry_date_start + "T00:00:00.000Z"))
                if entry_date_end:
                    params.append(("entryDateEnd", entry_date_end + "T00:00:00.000Z"))
                manifest["last_page_requested"] = page
                response = await _get_with_retries(client, params)
                manifest["pages_fetched"] = page
                total_header = response.headers.get("x-total-count")
                api_total = int(total_header) if total_header and total_header.isdigit() else None
                if manifest["api_total_count_initial"] is None:
                    manifest["api_total_count_initial"] = api_total
                manifest["response_encodings"].append({
                    "api_total_count": api_total,
                    "page": page, "content_type": response.headers.get("content-type"),
                    "httpx_encoding": response.encoding,
                    "valid_utf8": _valid_utf8(response.content),
                })

                # Keep exact response bytes before parsing or normalizing text.
                raw_path = pages_dir / f"page_{page:06d}.json"
                raw_path.write_bytes(response.content)
                try:
                    payload = response.json()
                except ValueError as exc:
                    sample_path = snapshot_dir / f"invalid_json_page_{page:06d}.txt"
                    sample_path.write_bytes(response.content[:20_000])
                    raise RuntimeError(
                        f"API returned invalid JSON on page {page}; "
                        f"sample saved to {sample_path}"
                    ) from exc

                records = extract_records(payload)
                if records is None:
                    sample_path = snapshot_dir / f"unknown_schema_page_{page:06d}.json"
                    sample_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    raise RuntimeError(
                        f"Unrecognized response schema on page {page}; "
                        f"sample saved to {sample_path}"
                    )

                manifest["pages_fetched"] = page
                last_page = page
                manifest["records_received_including_duplicates"] += len(records)

                if not records:
                    manifest["termination_reason"] = "empty_page"
                    manifest["pagination_complete"] = True
                    manifest["query_complete"] = api_total is None or unique_count == api_total
                    LOG.info("API returned an empty page at page=%s", page)
                    break

                initial_total = manifest["api_total_count_initial"]
                if initial_total is not None and api_total != initial_total:
                    manifest["termination_reason"] = "total_count_changed"
                    manifest["errors"].append("API x-total-count changed during pagination")
                    break

                if is_unusable_page(records):
                    manifest["termination_reason"] = "unusable_page"
                    manifest["quality_issue"] = {
                        "page": page,
                        "records": len(records),
                        "reason": "All records lack both lpId and commercialName",
                    }
                    LOG.error(
                        "page=%s contains no lpId or trade names; raw saved, "
                        "stopping incomplete snapshot for audit", page,
                    )
                    break

                signature = _page_signature(records)
                if signature in seen_page_signatures:
                    manifest["termination_reason"] = "repeated_page"
                    manifest["complete"] = False
                    LOG.warning(
                        "API repeated an earlier page at page=%s; "
                        "snapshot is marked incomplete", page
                    )
                    break
                seen_page_signatures.add(signature)

                page_new = 0
                for record in records:
                    record_id = record.get("lpId")
                    identity = f"lpId:{record_id}" if record_id else (
                        "sha256:" + _record_fingerprint(record)
                    )
                    if identity in seen_records:
                        continue
                    seen_records.add(identity)
                    page_new += 1
                    unique_count += 1
                    if record.get("atx"):
                        atx_count += 1
                    if record.get("mnnName") and record.get("mnnName") != "~":
                        mnn_count += 1
                    condition_id = record.get("conditionId")
                    if condition_id is not None:
                        status_counts[str(condition_id)] += 1

                LOG.info(
                    "page=%s rows=%s new_unique=%s cumulative_unique=%s",
                    page, len(records), page_new, unique_count,
                )

                if page_new == 0:
                    manifest["termination_reason"] = "no_new_records"
                    manifest["complete"] = False
                    LOG.warning(
                        "No new IDs on page=%s; stopping, completeness is unknown",
                        page,
                    )
                    break

                if initial_total is not None and unique_count >= initial_total:
                    manifest["query_complete"] = unique_count == initial_total
                    manifest["pagination_complete"] = manifest["query_complete"]
                    manifest["termination_reason"] = (
                        "reported_total_reached" if manifest["query_complete"] else "unexpected_record_count"
                    )
                    break

                persist_progress()
                if page < max_pages and delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
            else:
                manifest["termination_reason"] = "max_pages_reached"
                manifest["complete"] = False

    except (KeyboardInterrupt, asyncio.CancelledError):
        manifest["termination_reason"] = "interrupted"
        manifest["complete"] = False
        raise
    except Exception as exc:
        manifest["termination_reason"] = "error"
        manifest["complete"] = False
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.setdefault("last_page_requested", last_page)
        persist_progress()

    LOG.info("Snapshot saved to %s", snapshot_dir)
    LOG.info("Completion flag: %s", manifest["complete"])
    return snapshot_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw paginated records from the public GRLS API."
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/grls"))
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--condition-id", action="append", default=[],
        help="Optional repeatable status filter; omitted by default to avoid filtering records.",
    )
    parser.add_argument("--entry-date-start", help="Inclusive registration date, YYYY-MM-DD")
    parser.add_argument("--entry-date-end", help="Upper registration-date boundary, YYYY-MM-DD")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--max-pages", type=int, default=100_000)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    snapshot = asyncio.run(download_registry(
        output_root=args.output_root,
        page_size=args.page_size,
        condition_ids=args.condition_id,
        delay_seconds=args.delay,
        max_pages=args.max_pages,
        entry_date_start=args.entry_date_start,
        entry_date_end=args.entry_date_end,
    ))
    print(f"Raw snapshot: {snapshot.resolve()}")
    print(f"Manifest: {(snapshot / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
