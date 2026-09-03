from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, Mapping
import json

import pandas as pd

from release_contract import SCANNER_RELEASE_VERSION

from persistence import DatabaseConfig, _request, database_status
from data_providers import UNIVERSE_METADATA_COLUMNS, normalize_ticker
from free_tier_storage import prune_scan_history_best_effort

JOB_VERSION = SCANNER_RELEASE_VERSION
ACTIVE_JOB_STATUSES = ("CREATED", "RUNNING", "PAUSED", "FINALIZE_RETRY_REQUIRED")
TERMINAL_JOB_STATUSES = ("COMPLETED", "COMPLETED_PARTIAL_PERSISTENCE", "CANCELLED", "FAILED")
STAGE_ORDER = (
    "BENCHMARK",
    "OHLCV",
    "FAST_RANKING",
    "KSEI_SHORTLIST",
    "NEWS_SHORTLIST",
    "FUNDAMENTAL_SHORTLIST",
    "IDX_FUNDAMENTAL_SHORTLIST",
    "FINALIZE",
    "COMPLETED",
)


def _now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def normalized_universe_records(universe: pd.DataFrame | Iterable[str]) -> list[dict[str, str]]:
    if isinstance(universe, pd.DataFrame):
        local = universe.copy()
        local.columns = [str(column).strip().lower() for column in local.columns]
        if "ticker" not in local.columns:
            return []
        for column in UNIVERSE_METADATA_COLUMNS:
            if column not in local.columns:
                local[column] = ""
        records = local[list(UNIVERSE_METADATA_COLUMNS)].to_dict(orient="records")
    else:
        records = [{"ticker": ticker} for ticker in universe]

    def clean(value: Any) -> str:
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value or "").strip()

    output: list[dict[str, str]] = []
    positions: dict[str, int] = {}
    for record in records:
        ticker = normalize_ticker(record.get("ticker"))
        if not ticker:
            continue
        normalized = {
            column: ticker if column == "ticker" else clean(record.get(column))
            for column in UNIVERSE_METADATA_COLUMNS
        }
        if ticker in positions:
            existing = output[positions[ticker]]
            for column in UNIVERSE_METADATA_COLUMNS:
                if column != "ticker" and not existing[column] and normalized[column]:
                    existing[column] = normalized[column]
            continue
        positions[ticker] = len(output)
        output.append(normalized)
    return output


def universe_hash(universe: pd.DataFrame | Iterable[str] | list[dict[str, Any]]) -> str:
    if isinstance(universe, list) and (not universe or isinstance(universe[0], Mapping)):
        records = normalized_universe_records(pd.DataFrame(universe))
    elif isinstance(universe, pd.DataFrame):
        records = normalized_universe_records(universe)
    else:
        records = normalized_universe_records(universe)
    canonical = json.dumps(
        sorted(records, key=lambda record: record["ticker"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def stage_progress(stage: str, offset: int, stage_total: int) -> float:
    """Weighted progress. It is monotonic across resumable stages."""
    weights = {
        "BENCHMARK": (0.0, 2.0),
        "OHLCV": (2.0, 57.0),
        "FAST_RANKING": (57.0, 62.0),
        "KSEI_SHORTLIST": (62.0, 74.0),
        "NEWS_SHORTLIST": (74.0, 84.0),
        "FUNDAMENTAL_SHORTLIST": (84.0, 91.0),
        "IDX_FUNDAMENTAL_SHORTLIST": (91.0, 97.0),
        "FINALIZE": (97.0, 100.0),
        "COMPLETED": (100.0, 100.0),
    }
    start, end = weights.get(str(stage), (0.0, 0.0))
    if stage == "COMPLETED":
        return 100.0
    fraction = min(1.0, max(0.0, float(offset) / max(int(stage_total), 1)))
    return round(start + (end - start) * fraction, 1)


def next_chunk(items: list[str], offset: int, chunk_size: int) -> tuple[list[str], int, bool]:
    start = max(0, int(offset))
    size = max(1, int(chunk_size))
    chunk = items[start:start + size]
    next_offset = start + len(chunk)
    return chunk, next_offset, next_offset >= len(items)



def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return value if pd.notna(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value if isinstance(value, (str, int, bool, float)) or value is None else str(value)

def _job_from_row(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    job = dict(row)
    for key, default in (("universe", []), ("settings", {}), ("shortlist", []), ("failures", {}), ("result_summary", {})):
        value = job.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                value = default
        job[key] = value if isinstance(value, type(default)) else default
    return job


def create_scan_job(
    config: DatabaseConfig,
    *,
    scan_id: str,
    universe: pd.DataFrame | list[dict[str, Any]],
    settings: Mapping[str, Any],
    chunk_size: int = 20,
) -> dict[str, Any]:
    records = normalized_universe_records(universe) if isinstance(universe, pd.DataFrame) else normalized_universe_records(pd.DataFrame(universe))
    # Remove old terminal jobs/chunks before allocating another resumable job.
    prune_scan_history_best_effort(config, keep_scan_runs=2, keep_terminal_jobs=2, exclude_scan_id=str(scan_id))
    now = _now_iso()
    universe_hash_value = universe_hash(records)
    payload = [{
        "scan_id": str(scan_id),
        "universe_hash": universe_hash_value,
        "scanner_version": JOB_VERSION,
        "status": "CREATED",
        "current_stage": "BENCHMARK",
        "current_offset": 0,
        "current_chunk": 0,
        "chunk_size": max(5, min(int(chunk_size), 50)),
        "total_tickers": len(records),
        "processed_tickers": 0,
        "failed_tickers": 0,
        "progress_pct": 0.0,
        "scan_mode": str(settings.get("scan_mode") or "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP"),
        "result_status": "PENDING",
        "universe": _json_safe(records),
        "settings": _json_safe(dict(settings)),
        "shortlist": [],
        "failures": {},
        "result_summary": {},
        "last_error": "",
        "heartbeat_at": now,
        "updated_at": now,
    }]
    if not config.ready:
        return _job_from_row(payload[0]) or payload[0]
    try:
        response = _request(
            config,
            "POST",
            "cak_scan_jobs",
            params={"on_conflict": "scan_id"},
            payload=payload,
        )
    except Exception:
        # The database enforces one active job per universe + scanner version.
        # If two Streamlit sessions race, the loser reuses the winner rather
        # than leaving a duplicate orphan or surfacing a transient 409.
        existing = find_latest_job(
            config,
            universe_hash_value=universe_hash_value,
            include_completed=False,
        )
        if existing and str(existing.get("scanner_version") or "") == JOB_VERSION:
            return existing
        raise
    rows = response.json()
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("cak_scan_jobs create did not return exactly one row.")
    return _job_from_row(rows[0]) or rows[0]


def get_scan_job(config: DatabaseConfig, scan_id: str) -> dict[str, Any] | None:
    if not config.ready or not scan_id:
        return None
    response = _request(
        config,
        "GET",
        "cak_scan_jobs",
        params={"select": "*", "scan_id": f"eq.{scan_id}", "limit": "1"},
        timeout=10,
    )
    rows = response.json()
    return _job_from_row(rows[0]) if isinstance(rows, list) and rows else None


def find_unique_active_job(
    config: DatabaseConfig,
    *,
    scanner_version: str = JOB_VERSION,
) -> dict[str, Any] | None:
    """Recover a current-version active job when there is exactly one candidate.

    This is intentionally universe-agnostic so a fresh Streamlit process can recover
    before the file uploader exists. If multiple active universes exist, return None
    rather than silently attaching the UI to the wrong job.
    """
    if not config.ready:
        return None
    status_filter = "in.(" + ",".join(ACTIVE_JOB_STATUSES) + ")"
    response = _request(
        config,
        "GET",
        "cak_scan_jobs",
        params={
            "select": "*",
            "scanner_version": f"eq.{scanner_version}",
            "status": status_filter,
            "order": "updated_at.desc",
            "limit": "2",
        },
        timeout=10,
    )
    rows = response.json()
    if not isinstance(rows, list) or len(rows) != 1:
        return None
    return _job_from_row(rows[0])


def find_latest_job(
    config: DatabaseConfig,
    *,
    universe_hash_value: str,
    include_completed: bool = False,
) -> dict[str, Any] | None:
    if not config.ready or not universe_hash_value:
        return None
    statuses = (*ACTIVE_JOB_STATUSES, *TERMINAL_JOB_STATUSES) if include_completed else ACTIVE_JOB_STATUSES
    status_filter = "in.(" + ",".join(statuses) + ")"
    response = _request(
        config,
        "GET",
        "cak_scan_jobs",
        params={
            "select": "*",
            "universe_hash": f"eq.{universe_hash_value}",
            "status": status_filter,
            "order": "updated_at.desc",
            "limit": "1",
        },
        timeout=10,
    )
    rows = response.json()
    return _job_from_row(rows[0]) if isinstance(rows, list) and rows else None


def update_scan_job(config: DatabaseConfig, scan_id: str, patch: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(patch)
    payload["updated_at"] = _now_iso()
    payload["heartbeat_at"] = payload.get("heartbeat_at") or payload["updated_at"]
    if not config.ready:
        return {"scan_id": scan_id, **payload}
    response = _request(
        config,
        "PATCH",
        "cak_scan_jobs",
        params={"scan_id": f"eq.{scan_id}"},
        payload=payload,
    )
    rows = response.json()
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("cak_scan_jobs update did not return exactly one row.")
    return _job_from_row(rows[0]) or rows[0]


def update_scan_job_minimal(
    config: DatabaseConfig,
    scan_id: str,
    patch: Mapping[str, Any],
    *,
    base_job: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Commit a job transition without asking PostgREST to return the JSON-heavy row.

    FINALIZE already has the complete job in memory. Requesting the full row again can
    exceed the hosted statement timeout while the database is flushing the result and
    evidence writes. A minimal PATCH makes the terminal transition small and atomic;
    the locally merged row remains sufficient for the current Streamlit render.
    """
    payload = dict(patch)
    payload["updated_at"] = _now_iso()
    payload["heartbeat_at"] = payload.get("heartbeat_at") or payload["updated_at"]
    merged = {**dict(base_job or {}), "scan_id": scan_id, **payload}
    if not config.ready:
        return merged
    _request(
        config,
        "PATCH",
        "cak_scan_jobs",
        params={"scan_id": f"eq.{scan_id}"},
        payload=payload,
        return_rows=False,
        timeout=20,
    )
    return _job_from_row(merged) or merged


def record_job_chunk(
    config: DatabaseConfig,
    *,
    scan_id: str,
    stage: str,
    chunk_no: int,
    tickers: list[str],
    processed_count: int,
    failed_count: int,
    status: str,
    payload: Mapping[str, Any] | None = None,
    started_at: Any = None,
) -> dict[str, Any]:
    now = _now_iso()
    row = {
        "chunk_id": f"{scan_id}:{stage}:{int(chunk_no)}",
        "scan_id": scan_id,
        "stage": stage,
        "chunk_no": int(chunk_no),
        "ticker_count": len(tickers),
        "processed_count": int(processed_count),
        "failed_count": int(failed_count),
        "status": status,
        "started_at": pd.Timestamp(started_at).isoformat() if started_at is not None else now,
        "completed_at": now,
        "payload": dict(payload or {}),
    }
    if not config.ready:
        return row
    response = _request(
        config,
        "POST",
        "cak_scan_job_chunks",
        params={"on_conflict": "chunk_id"},
        payload=[row],
    )
    rows = response.json()
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("cak_scan_job_chunks upsert did not return exactly one row.")
    return dict(rows[0])


def load_job_chunks(config: DatabaseConfig, scan_id: str) -> pd.DataFrame:
    if not config.ready or not scan_id:
        return pd.DataFrame()
    response = _request(
        config,
        "GET",
        "cak_scan_job_chunks",
        params={"select": "*", "scan_id": f"eq.{scan_id}", "order": "completed_at.asc"},
        timeout=15,
    )
    rows = response.json()
    return pd.DataFrame(rows if isinstance(rows, list) else [])


def cancel_scan_job(config: DatabaseConfig, scan_id: str) -> dict[str, Any]:
    return update_scan_job(
        config,
        scan_id,
        {
            "status": "CANCELLED",
            "result_status": "CANCELLED",
            "last_error": "Cancelled by user.",
        },
    )


def job_status_frame(job: Mapping[str, Any] | None) -> pd.DataFrame:
    if not job:
        return pd.DataFrame()
    fields = [
        "scan_id", "scanner_version", "status", "current_stage", "current_offset", "current_chunk",
        "chunk_size", "total_tickers", "processed_tickers", "failed_tickers", "progress_pct",
        "scan_mode", "result_status", "heartbeat_at", "updated_at", "last_error",
    ]
    return pd.DataFrame([{key: job.get(key) for key in fields}])


__all__ = [
    "JOB_VERSION", "ACTIVE_JOB_STATUSES", "TERMINAL_JOB_STATUSES", "STAGE_ORDER",
    "normalized_universe_records", "universe_hash", "stage_progress", "next_chunk",
    "create_scan_job", "get_scan_job", "find_latest_job", "find_unique_active_job", "update_scan_job",
    "record_job_chunk", "load_job_chunks", "cancel_scan_job", "job_status_frame",
]
