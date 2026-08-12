from __future__ import annotations

"""Free-tier storage retention for the Emir scanner.

This module intentionally performs only best-effort deletion of historical rows.
Current caches are upsert-based and are never removed here. The scanner remains
functional if housekeeping cannot run.
"""

from typing import Any
import os

import pandas as pd

from persistence import DatabaseConfig, _request

FREE_TIER_STORAGE_VERSION = "1.9.15-free-tier-storage"
TERMINAL_JOB_STATUSES = {
    "COMPLETED", "COMPLETED_PARTIAL_PERSISTENCE", "CANCELLED", "FAILED",
}


def free_tier_storage_enabled() -> bool:
    return str(os.getenv("CAK_FREE_TIER_STORAGE_MODE", "1")).strip().lower() not in {"0", "false", "off", "no"}


def _quoted_in(values: list[str]) -> str:
    safe = [str(value).replace('"', "") for value in values if str(value)]
    return "in.(" + ",".join('"' + value + '"' for value in safe) + ")"


def _delete_ids(config: DatabaseConfig, table: str, column: str, ids: list[str], *, chunk_size: int = 80) -> int:
    if not config.ready or not ids:
        return 0
    deleted = 0
    for start in range(0, len(ids), max(20, int(chunk_size))):
        chunk = ids[start:start + max(20, int(chunk_size))]
        try:
            response = _request(
                config, "DELETE", table,
                params={column: _quoted_in(chunk)},
                timeout=20,
                return_rows=True,
            )
            payload = response.json()
            deleted += len(payload) if isinstance(payload, list) else len(chunk)
        except Exception:
            # Housekeeping must never break a scan.
            continue
    return deleted


def prune_scan_history_best_effort(
    config: DatabaseConfig,
    *,
    keep_scan_runs: int = 2,
    keep_terminal_jobs: int = 2,
    exclude_scan_id: str = "",
) -> dict[str, Any]:
    """Bound scan-result and resumable-job history.

    Deleting ``cak_scan_runs`` cascades to radar/narrative/provider/direct/autonomous
    rows by the existing schema. Outcome memory is preserved (its FK is SET NULL).
    Deleting ``cak_scan_jobs`` cascades to job chunks.
    """
    report: dict[str, Any] = {
        "state": "SKIPPED",
        "scan_runs_deleted": 0,
        "jobs_deleted": 0,
        "version": FREE_TIER_STORAGE_VERSION,
    }
    if not free_tier_storage_enabled() or not config.ready:
        return report

    try:
        response = _request(
            config, "GET", "cak_scan_runs",
            params={"select": "scan_id,created_at", "order": "created_at.desc", "limit": "50"},
            timeout=15,
        )
        rows = response.json()
        rows = rows if isinstance(rows, list) else []
        keep = max(1, int(keep_scan_runs))
        old = [
            str(row.get("scan_id") or "")
            for row in rows[keep:]
            if str(row.get("scan_id") or "") and str(row.get("scan_id") or "") != str(exclude_scan_id or "")
        ]
        report["scan_runs_deleted"] = _delete_ids(config, "cak_scan_runs", "scan_id", old)
    except Exception as exc:
        report["scan_runs_error"] = f"{type(exc).__name__}: {exc}"[:300]

    try:
        response = _request(
            config, "GET", "cak_scan_jobs",
            params={"select": "scan_id,status,updated_at,created_at", "order": "updated_at.desc", "limit": "80"},
            timeout=15,
        )
        rows = response.json()
        rows = rows if isinstance(rows, list) else []
        terminal = [row for row in rows if str(row.get("status") or "") in TERMINAL_JOB_STATUSES]
        keep = max(1, int(keep_terminal_jobs))
        old = [
            str(row.get("scan_id") or "")
            for row in terminal[keep:]
            if str(row.get("scan_id") or "") and str(row.get("scan_id") or "") != str(exclude_scan_id or "")
        ]
        report["jobs_deleted"] = _delete_ids(config, "cak_scan_jobs", "scan_id", old)
    except Exception as exc:
        report["jobs_error"] = f"{type(exc).__name__}: {exc}"[:300]

    report["state"] = "HOUSEKEEPING_ATTEMPTED"
    return report


def prune_research_memory_best_effort(
    config: DatabaseConfig,
    *,
    keep_default: int = 6,
    keep_narrative: int = 8,
    max_rows_to_inspect: int = 50_000,
) -> dict[str, Any]:
    """Keep only the latest N rows per ticker/family.

    This bounds durable research memory without touching current provider/source
    cache. Metadata is paged without loading JSON payloads.
    """
    report: dict[str, Any] = {
        "state": "SKIPPED",
        "rows_inspected": 0,
        "rows_deleted": 0,
        "version": FREE_TIER_STORAGE_VERSION,
    }
    if not free_tier_storage_enabled() or not config.ready:
        return report

    rows: list[dict[str, Any]] = []
    page_size = 1000
    try:
        for start in range(0, max(0, int(max_rows_to_inspect)), page_size):
            response = _request(
                config, "GET", "cak_research_memory",
                params={
                    "select": "memory_id,ticker,family,effective_period,observed_at,updated_at",
                    "order": "ticker.asc,family.asc,effective_period.desc.nullslast,observed_at.desc.nullslast,updated_at.desc",
                },
                extra_headers={"Range": f"{start}-{start + page_size - 1}"},
                timeout=20,
            )
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break
            rows.extend(payload)
            if len(payload) < page_size:
                break
    except Exception as exc:
        report["state"] = "READ_FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return report

    report["rows_inspected"] = len(rows)
    if not rows:
        report["state"] = "NO_ROWS"
        return report

    seen: dict[tuple[str, str], int] = {}
    delete_ids: list[str] = []
    for row in rows:
        ticker = str(row.get("ticker") or "")
        family = str(row.get("family") or "")
        key = (ticker, family)
        seen[key] = seen.get(key, 0) + 1
        limit = max(1, int(keep_narrative if family == "NARRATIVE_EVENT" else keep_default))
        if seen[key] > limit:
            memory_id = str(row.get("memory_id") or "")
            if memory_id:
                delete_ids.append(memory_id)

    report["rows_deleted"] = _delete_ids(config, "cak_research_memory", "memory_id", delete_ids, chunk_size=60)
    report["state"] = "PRUNED" if delete_ids else "WITHIN_LIMIT"
    report["rows_prune_candidates"] = len(delete_ids)
    return report


__all__ = [
    "FREE_TIER_STORAGE_VERSION",
    "free_tier_storage_enabled",
    "prune_scan_history_best_effort",
    "prune_research_memory_best_effort",
]
