from __future__ import annotations

"""Free-tier storage retention for the Emir scanner.

Housekeeping is best-effort and must never block a scan. Current upsert caches
are not removed here. Durable research memory is content-deduplicated before
bounded ticker/family retention so repeated scans do not store the same evidence
multiple times.
"""

from typing import Any
import os

from persistence import DatabaseConfig, _request

FREE_TIER_STORAGE_VERSION = "1.9.21-terminal-maintenance-latency"
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
            continue
    return deleted


def prune_scan_history_best_effort(
    config: DatabaseConfig,
    *,
    keep_scan_runs: int = 1,
    keep_terminal_jobs: int = 1,
    exclude_scan_id: str = "",
) -> dict[str, Any]:
    """Keep one completed scan/job checkpoint on the Free plan.

    Deleting ``cak_scan_runs`` cascades to per-scan radar/narrative/provider/
    direct/autonomous rows. Outcome memory is preserved because its FK uses
    ``ON DELETE SET NULL``. Deleting terminal jobs cascades to job chunks.
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


def run_outcome_maintenance_best_effort(
    config: DatabaseConfig,
    *,
    scan_id: str,
    resolve_limit: int = 500,
    seed_limit: int = 60,
) -> dict[str, Any]:
    """Run bounded outcome maintenance after the terminal job commit.

    This work deliberately stays outside the ``cak_scan_jobs`` status-update
    transaction.  A terminal trigger previously resolved up to 5,000 outcomes,
    seeded new observations and pruned the full database synchronously.  On the
    hosted Free plan that work could exceed ``statement_timeout`` and roll back
    an otherwise valid COMPLETED transition.

    Outcome maintenance is useful but never allowed to make the analytical
    result or its terminal checkpoint fail.  Each RPC is therefore bounded and
    independently reported.
    """
    report: dict[str, Any] = {
        "state": "SKIPPED",
        "scan_id": str(scan_id or ""),
        "resolve_state": "SKIPPED",
        "seed_state": "SKIPPED",
        "version": FREE_TIER_STORAGE_VERSION,
    }
    if not config.ready or not scan_id:
        return report

    calls = (
        (
            "resolve",
            "cak_resolve_outcome_memory",
            {"p_limit": max(1, min(1000, int(resolve_limit or 500)))},
        ),
        (
            "seed",
            "cak_seed_outcomes_for_scan",
            {
                "p_scan_id": str(scan_id),
                "p_limit": max(1, min(100, int(seed_limit or 60))),
            },
        ),
    )
    completed = 0
    for label, function_name, payload in calls:
        try:
            _request(
                config,
                "POST",
                f"rpc/{function_name}",
                payload=payload,
                timeout=20,
                return_rows=False,
            )
            report[f"{label}_state"] = "COMPLETED"
            completed += 1
        except Exception as exc:
            report[f"{label}_state"] = "FAILED_BEST_EFFORT"
            report[f"{label}_error"] = f"{type(exc).__name__}: {exc}"[:300]

    report["state"] = "COMPLETED" if completed == len(calls) else "PARTIAL" if completed else "FAILED_BEST_EFFORT"
    return report


def prune_research_memory_best_effort(
    config: DatabaseConfig,
    *,
    keep_default: int = 4,
    keep_narrative: int = 6,
    max_rows_to_inspect: int = 50_000,
) -> dict[str, Any]:
    """Content-deduplicate, then bound durable memory per ticker/family.

    Exact repeated evidence is identified by ``content_sha256``. The newest row
    is retained, then the remaining unique observations are bounded by family.
    No JSON payloads need to be transferred for this housekeeping path.
    """
    report: dict[str, Any] = {
        "state": "SKIPPED",
        "rows_inspected": 0,
        "rows_deleted": 0,
        "duplicate_rows": 0,
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
                    "select": "memory_id,ticker,family,content_sha256,effective_period,observed_at,updated_at",
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

    seen_counts: dict[tuple[str, str], int] = {}
    seen_hashes: set[tuple[str, str, str]] = set()
    delete_ids: list[str] = []
    duplicates = 0
    for row in rows:
        ticker = str(row.get("ticker") or "")
        family = str(row.get("family") or "")
        content_hash = str(row.get("content_sha256") or "").strip()
        memory_id = str(row.get("memory_id") or "")
        if content_hash:
            hash_key = (ticker, family, content_hash)
            if hash_key in seen_hashes:
                if memory_id:
                    delete_ids.append(memory_id)
                    duplicates += 1
                continue
            seen_hashes.add(hash_key)

        key = (ticker, family)
        seen_counts[key] = seen_counts.get(key, 0) + 1
        if family == "NARRATIVE_EVENT":
            limit = max(1, int(keep_narrative))
        elif family == "KSEI_CORPORATE_ACTION":
            limit = max(1, min(6, int(keep_narrative)))
        else:
            limit = max(1, int(keep_default))
        if seen_counts[key] > limit and memory_id:
            delete_ids.append(memory_id)

    report["duplicate_rows"] = duplicates
    report["rows_deleted"] = _delete_ids(config, "cak_research_memory", "memory_id", list(dict.fromkeys(delete_ids)), chunk_size=60)
    report["state"] = "PRUNED" if delete_ids else "WITHIN_LIMIT"
    report["rows_prune_candidates"] = len(set(delete_ids))
    return report


__all__ = [
    "FREE_TIER_STORAGE_VERSION",
    "free_tier_storage_enabled",
    "prune_scan_history_best_effort",
    "prune_research_memory_best_effort",
    "run_outcome_maintenance_best_effort",
]
