from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

import numpy as np
import pandas as pd
import requests


BRIDGE_VERSION = "1.9.0"
DATABASE_SCHEMA_VERSION = "emir_autonomous_schema_v8"
SCANNER_VERSION = "1.9.0-real-money-official-xbrl"
DEFAULT_WRITE_CHUNK_SIZE = 200


@dataclass(frozen=True)
class DatabaseConfig:
    enabled: bool
    url: str
    key: str
    schema: str = "public"
    key_type: str = "NONE"

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self.url and self.key)


def _detect_key_type(key: str, source_name: str = "") -> str:
    if not key:
        return "NONE"
    if key.startswith("sb_secret_"):
        return "SECRET"
    if key.startswith("sb_publishable_"):
        return "PUBLISHABLE_REJECTED"
    if key.count(".") == 2 or "SERVICE_ROLE" in source_name:
        return "LEGACY_SERVICE_ROLE"
    return "BACKEND_KEY_UNKNOWN_FORMAT"


def config_from_mapping(mapping: Any) -> DatabaseConfig:
    get = mapping.get if hasattr(mapping, "get") else lambda key, default=None: default
    enabled = str(get("CAK_DATABASE_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    url = str(get("SUPABASE_URL", "") or "").strip().rstrip("/")
    candidates = [
        ("SUPABASE_SECRET_KEY", str(get("SUPABASE_SECRET_KEY", "") or "").strip()),
        ("SUPABASE_SERVICE_ROLE_KEY", str(get("SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()),
    ]
    source_name, key = next(((name, value) for name, value in candidates if value), ("", ""))
    schema = str(get("CAK_DATABASE_SCHEMA", "public") or "public").strip()
    key_type = _detect_key_type(key, source_name)
    if key_type == "PUBLISHABLE_REJECTED":
        key = ""
    return DatabaseConfig(enabled=enabled, url=url, key=key, schema=schema, key_type=key_type)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value if isinstance(value, (dict, list, str)) else str(value)


def _headers(config: DatabaseConfig, *, return_rows: bool = True) -> dict[str, str]:
    prefer = "resolution=merge-duplicates"
    if return_rows:
        prefer += ",return=representation"
    headers = {
        "apikey": config.key,
        "Content-Type": "application/json",
        "Accept-Profile": config.schema,
        "Content-Profile": config.schema,
        "Prefer": prefer,
    }
    # Opaque sb_secret keys are API keys, not JWT bearer tokens.
    if config.key_type == "LEGACY_SERVICE_ROLE":
        headers["Authorization"] = f"Bearer {config.key}"
    return headers


def _request(
    config: DatabaseConfig,
    method: str,
    table: str,
    *,
    params: dict[str, Any] | None = None,
    payload: Any = None,
    timeout: int = 35,
    return_rows: bool = True,
    extra_headers: dict[str, str] | None = None,
) -> requests.Response:
    headers = _headers(config, return_rows=return_rows)
    if extra_headers:
        headers.update(extra_headers)
    response = requests.request(
        method,
        f"{config.url}/rest/v1/{table}",
        params=params,
        data=json.dumps(payload, default=str) if payload is not None else None,
        headers=headers,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{table} HTTP {response.status_code}: {response.text[:1500]}")
    return response


def database_status(config: DatabaseConfig) -> dict[str, Any]:
    return {
        "bridge_version": BRIDGE_VERSION,
        "schema_version": DATABASE_SCHEMA_VERSION,
        "database_mode": "SUPABASE_REST" if config.ready else "DISABLED",
        "database_key_type": config.key_type,
        "read_enabled": config.ready,
        "write_policy": "RESUMABLE_CHUNK_CHECKPOINT_PLUS_BEST_EFFORT_RESULTS",
    }


def test_connection(config: DatabaseConfig) -> pd.DataFrame:
    columns = [
        "bridge_version", "schema_version", "database_mode", "database_key_type", "write_policy",
        "state", "table", "detail",
    ]
    base = database_status(config)
    if not config.ready:
        return pd.DataFrame(
            [{
                **base,
                "state": "DATABASE_DISABLED",
                "table": "__SUMMARY__",
                "detail": "Set CAK_DATABASE_ENABLED, SUPABASE_URL, and SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY.",
            }],
            columns=columns,
        )
    checks = {
        "cak_scan_runs": "scan_id",
        "cak_radar_snapshots": "scan_id",
        "cak_narrative_events": "event_id",
        "cak_provider_audit": "audit_id",
        "cak_direct_evidence": "evidence_id",
        "cak_autonomous_evidence": "evidence_id",
        "cak_outcome_memory": "outcome_id",
        "cak_ohlcv_cache": "ticker",
        "cak_source_cache": "cache_key",
        "cak_scan_jobs": "scan_id",
        "cak_scan_job_chunks": "chunk_id",
        "cak_research_memory": "memory_id",
    }
    rows: list[dict[str, Any]] = []
    for table, select_column in checks.items():
        try:
            _request(config, "GET", table, params={"select": select_column, "limit": "1"}, timeout=8)
            rows.append({**base, "state": "READY", "table": table, "detail": "REST read succeeded."})
        except Exception as exc:
            rows.append({**base, "state": "MIGRATION_OR_PERMISSION_REQUIRED", "table": table, "detail": str(exc)})
    healthy = all(row["state"] == "READY" for row in rows)
    rows.insert(
        0,
        {
            **base,
            "state": "HEALTHY_EMIR_DATABASE_V8" if healthy else "DATABASE_NOT_READY_V8",
            "table": "__SUMMARY__",
            "detail": f"{sum(row['state'] == 'READY' for row in rows)}/{len(checks)} tables readable.",
        },
    )
    return pd.DataFrame(rows, columns=columns)


def _written_count(response: requests.Response, fallback: int) -> int:
    try:
        payload = response.json()
        return len(payload) if isinstance(payload, list) else fallback
    except Exception:
        return fallback


def _post_payload_in_chunks(
    config: DatabaseConfig,
    *,
    table: str,
    conflict: str,
    payload: list[dict[str, Any]],
    chunk_size: int = DEFAULT_WRITE_CHUNK_SIZE,
    return_rows: bool = True,
) -> int:
    if not payload:
        return 0
    safe_chunk_size = max(1, int(chunk_size))
    written_total = 0
    for start in range(0, len(payload), safe_chunk_size):
        chunk = payload[start:start + safe_chunk_size]
        response = _request(
            config,
            "POST",
            table,
            params={"on_conflict": conflict},
            payload=chunk,
            return_rows=return_rows,
        )
        written = _written_count(response, len(chunk))
        if written != len(chunk):
            raise RuntimeError(
                f"{table} partial chunk write: expected {len(chunk)} rows, provider returned {written}."
            )
        written_total += written
    return written_total


def _direct_evidence_payload(scan_id: str, frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return payload
    for index, row in frame.reset_index(drop=True).iterrows():
        record = {key: _json_value(value) for key, value in row.to_dict().items()}
        ticker = str(record.get("ticker") or "")
        evidence_type = str(record.get("evidence_type") or "DIRECT_EVIDENCE")
        observed_at = record.get("observed_at") or record.get("date") or record.get("published_at")
        payload.append({
            "evidence_id": f"{scan_id}:{evidence_type}:{ticker}:{index}",
            "scan_id": scan_id,
            "ticker": ticker,
            "evidence_type": evidence_type,
            "observed_at": observed_at,
            "source_verified": bool(record.get("source_verified", False)),
            "payload": record,
        })
    return payload


def _outcome_payload(scan_id: str, frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return payload
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    for index, row in local.reset_index(drop=True).iterrows():
        record = {key: _json_value(value) for key, value in row.to_dict().items()}
        ticker = str(record.get("ticker") or "")
        signal_date = record.get("signal_date") or record.get("date")
        horizon = record.get("horizon_days")
        outcome_id = str(record.get("outcome_id") or f"{ticker}:{signal_date}:{horizon}:{index}")
        payload.append({
            "outcome_id": outcome_id,
            "scan_id": scan_id,
            "ticker": ticker,
            "signal_date": signal_date,
            "horizon_days": horizon,
            "outcome_verified": str(record.get("outcome_verified", False)).strip().lower() in {"1", "true", "yes", "y", "verified"},
            "payload": record,
        })
    return payload


def _set_scan_status(config: DatabaseConfig, scan_id: str, status: str) -> str:
    response = _request(
        config,
        "PATCH",
        "cak_scan_runs",
        params={"scan_id": f"eq.{scan_id}"},
        payload={"status": status},
    )
    rows = response.json()
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError(f"cak_scan_runs status update returned {len(rows) if isinstance(rows, list) else 'non-list'} rows.")
    observed = str(rows[0].get("status") or "")
    if observed != status:
        raise RuntimeError(f"cak_scan_runs status mismatch: expected {status}, got {observed or 'EMPTY'}.")
    return observed


def persist_scan(
    config: DatabaseConfig,
    *,
    scan_id: str,
    as_of: Any,
    radar: pd.DataFrame,
    events: pd.DataFrame | None,
    provider_audit: pd.DataFrame | None,
    direct_evidence: pd.DataFrame | None,
    autonomous_evidence: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
    mode: str = "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
    chunk_size: int = DEFAULT_WRITE_CHUNK_SIZE,
) -> pd.DataFrame:
    columns = [
        "bridge_version", "schema_version", "database_mode", "database_key_type", "write_policy", "table",
        "rows_attempted", "rows_written", "state", "detail",
    ]
    base = database_status(config)
    if not config.ready:
        return pd.DataFrame(
            [{
                **base,
                "table": "__SUMMARY__",
                "rows_attempted": 0,
                "rows_written": 0,
                "state": "DATABASE_DISABLED",
                "detail": "Database disabled; scan may continue in memory and missing cache rows will be fetched from providers.",
            }],
            columns=columns,
        )

    reports: list[dict[str, Any]] = []
    as_of_iso = pd.Timestamp(as_of).isoformat()
    run_payload = [{
        "scan_id": scan_id,
        "as_of": as_of_iso,
        "scanner_version": SCANNER_VERSION,
        "scan_mode": mode,
        "ticker_count": int(radar["ticker"].nunique()) if not radar.empty else 0,
        "production_ready_count": int(radar.get("production_ready", pd.Series(False)).fillna(False).sum()) if not radar.empty else 0,
        "status": "PERSISTING",
    }]
    payloads: list[tuple[str, str, list[dict[str, Any]]]] = [("cak_scan_runs", "scan_id", run_payload)]

    radar_payload: list[dict[str, Any]] = []
    for _, row in radar.iterrows():
        record = {key: _json_value(value) for key, value in row.to_dict().items()}
        radar_payload.append({
            "scan_id": scan_id,
            "ticker": str(record.pop("ticker")),
            "as_of": as_of_iso,
            "public_method_state": record.get("emir_decision_state"),
            "action": record.get("action"),
            "conviction_score": record.get("emir_conviction_score"),
            "coverage_pct": record.get("emir_evidence_coverage_pct"),
            "production_ready": record.get("production_ready", False),
            "payload": record,
        })
    payloads.append(("cak_radar_snapshots", "scan_id,ticker", radar_payload))

    event_payload: list[dict[str, Any]] = []
    if isinstance(events, pd.DataFrame) and not events.empty:
        for index, row in events.reset_index(drop=True).iterrows():
            record = {key: _json_value(value) for key, value in row.to_dict().items()}
            event_payload.append({
                "event_id": f"{scan_id}:{record.get('ticker', '')}:{index}",
                "scan_id": scan_id,
                "ticker": str(record.get("ticker") or ""),
                "published_at": record.get("published_at") or record.get("event_date"),
                "title": record.get("title"),
                "publisher": record.get("publisher"),
                "source_url": record.get("url") or record.get("source_url"),
                "payload": record,
            })
    payloads.append(("cak_narrative_events", "event_id", event_payload))

    provider_payload: list[dict[str, Any]] = []
    if isinstance(provider_audit, pd.DataFrame) and not provider_audit.empty:
        for index, row in provider_audit.reset_index(drop=True).iterrows():
            record = {key: _json_value(value) for key, value in row.to_dict().items()}
            provider_payload.append({
                "audit_id": f"{scan_id}:{index}",
                "scan_id": scan_id,
                "ticker": str(record.get("ticker") or ""),
                "provider": str(record.get("provider") or record.get("audit_family") or ""),
                "status": str(record.get("status") or ""),
                "payload": record,
            })
    payloads.append(("cak_provider_audit", "audit_id", provider_payload))
    payloads.append(("cak_direct_evidence", "evidence_id", _direct_evidence_payload(scan_id, direct_evidence)))
    payloads.append(("cak_autonomous_evidence", "evidence_id", _direct_evidence_payload(scan_id, autonomous_evidence)))
    payloads.append(("cak_outcome_memory", "outcome_id", _outcome_payload(scan_id, outcomes)))

    for table, conflict, payload in payloads:
        attempted = len(payload)
        try:
            written = _post_payload_in_chunks(
                config,
                table=table,
                conflict=conflict,
                payload=payload,
                chunk_size=chunk_size,
            )
            reports.append({
                **base,
                "table": table,
                "rows_attempted": attempted,
                "rows_written": written,
                "state": "EMPTY_EXPECTED" if attempted == 0 else "WRITTEN",
                "detail": "" if attempted else "No rows expected for this table.",
            })
        except Exception as exc:
            reports.append({
                **base,
                "table": table,
                "rows_attempted": attempted,
                "rows_written": 0,
                "state": "WRITE_FAILED",
                "detail": str(exc),
            })

    attempted_total = sum(row["rows_attempted"] for row in reports)
    written_total = sum(row["rows_written"] for row in reports)
    all_tables_exact = all(
        row["state"] in {"WRITTEN", "EMPTY_EXPECTED"}
        and row["rows_written"] == row["rows_attempted"]
        for row in reports
    )
    state = "WRITE_ALL_TABLES" if all_tables_exact and written_total == attempted_total else "WRITE_PARTIAL"
    try:
        _set_scan_status(config, scan_id, "PERSISTED_PENDING_READBACK" if state == "WRITE_ALL_TABLES" else "WRITE_PARTIAL")
        status_detail = "scan_runs status updated."
    except Exception as exc:
        state = "WRITE_PARTIAL"
        status_detail = f"scan_runs status update failed: {exc}"
    reports.insert(0, {
        **base,
        "table": "__SUMMARY__",
        "rows_attempted": attempted_total,
        "rows_written": written_total,
        "state": state,
        "detail": f"{written_total}/{attempted_total} rows written; {status_detail}",
    })
    return pd.DataFrame(reports, columns=columns)


def _exact_count_from_response(response: requests.Response) -> int:
    """Return the PostgREST exact total from Content-Range, with a safe test fallback.

    PostgREST normally caps returned rows (commonly 1,000). Counting ``len(json())``
    therefore produces a false mismatch for larger scan evidence tables. ``count=exact``
    exposes the real total in ``Content-Range`` even when only one row is returned.
    """
    headers = getattr(response, "headers", {}) or {}
    content_range = str(headers.get("Content-Range") or headers.get("content-range") or "")
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[-1].strip()
        if total.isdigit():
            return int(total)
    try:
        payload = response.json()
    except Exception:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _count_rows_for_scan(config: DatabaseConfig, table: str, scan_id: str) -> int:
    response = _request(
        config,
        "GET",
        table,
        params={"select": "scan_id", "scan_id": f"eq.{scan_id}", "limit": "1"},
        extra_headers={"Prefer": "count=exact", "Range": "0-0"},
    )
    return _exact_count_from_response(response)


def verify_scan(
    config: DatabaseConfig,
    *,
    scan_id: str,
    expected_radar: int,
    expected_events: int,
    expected_provider_audit: int,
    expected_direct_evidence: int,
    expected_autonomous_evidence: int = 0,
    expected_outcomes: int = 0,
) -> pd.DataFrame:
    columns = [
        "bridge_version", "schema_version", "database_mode", "database_key_type", "write_policy", "state", "table",
        "scan_id", "rows_attempted", "rows_written", "rows_verified", "verification_pct", "detail",
    ]
    base = database_status(config)
    if not config.ready:
        return pd.DataFrame(
            [{
                **base,
                "state": "DATABASE_DISABLED",
                "table": "__SUMMARY__",
                "scan_id": scan_id,
                "rows_attempted": 0,
                "rows_written": 0,
                "rows_verified": 0,
                "verification_pct": 0.0,
                "detail": "Database is not configured.",
            }],
            columns=columns,
        )

    checks = [
        ("cak_scan_runs", 1),
        ("cak_radar_snapshots", expected_radar),
        ("cak_narrative_events", expected_events),
        ("cak_provider_audit", expected_provider_audit),
        ("cak_direct_evidence", expected_direct_evidence),
        ("cak_autonomous_evidence", expected_autonomous_evidence),
        ("cak_outcome_memory", expected_outcomes),
    ]
    reports: list[dict[str, Any]] = []
    total_expected = total_verified_exact = 0
    all_exact = True
    for table, expected in checks:
        total_expected += expected
        try:
            verified = _count_rows_for_scan(config, table, scan_id)
            exact = verified == expected
            all_exact = all_exact and exact
            total_verified_exact += verified if exact else 0
            pct = 100.0 if exact else 100.0 * min(verified, expected) / max(expected, 1)
            reports.append({
                **base,
                "state": "VERIFIED_EXACT" if exact else "ROW_COUNT_MISMATCH",
                "table": table,
                "scan_id": scan_id,
                "rows_attempted": expected,
                "rows_written": expected,
                "rows_verified": verified,
                "verification_pct": round(pct, 1),
                "detail": "Exact persisted scan_id row count found." if exact else f"Expected {expected}, found {verified}.",
            })
        except Exception as exc:
            all_exact = False
            reports.append({
                **base,
                "state": "READBACK_FAILED",
                "table": table,
                "scan_id": scan_id,
                "rows_attempted": expected,
                "rows_written": 0,
                "rows_verified": 0,
                "verification_pct": 0.0,
                "detail": str(exc),
            })

    pct = 100.0 if all_exact else 100.0 * total_verified_exact / max(total_expected, 1)
    state = "VERIFIED_ALL_TABLES" if all_exact else "PARTIAL_READBACK"
    reports.insert(0, {
        **base,
        "state": state,
        "table": "__SUMMARY__",
        "scan_id": scan_id,
        "rows_attempted": total_expected,
        "rows_written": total_expected,
        "rows_verified": total_expected if all_exact else total_verified_exact,
        "verification_pct": round(pct, 1),
        "detail": (
            f"Exact row counts verified across 7 tables ({total_expected}/{total_expected})."
            if all_exact else "One or more tables failed exact row-count readback."
        ),
    })
    return pd.DataFrame(reports, columns=columns)


def finalize_scan_commit(config: DatabaseConfig, *, scan_id: str, committed: bool) -> pd.DataFrame:
    columns = [
        "bridge_version", "schema_version", "database_mode", "database_key_type", "write_policy",
        "state", "scan_id", "requested_status", "observed_status", "detail",
    ]
    base = database_status(config)
    requested_status = "VERIFIED_COMMITTED" if committed else "COMMIT_FAILED"
    if not config.ready:
        return pd.DataFrame([{
            **base,
            "state": "DATABASE_DISABLED",
            "scan_id": scan_id,
            "requested_status": requested_status,
            "observed_status": "",
            "detail": "Database is not configured.",
        }], columns=columns)
    try:
        _set_scan_status(config, scan_id, requested_status)
        response = _request(
            config,
            "GET",
            "cak_scan_runs",
            params={"select": "scan_id,status", "scan_id": f"eq.{scan_id}"},
        )
        rows = response.json()
        observed = str(rows[0].get("status") or "") if isinstance(rows, list) and len(rows) == 1 else ""
        exact = observed == requested_status
        state = "DATABASE_FIRST_COMMITTED" if committed and exact else "COMMIT_FAILED_CONFIRMED" if (not committed and exact) else "COMMIT_STATUS_MISMATCH"
        detail = "Scan is publishable only because final database status was read back exactly." if state == "DATABASE_FIRST_COMMITTED" else "Scan is not publishable."
    except Exception as exc:
        observed = ""
        state = "COMMIT_STATUS_UPDATE_FAILED"
        detail = str(exc)
    return pd.DataFrame([{
        **base,
        "state": state,
        "scan_id": scan_id,
        "requested_status": requested_status,
        "observed_status": observed,
        "detail": detail,
    }], columns=columns)


def persist_verify_commit_scan(
    config: DatabaseConfig,
    *,
    scan_id: str,
    as_of: Any,
    radar: pd.DataFrame,
    events: pd.DataFrame | None,
    provider_audit: pd.DataFrame | None,
    direct_evidence: pd.DataFrame | None,
    autonomous_evidence: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
    mode: str = "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
    chunk_size: int = DEFAULT_WRITE_CHUNK_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Persist, exact-readback verify, and commit a scan before it can be published."""
    write_report = persist_scan(
        config,
        scan_id=scan_id,
        as_of=as_of,
        radar=radar,
        events=events,
        provider_audit=provider_audit,
        direct_evidence=direct_evidence,
        autonomous_evidence=autonomous_evidence,
        outcomes=outcomes,
        mode=mode,
        chunk_size=chunk_size,
    )
    write_ok = not write_report.empty and str(write_report.iloc[0].get("state")) == "WRITE_ALL_TABLES"
    if write_ok:
        verification = verify_scan(
            config,
            scan_id=scan_id,
            expected_radar=len(radar) if isinstance(radar, pd.DataFrame) else 0,
            expected_events=len(events) if isinstance(events, pd.DataFrame) else 0,
            expected_provider_audit=len(provider_audit) if isinstance(provider_audit, pd.DataFrame) else 0,
            expected_direct_evidence=len(direct_evidence) if isinstance(direct_evidence, pd.DataFrame) else 0,
            expected_autonomous_evidence=len(autonomous_evidence) if isinstance(autonomous_evidence, pd.DataFrame) else 0,
            expected_outcomes=len(outcomes) if isinstance(outcomes, pd.DataFrame) else 0,
        )
    else:
        verification = pd.DataFrame([{
            **database_status(config),
            "state": "READBACK_SKIPPED_WRITE_INCOMPLETE",
            "table": "__SUMMARY__",
            "scan_id": scan_id,
            "rows_attempted": 0,
            "rows_written": 0,
            "rows_verified": 0,
            "verification_pct": 0.0,
            "detail": "Exact readback was skipped because database write was incomplete.",
        }])
    verify_ok = not verification.empty and str(verification.iloc[0].get("state")) == "VERIFIED_ALL_TABLES" and float(verification.iloc[0].get("verification_pct", 0) or 0) == 100.0
    commit_report = finalize_scan_commit(config, scan_id=scan_id, committed=write_ok and verify_ok)
    return write_report, verification, commit_report


def database_commit_succeeded(commit_report: pd.DataFrame | None) -> bool:
    return bool(
        isinstance(commit_report, pd.DataFrame)
        and not commit_report.empty
        and str(commit_report.iloc[0].get("state")) == "DATABASE_FIRST_COMMITTED"
        and str(commit_report.iloc[0].get("observed_status")) == "VERIFIED_COMMITTED"
    )


def _summary_row(frame: pd.DataFrame | None) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    if "table" in frame.columns:
        summary = frame.loc[frame["table"].astype(str).eq("__SUMMARY__")]
        if not summary.empty:
            return summary.iloc[0].to_dict()
    return frame.iloc[0].to_dict()


def finalize_scan_persistence(
    config: DatabaseConfig,
    *,
    scan_id: str,
    write_report: pd.DataFrame | None,
    verification: pd.DataFrame | None,
) -> pd.DataFrame:
    """Classify persistence without making database completeness a publication gate.

    The scan remains publishable from in-memory results. Supabase is used opportunistically:
    fully verified rows are marked committed; partial rows are marked partial; an unavailable
    database produces a memory-only scan. Analytical data-quality gates remain unchanged.
    """
    columns = [
        "bridge_version", "schema_version", "database_mode", "database_key_type", "write_policy",
        "state", "scan_id", "requested_status", "observed_status", "publishable",
        "rows_attempted", "rows_written", "rows_verified", "verification_pct", "detail",
    ]
    base = database_status(config)
    write = _summary_row(write_report)
    verify = _summary_row(verification)
    attempted = int(float(write.get("rows_attempted", 0) or 0))
    written = int(float(write.get("rows_written", 0) or 0))
    verified = int(float(verify.get("rows_verified", 0) or 0))
    pct = float(verify.get("verification_pct", 0) or 0)
    full = (
        str(write.get("state", "")) == "WRITE_ALL_TABLES"
        and str(verify.get("state", "")) == "VERIFIED_ALL_TABLES"
        and pct == 100.0
    )

    if full:
        requested = "VERIFIED_COMMITTED"
        state = "SCAN_COMPLETED_FULL_PERSISTENCE"
        detail = "All expected scan rows were written and exact-count verified."
    elif written > 0:
        requested = "PARTIAL_PERSISTENCE"
        state = "SCAN_COMPLETED_PARTIAL_PERSISTENCE"
        detail = "Some scan rows were persisted. Missing/unverified rows remain publishable in memory and will be fetched/recomputed on a later scan."
    else:
        requested = "MEMORY_ONLY"
        state = "SCAN_COMPLETED_MEMORY_ONLY"
        detail = "No database rows were confirmed. The scan is published from memory; cache misses are fetched from live providers."

    observed = ""
    if config.ready and written > 0:
        try:
            _set_scan_status(config, scan_id, requested)
            response = _request(
                config,
                "GET",
                "cak_scan_runs",
                params={"select": "scan_id,status", "scan_id": f"eq.{scan_id}"},
            )
            rows = response.json()
            observed = str(rows[0].get("status") or "") if isinstance(rows, list) and len(rows) == 1 else ""
            if observed != requested:
                detail += f" Status readback mismatch: requested {requested}, observed {observed or 'EMPTY'}."
        except Exception as exc:
            detail += f" Status update/readback failed: {exc}"

    return pd.DataFrame([{
        **base,
        "state": state,
        "scan_id": scan_id,
        "requested_status": requested,
        "observed_status": observed,
        "publishable": True,
        "rows_attempted": attempted,
        "rows_written": written,
        "rows_verified": verified,
        "verification_pct": round(pct, 1),
        "detail": detail,
    }], columns=columns)


def persist_verify_scan_best_effort(
    config: DatabaseConfig,
    *,
    scan_id: str,
    as_of: Any,
    radar: pd.DataFrame,
    events: pd.DataFrame | None,
    provider_audit: pd.DataFrame | None,
    direct_evidence: pd.DataFrame | None,
    autonomous_evidence: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
    mode: str = "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
    chunk_size: int = DEFAULT_WRITE_CHUNK_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Persist what can be persisted, verify what can be verified, never block publication."""
    write_report = persist_scan(
        config,
        scan_id=scan_id,
        as_of=as_of,
        radar=radar,
        events=events,
        provider_audit=provider_audit,
        direct_evidence=direct_evidence,
        autonomous_evidence=autonomous_evidence,
        outcomes=outcomes,
        mode=mode,
        chunk_size=chunk_size,
    )
    if config.ready:
        verification = verify_scan(
            config,
            scan_id=scan_id,
            expected_radar=len(radar) if isinstance(radar, pd.DataFrame) else 0,
            expected_events=len(events) if isinstance(events, pd.DataFrame) else 0,
            expected_provider_audit=len(provider_audit) if isinstance(provider_audit, pd.DataFrame) else 0,
            expected_direct_evidence=len(direct_evidence) if isinstance(direct_evidence, pd.DataFrame) else 0,
            expected_autonomous_evidence=len(autonomous_evidence) if isinstance(autonomous_evidence, pd.DataFrame) else 0,
            expected_outcomes=len(outcomes) if isinstance(outcomes, pd.DataFrame) else 0,
        )
    else:
        verification = verify_scan(
            config,
            scan_id=scan_id,
            expected_radar=0,
            expected_events=0,
            expected_provider_audit=0,
            expected_direct_evidence=0,
            expected_autonomous_evidence=0,
            expected_outcomes=0,
        )
    persistence_report = finalize_scan_persistence(
        config, scan_id=scan_id, write_report=write_report, verification=verification
    )
    return write_report, verification, persistence_report


def scan_publication_allowed(persistence_report: pd.DataFrame | None) -> bool:
    return bool(
        isinstance(persistence_report, pd.DataFrame)
        and not persistence_report.empty
        and bool(persistence_report.iloc[0].get("publishable", False))
        and str(persistence_report.iloc[0].get("state", "")).startswith("SCAN_COMPLETED_")
    )


def full_persistence_succeeded(persistence_report: pd.DataFrame | None) -> bool:
    return bool(
        isinstance(persistence_report, pd.DataFrame)
        and not persistence_report.empty
        and str(persistence_report.iloc[0].get("state", "")) == "SCAN_COMPLETED_FULL_PERSISTENCE"
    )


__all__ = [
    "BRIDGE_VERSION", "DATABASE_SCHEMA_VERSION", "SCANNER_VERSION", "DEFAULT_WRITE_CHUNK_SIZE",
    "DatabaseConfig", "_headers", "_exact_count_from_response", "_count_rows_for_scan", "config_from_mapping", "database_status", "persist_scan",
    "test_connection", "verify_scan", "finalize_scan_commit", "persist_verify_commit_scan",
    "database_commit_succeeded", "finalize_scan_persistence", "persist_verify_scan_best_effort",
    "scan_publication_allowed", "full_persistence_succeeded",
]
