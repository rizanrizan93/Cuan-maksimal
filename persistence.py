from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

import numpy as np
import pandas as pd
import requests


BRIDGE_VERSION = "1.4.0"
DATABASE_SCHEMA_VERSION = "emir_autonomous_schema_v5"
SCANNER_VERSION = "1.4.0-autonomous-public-data-and-eod-proxy"


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
) -> requests.Response:
    response = requests.request(
        method,
        f"{config.url}/rest/v1/{table}",
        params=params,
        data=json.dumps(payload, default=str) if payload is not None else None,
        headers=_headers(config, return_rows=return_rows),
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
    }


def test_connection(config: DatabaseConfig) -> pd.DataFrame:
    columns = ["bridge_version", "schema_version", "database_mode", "database_key_type", "state", "table", "detail"]
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
    checks = [
        "cak_scan_runs",
        "cak_radar_snapshots",
        "cak_narrative_events",
        "cak_provider_audit",
        "cak_direct_evidence",
        "cak_autonomous_evidence",
        "cak_outcome_memory",
    ]
    rows: list[dict[str, Any]] = []
    for table in checks:
        try:
            _request(config, "GET", table, params={"select": "*", "limit": "1"})
            rows.append({**base, "state": "READY", "table": table, "detail": "REST read succeeded."})
        except Exception as exc:
            rows.append({**base, "state": "MIGRATION_OR_PERMISSION_REQUIRED", "table": table, "detail": str(exc)})
    healthy = all(row["state"] == "READY" for row in rows)
    rows.insert(
        0,
        {
            **base,
            "state": "HEALTHY_EMIR_DATABASE_V5" if healthy else "DATABASE_NOT_READY_V5",
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
) -> pd.DataFrame:
    columns = [
        "bridge_version", "schema_version", "database_mode", "database_key_type", "table",
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
                "detail": "Configure Streamlit Secrets.",
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
        "status": "COMPLETED",
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
            response = _request(config, "POST", table, params={"on_conflict": conflict}, payload=payload) if payload else None
            written = _written_count(response, attempted) if response is not None else 0
            reports.append({
                **base,
                "table": table,
                "rows_attempted": attempted,
                "rows_written": written,
                "state": "WRITTEN",
                "detail": "",
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
    state = "WRITE_ALL_TABLES" if written_total == attempted_total else "WRITE_PARTIAL"
    reports.insert(0, {
        **base,
        "table": "__SUMMARY__",
        "rows_attempted": attempted_total,
        "rows_written": written_total,
        "state": state,
        "detail": f"{written_total}/{attempted_total} rows written.",
    })
    return pd.DataFrame(reports, columns=columns)


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
        "bridge_version", "schema_version", "database_mode", "database_key_type", "state", "table",
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
    total_expected = total_verified = 0
    for table, expected in checks:
        total_expected += expected
        try:
            response = _request(config, "GET", table, params={"select": "scan_id", "scan_id": f"eq.{scan_id}"})
            rows = response.json()
            verified = len(rows) if isinstance(rows, list) else 0
            total_verified += min(verified, expected)
            pct = 100.0 if expected == 0 and verified == 0 else 100.0 * min(verified, expected) / max(expected, 1)
            reports.append({
                **base,
                "state": "VERIFIED" if verified >= expected else "MISSING_ROWS",
                "table": table,
                "scan_id": scan_id,
                "rows_attempted": expected,
                "rows_written": expected,
                "rows_verified": verified,
                "verification_pct": round(pct, 1),
                "detail": "Exact persisted scan_id rows found.",
            })
        except Exception as exc:
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

    pct = 100.0 * total_verified / max(total_expected, 1)
    state = "VERIFIED_ALL_TABLES" if total_verified == total_expected else "PARTIAL_READBACK"
    reports.insert(0, {
        **base,
        "state": state,
        "table": "__SUMMARY__",
        "scan_id": scan_id,
        "rows_attempted": total_expected,
        "rows_written": total_expected,
        "rows_verified": total_verified,
        "verification_pct": round(pct, 1),
        "detail": f"{total_verified}/{total_expected} rows verified across 7 tables.",
    })
    return pd.DataFrame(reports, columns=columns)


__all__ = [
    "BRIDGE_VERSION", "DATABASE_SCHEMA_VERSION", "SCANNER_VERSION", "DatabaseConfig", "_headers",
    "config_from_mapping", "database_status", "persist_scan", "test_connection", "verify_scan",
]
