from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class DatabaseConfig:
    enabled: bool
    url: str
    key: str
    schema: str = "public"

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self.url and self.key)


def config_from_mapping(mapping: Any) -> DatabaseConfig:
    get = mapping.get if hasattr(mapping, "get") else lambda key, default=None: default
    enabled = str(get("CAK_DATABASE_ENABLED", get("SCANNER_DATABASE_ENABLED", "false"))).strip().lower() in {"1", "true", "yes", "on"}
    url = str(get("SUPABASE_URL", "") or "").strip().rstrip("/")
    key = str(get("SUPABASE_SECRET_KEY", get("SUPABASE_SERVICE_ROLE_KEY", "")) or "").strip()
    schema = str(get("CAK_DATABASE_SCHEMA", "public") or "public").strip()
    return DatabaseConfig(enabled=enabled, url=url, key=key, schema=schema)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
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
    return {
        "apikey": config.key,
        "Authorization": f"Bearer {config.key}",
        "Content-Type": "application/json",
        "Accept-Profile": config.schema,
        "Content-Profile": config.schema,
        "Prefer": prefer,
    }


def _request(config: DatabaseConfig, method: str, table: str, *, params: dict[str, Any] | None = None, payload: Any = None, timeout: int = 35) -> requests.Response:
    response = requests.request(
        method,
        f"{config.url}/rest/v1/{table}",
        params=params,
        data=json.dumps(payload, default=str) if payload is not None else None,
        headers=_headers(config),
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{table} HTTP {response.status_code}: {response.text[:1000]}")
    return response


def persist_scan(
    config: DatabaseConfig,
    *,
    scan_id: str,
    as_of: Any,
    radar: pd.DataFrame,
    events: pd.DataFrame | None,
    mode: str,
) -> pd.DataFrame:
    columns = ["table", "rows_attempted", "rows_written", "state", "detail"]
    if not config.ready:
        return pd.DataFrame([{"table": "__SUMMARY__", "rows_attempted": 0, "rows_written": 0, "state": "DATABASE_DISABLED", "detail": "Configure CAK_DATABASE_ENABLED, SUPABASE_URL, and a secret/service key."}], columns=columns)
    reports: list[dict[str, Any]] = []
    run_payload = [{
        "scan_id": scan_id,
        "as_of": pd.Timestamp(as_of).isoformat(),
        "scanner_version": "1.0.0-public-narrative-flow",
        "scan_mode": mode,
        "ticker_count": int(radar["ticker"].nunique()) if not radar.empty else 0,
        "production_ready_count": int(radar.get("production_ready", pd.Series(False)).fillna(False).sum()) if not radar.empty else 0,
        "status": "COMPLETED",
    }]
    try:
        written = _request(config, "POST", "cak_scan_runs", params={"on_conflict": "scan_id"}, payload=run_payload).json()
        reports.append({"table": "cak_scan_runs", "rows_attempted": 1, "rows_written": len(written) if isinstance(written, list) else 1, "state": "WRITTEN", "detail": ""})
    except Exception as exc:
        reports.append({"table": "cak_scan_runs", "rows_attempted": 1, "rows_written": 0, "state": "WRITE_FAILED", "detail": str(exc)})
    radar_payload: list[dict[str, Any]] = []
    for _, row in radar.iterrows():
        record = {key: _json_value(value) for key, value in row.to_dict().items()}
        radar_payload.append({
            "scan_id": scan_id,
            "ticker": str(record.pop("ticker")),
            "as_of": pd.Timestamp(as_of).isoformat(),
            "public_method_state": record.pop("public_method_state", None),
            "action": record.pop("action", None),
            "conviction_score": record.pop("narrative_flow_conviction_score", None),
            "coverage_pct": record.pop("narrative_flow_coverage_pct", None),
            "production_ready": record.pop("production_ready", False),
            "payload": record,
        })
    try:
        written = _request(config, "POST", "cak_radar_snapshots", params={"on_conflict": "scan_id,ticker"}, payload=radar_payload).json() if radar_payload else []
        reports.append({"table": "cak_radar_snapshots", "rows_attempted": len(radar_payload), "rows_written": len(written) if isinstance(written, list) else len(radar_payload), "state": "WRITTEN", "detail": ""})
    except Exception as exc:
        reports.append({"table": "cak_radar_snapshots", "rows_attempted": len(radar_payload), "rows_written": 0, "state": "WRITE_FAILED", "detail": str(exc)})
    event_payload: list[dict[str, Any]] = []
    if isinstance(events, pd.DataFrame) and not events.empty:
        for index, row in events.reset_index(drop=True).iterrows():
            record = {key: _json_value(value) for key, value in row.to_dict().items()}
            event_id = f"{scan_id}:{record.get('ticker','')}:{index}"
            event_payload.append({
                "event_id": event_id,
                "scan_id": scan_id,
                "ticker": str(record.get("ticker") or ""),
                "published_at": record.get("published_at") or record.get("event_date"),
                "title": record.get("title"),
                "publisher": record.get("publisher"),
                "source_url": record.get("url") or record.get("source_url"),
                "payload": record,
            })
    try:
        written = _request(config, "POST", "cak_narrative_events", params={"on_conflict": "event_id"}, payload=event_payload).json() if event_payload else []
        reports.append({"table": "cak_narrative_events", "rows_attempted": len(event_payload), "rows_written": len(written) if isinstance(written, list) else len(event_payload), "state": "WRITTEN", "detail": ""})
    except Exception as exc:
        reports.append({"table": "cak_narrative_events", "rows_attempted": len(event_payload), "rows_written": 0, "state": "WRITE_FAILED", "detail": str(exc)})
    return pd.DataFrame(reports, columns=columns)


def verify_scan(config: DatabaseConfig, *, scan_id: str, expected_radar: int, expected_events: int) -> pd.DataFrame:
    columns = ["table", "expected", "verified", "verification_pct", "state", "detail"]
    if not config.ready:
        return pd.DataFrame([{"table": "__SUMMARY__", "expected": 0, "verified": 0, "verification_pct": 0.0, "state": "DATABASE_DISABLED", "detail": "Database is not configured."}], columns=columns)
    checks = [("cak_scan_runs", 1), ("cak_radar_snapshots", expected_radar), ("cak_narrative_events", expected_events)]
    reports: list[dict[str, Any]] = []
    total_expected = 0
    total_verified = 0
    for table, expected in checks:
        total_expected += expected
        try:
            response = _request(config, "GET", table, params={"select": "scan_id", "scan_id": f"eq.{scan_id}"})
            rows = response.json()
            verified = len(rows) if isinstance(rows, list) else 0
            total_verified += min(verified, expected)
            pct = 100.0 if expected == 0 and verified == 0 else 100.0 * min(verified, expected) / max(expected, 1)
            reports.append({"table": table, "expected": expected, "verified": verified, "verification_pct": round(pct, 1), "state": "VERIFIED" if verified >= expected else "MISSING_ROWS", "detail": "Exact scan_id rows found."})
        except Exception as exc:
            reports.append({"table": table, "expected": expected, "verified": 0, "verification_pct": 0.0, "state": "READBACK_FAILED", "detail": str(exc)})
    pct = 100.0 * total_verified / max(total_expected, 1)
    state = "VERIFIED_ALL_TABLES" if total_verified == total_expected else "PARTIAL_READBACK"
    reports.insert(0, {"table": "__SUMMARY__", "expected": total_expected, "verified": total_verified, "verification_pct": round(pct, 1), "state": state, "detail": f"{total_verified}/{total_expected} rows verified."})
    return pd.DataFrame(reports, columns=columns)


__all__ = ["DatabaseConfig", "config_from_mapping", "persist_scan", "verify_scan"]
