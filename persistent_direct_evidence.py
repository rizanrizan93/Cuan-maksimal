from __future__ import annotations

"""Load durable, verified direct evidence for reuse in later Emir scans.

The production source of truth is ``cak_persistent_direct_evidence``. It has no
foreign key to a scan run, so pruning old scan snapshots cannot delete evidence.
The legacy ``cak_direct_evidence`` table is used only as a schema-compatibility
fallback when the master table itself is unavailable.

Rows remain evidence-specific and freshness-bounded. Missing regulatory states
are never converted into a clean/false state and unverified/revoked rows are
never promoted.
"""

from typing import Any, Iterable

import pandas as pd

from persistence import DatabaseConfig, _request

PERSISTENT_DIRECT_EVIDENCE_VERSION = "1.1.0-master-store"
_MASTER_TABLE = "cak_persistent_direct_evidence"
_LEGACY_TABLE = "cak_direct_evidence"

_MAX_AGE_DAYS = {
    "BROKER_INVENTORY": 35,
    "OWNERSHIP_FREE_FLOAT": 180,
    "ORDERBOOK_BID_OFFER": 5,
    "IDX_INTEGRITY_REGULATORY": 60,
    "OFFICIAL_FORWARD_EVENT": 540,
}


def _empty() -> dict[str, pd.DataFrame]:
    return {
        "broker": pd.DataFrame(),
        "ownership": pd.DataFrame(),
        "orderbook": pd.DataFrame(),
        "idx_integrity": pd.DataFrame(),
        "official_forward_events": pd.DataFrame(),
        "audit": pd.DataFrame(),
    }


def _read_rows(
    config: DatabaseConfig,
    *,
    table: str,
    limit: int,
    page_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    master = table == _MASTER_TABLE
    select = (
        "evidence_key,source_scan_id,ticker,evidence_type,observed_at,source_verified,"
        "source_url,payload,freshness_policy_days,revoked,last_seen_at"
        if master
        else "evidence_id,scan_id,ticker,evidence_type,observed_at,source_verified,payload"
    )
    for start in range(0, limit, page_size):
        params: dict[str, str] = {
            "select": select,
            "source_verified": "eq.true",
            "order": "observed_at.desc",
        }
        if master:
            params["revoked"] = "eq.false"
        response = _request(
            config,
            "GET",
            table,
            params=params,
            extra_headers={"Range": f"{start}-{min(start + page_size - 1, limit - 1)}"},
            timeout=20,
        )
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        if len(payload) < page_size:
            break
    return rows


def load_verified_direct_evidence(
    config: DatabaseConfig,
    tickers: Iterable[str] | None = None,
    *,
    as_of: Any = None,
    limit: int = 10_000,
    page_size: int = 500,
) -> dict[str, pd.DataFrame]:
    result = _empty()
    if not config.ready:
        return result

    wanted = {str(t).upper().strip() for t in (tickers or []) if str(t).strip()}
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    safe_limit = max(1, int(limit))
    safe_page = max(1, min(1000, int(page_size)))

    source_store = _MASTER_TABLE
    read_error = ""
    try:
        rows = _read_rows(config, table=_MASTER_TABLE, limit=safe_limit, page_size=safe_page)
    except Exception as exc:
        # Compatibility only: older deployments may not have the master table yet.
        # An empty master table is NOT a reason to fall back, because empty may be
        # intentional after revocation.
        read_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        source_store = _LEGACY_TABLE
        try:
            rows = _read_rows(config, table=_LEGACY_TABLE, limit=safe_limit, page_size=safe_page)
        except Exception as legacy_exc:
            result["audit"] = pd.DataFrame([{
                "provider": "PERSISTENT_DIRECT_EVIDENCE",
                "status": "READ_FAIL_SOFT",
                "source_store": source_store,
                "error": f"master={read_error}; legacy={type(legacy_exc).__name__}: {str(legacy_exc)[:300]}",
            }])
            return result

    flattened: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    if read_error:
        audit.append({
            "provider": "PERSISTENT_DIRECT_EVIDENCE",
            "status": "MASTER_UNAVAILABLE_LEGACY_FALLBACK",
            "source_store": source_store,
            "error": read_error,
        })

    seen: set[tuple[str, str, str]] = set()
    for item in rows:
        if not bool(item.get("source_verified")) or bool(item.get("revoked", False)):
            continue
        ticker = str(item.get("ticker") or "").upper().strip()
        evidence_type = str(item.get("evidence_type") or "").upper().strip()
        if not ticker or evidence_type not in _MAX_AGE_DAYS or (wanted and ticker not in wanted):
            continue

        record = dict(item.get("payload") or {}) if isinstance(item.get("payload"), dict) else {}
        record["ticker"] = ticker
        record["evidence_type"] = evidence_type
        record["source_verified"] = True
        observed_raw = item.get("observed_at") or record.get("observed_at") or record.get("published_at") or record.get("date")
        observed = pd.to_datetime(observed_raw, errors="coerce", utc=True)
        age_days = (now - observed).total_seconds() / 86400.0 if pd.notna(observed) else float("inf")
        configured_age = item.get("freshness_policy_days")
        try:
            max_age = int(configured_age) if configured_age is not None else _MAX_AGE_DAYS[evidence_type]
        except (TypeError, ValueError):
            max_age = _MAX_AGE_DAYS[evidence_type]
        max_age = max(1, min(max_age, _MAX_AGE_DAYS[evidence_type]))
        fresh = pd.notna(observed) and 0 <= age_days <= max_age

        evidence_key = str(item.get("evidence_key") or item.get("evidence_id") or "")
        audit.append({
            "ticker": ticker,
            "evidence_type": evidence_type,
            "provider": "PERSISTENT_DIRECT_EVIDENCE",
            "status": "PERSISTED_VERIFIED_CURRENT" if fresh else "PERSISTED_VERIFIED_STALE",
            "source_store": source_store,
            "evidence_key": evidence_key,
            "observed_at": observed.isoformat() if pd.notna(observed) else "",
            "age_days": round(age_days, 1) if age_days != float("inf") else None,
            "freshness_policy_days": max_age,
        })
        if not fresh:
            continue

        record["observed_at"] = observed.isoformat()
        record["persistent_evidence_key"] = evidence_key
        record["persistent_evidence_store"] = source_store
        record["source_scan_id"] = item.get("source_scan_id") or item.get("scan_id") or ""
        if not record.get("source_url") and item.get("source_url"):
            record["source_url"] = item.get("source_url")

        # Same ticker/type may legitimately have multiple official forward events
        # from different issuer URLs. Other direct evidence types use latest-only.
        source_key = str(record.get("url") or record.get("source_url") or evidence_key)
        key = (ticker, evidence_type, source_key if evidence_type == "OFFICIAL_FORWARD_EVENT" else evidence_type)
        if key in seen:
            continue
        seen.add(key)
        flattened.append(record)

    frame = pd.DataFrame(flattened)
    if not frame.empty:
        mapping = {
            "BROKER_INVENTORY": "broker",
            "OWNERSHIP_FREE_FLOAT": "ownership",
            "ORDERBOOK_BID_OFFER": "orderbook",
            "IDX_INTEGRITY_REGULATORY": "idx_integrity",
            "OFFICIAL_FORWARD_EVENT": "official_forward_events",
        }
        for evidence_type, key in mapping.items():
            result[key] = frame.loc[frame["evidence_type"] == evidence_type].copy().reset_index(drop=True)
    result["audit"] = pd.DataFrame(audit)
    return result


__all__ = ["PERSISTENT_DIRECT_EVIDENCE_VERSION", "load_verified_direct_evidence"]