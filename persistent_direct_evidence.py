from __future__ import annotations

"""Load durable, verified direct evidence for reuse in later Emir scans.

Rows remain evidence-specific and freshness-bounded. This loader never converts
missing regulatory states into a clean/false state and never promotes an
unverified row. Manual evidence in the current session may still override a
persisted row after this loader returns it.
"""

from typing import Any, Iterable

import pandas as pd

from persistence import DatabaseConfig, _request

PERSISTENT_DIRECT_EVIDENCE_VERSION = "1.0.0"

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
    rows: list[dict[str, Any]] = []
    safe_limit = max(1, int(limit))
    safe_page = max(1, min(1000, int(page_size)))

    for start in range(0, safe_limit, safe_page):
        try:
            response = _request(
                config,
                "GET",
                "cak_direct_evidence",
                params={
                    "select": "evidence_id,scan_id,ticker,evidence_type,observed_at,source_verified,payload",
                    "source_verified": "eq.true",
                    "order": "observed_at.desc",
                },
                extra_headers={"Range": f"{start}-{min(start + safe_page - 1, safe_limit - 1)}"},
                timeout=20,
            )
            payload = response.json()
        except Exception as exc:
            result["audit"] = pd.DataFrame([{
                "provider": "PERSISTENT_DIRECT_EVIDENCE",
                "status": "READ_FAIL_SOFT",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }])
            return result
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        if len(payload) < safe_page:
            break

    flattened: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in rows:
        if not bool(item.get("source_verified")):
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
        max_age = _MAX_AGE_DAYS[evidence_type]
        fresh = pd.notna(observed) and 0 <= age_days <= max_age
        audit.append({
            "ticker": ticker,
            "evidence_type": evidence_type,
            "status": "PERSISTED_VERIFIED_CURRENT" if fresh else "PERSISTED_VERIFIED_STALE",
            "observed_at": observed.isoformat() if pd.notna(observed) else "",
            "age_days": round(age_days, 1) if age_days != float("inf") else None,
        })
        if not fresh:
            continue
        record["observed_at"] = observed.isoformat()
        # Keep the latest unique source/evidence tuple only. Same ticker/type may
        # legitimately have multiple official forward events from different URLs.
        source_key = str(record.get("url") or record.get("source_url") or item.get("evidence_id") or "")
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
