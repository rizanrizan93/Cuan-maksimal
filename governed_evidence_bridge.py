from __future__ import annotations

"""Fail-closed bridge from governed evidence tables into Emir event inputs.

The raw tables remain factual stores. This bridge never turns a board roster,
RUPS, ownership roster, or ordinary dividend into a bullish thesis. Only strict
forward evidence and explicitly directional capital/insider actions are eligible
for scoring; administrative evidence remains available for audit.
"""

from typing import Any, Iterable, Mapping
from urllib.parse import urlparse
import math

import numpy as np
import pandas as pd

from data_providers import normalize_ticker
from persistence import DatabaseConfig, _request

GOVERNED_EVIDENCE_BRIDGE_VERSION = "1.0.0-fail-closed"


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "verified", "on"}


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _https(value: Any) -> bool:
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    return parsed.scheme.lower() == "https" and bool(parsed.hostname)


def _timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    return pd.Timestamp(stamp) if pd.notna(stamp) else pd.NaT


def _strict(row: Mapping[str, Any], *, as_of: Any = None, max_age_days: int = 540) -> bool:
    if not _truthy(row.get("source_verified")):
        return False
    if not _truthy(row.get("source_quorum_verified")) or _finite(row.get("source_quorum_count"), 0) < 2:
        return False
    if not _truthy(row.get("entity_match_verified")):
        return False
    if not _https(row.get("source_url")):
        return False
    stamp = _timestamp(row.get("evidence_date") or row.get("observed_at"))
    if pd.isna(stamp):
        return False
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    age = (now - stamp).total_seconds() / 86400.0
    return -1.0 <= age <= float(max_age_days)


def persistent_forward_payload_is_strict(record: Mapping[str, Any], *, source_url: Any = None) -> bool:
    """Validate legacy/master OFFICIAL_FORWARD_EVENT payloads before reuse."""
    url = record.get("source_url") or record.get("url") or source_url
    return bool(
        _truthy(record.get("source_verified", True))
        and _truthy(record.get("source_quorum_verified"))
        and _finite(record.get("source_quorum_count"), 0) >= 2
        and _truthy(record.get("entity_match_verified"))
        and _https(url)
    )


def _read_strict_rows(
    config: DatabaseConfig,
    table: str,
    *,
    limit: int = 5000,
    page_size: int = 500,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not config.ready:
        return [], []
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    safe_limit = max(1, int(limit))
    safe_page = max(1, min(1000, int(page_size)))
    try:
        for start in range(0, safe_limit, safe_page):
            response = _request(
                config,
                "GET",
                table,
                params={
                    "select": "*",
                    "source_verified": "eq.true",
                    "source_quorum_verified": "eq.true",
                    "entity_match_verified": "eq.true",
                    "source_quorum_count": "gte.2",
                    "order": "observed_at.desc",
                },
                extra_headers={"Range": f"{start}-{min(start + safe_page - 1, safe_limit - 1)}"},
                timeout=20,
            )
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break
            rows.extend(payload)
            if len(payload) < safe_page:
                break
        audit.append({
            "provider": "GOVERNED_EVIDENCE_BRIDGE",
            "status": "DATABASE_CURRENT" if rows else "NO_ITEMS",
            "source_store": table,
            "rows": len(rows),
        })
    except Exception as exc:
        audit.append({
            "provider": "GOVERNED_EVIDENCE_BRIDGE",
            "status": "READ_FAIL_SOFT",
            "source_store": table,
            "rows": 0,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        })
    return rows, audit


def _source_tier(source_family: Any, source_url: Any) -> str:
    family = str(source_family or "").upper()
    host = (urlparse(str(source_url or "")).hostname or "").lower()
    if "REGULATOR" in family or host.endswith("go.id"):
        return "REGULATOR"
    if any(token in family for token in ("ISSUER", "PRESENTATION", "GOVERNANCE")):
        return "ISSUER"
    return "OFFICIAL"


def _forward_materiality(evidence_type: str) -> tuple[float, float]:
    text = str(evidence_type or "").upper()
    if any(token in text for token in ("BACKLOG", "CONTRACT", "OFFTAKE", "ORDER_VISIBILITY")):
        return 82.0, 78.0
    if any(token in text for token in ("CAPEX", "EXPANSION", "CAPACITY")):
        return 76.0, 70.0
    if "GUIDANCE" in text:
        return 72.0, 65.0
    if any(token in text for token in ("PRODUCT", "LAUNCH", "NEW_MARKET")):
        return 68.0, 58.0
    return 60.0, 52.0


def forward_rows_to_events(rows: Iterable[Mapping[str, Any]], *, as_of: Any = None) -> pd.DataFrame:
    events: list[dict[str, Any]] = []
    for row in rows:
        if not _strict(row, as_of=as_of, max_age_days=540):
            continue
        ticker = normalize_ticker(row.get("ticker"))
        if not ticker:
            continue
        evidence_type = str(row.get("evidence_type") or "FORWARD_EVIDENCE").upper()
        materiality, bridge = _forward_materiality(evidence_type)
        payload = dict(row.get("payload") or {}) if isinstance(row.get("payload"), dict) else {}
        title = str(row.get("title") or evidence_type.replace("_", " ")).strip()
        details: list[str] = []
        for key in ("guidance", "orders_through", "capacity_state", "milestone_state"):
            value = payload.get(key)
            if value not in (None, ""):
                details.append(f"{key}={value}")
        if row.get("value_numeric") not in (None, ""):
            details.append(f"value={row.get('value_numeric')} {row.get('unit') or ''}".strip())
        if row.get("horizon") not in (None, ""):
            details.append(f"horizon={row.get('horizon')}")
        evidence_date = _timestamp(row.get("evidence_date") or row.get("observed_at"))
        source_url = str(row.get("source_url") or "")
        events.append({
            "ticker": ticker,
            "published_at": evidence_date.isoformat(),
            "event_date": evidence_date.date().isoformat(),
            "title": title,
            "summary": " | ".join([title, *details]),
            "publisher": str(row.get("source_family") or "GOVERNED_OFFICIAL_EVIDENCE"),
            "url": source_url,
            "source_tier": _source_tier(row.get("source_family"), source_url),
            "materiality_score": materiality,
            "financial_bridge_score": bridge,
            "top_down_catalyst_score": np.nan,
            "industry_translation_score": np.nan,
            "issuer_alignment_score": np.nan,
            "category": evidence_type,
            "event_role": "FORWARD_FUNDAMENTAL",
            "narrative_eligible": True,
            "collection_provider": "GOVERNED_FORWARD_EVIDENCE",
            "source_verified": True,
            "source_quorum_verified": True,
            "source_quorum_count": int(_finite(row.get("source_quorum_count"), 0)),
            "entity_match_verified": True,
            "evidence_confidence": _finite(row.get("evidence_confidence"), np.nan),
            "governed_evidence_bridge_version": GOVERNED_EVIDENCE_BRIDGE_VERSION,
            "governed_evidence_state": "STRICT_FORWARD_EVIDENCE_CONSUMED",
        })
    return pd.DataFrame(events)


def _management_event_policy(evidence_type: str, action: str) -> tuple[str, bool, str, float, float]:
    text = f"{evidence_type} {action}".upper()
    if "BUYBACK" in text:
        return "BUYBACK", True, "CAPITAL_ACTION_VERIFIED", 72.0, 55.0
    if any(token in text for token in ("RIGHTS_ISSUE", "RIGHTS ISSUE", "HMETD", "DILUTION", "PRIVATE_PLACEMENT")):
        return "DILUTION_EQUITY_RAISE", True, "CAPITAL_ACTION_VERIFIED", 82.0, 75.0
    if any(token in text for token in ("INSIDER_BUY", "CONTROLLER_BUY")):
        return "INSIDER_BUY", True, "CAPITAL_ACTION_VERIFIED", 78.0, 45.0
    if any(token in text for token in ("INSIDER_SELL", "CONTROLLER_SELL")):
        return "INSIDER_SELL", True, "CAPITAL_ACTION_VERIFIED", 82.0, 45.0
    if any(token in text for token in ("CAPEX_DECISION", "CAPEX APPROVAL", "CAPACITY_EXPANSION")):
        return "CAPEX_DECISION", True, "FORWARD_FUNDAMENTAL", 75.0, 70.0
    if any(token in text for token in ("DIVIDEND", "TREASURY_SHARE", "RUPS", "BOARD_ROLE", "BOARD_CHANGE", "OWNERSHIP")):
        return str(evidence_type or "ADMINISTRATIVE_EVIDENCE").upper(), False, "GOVERNANCE_ADMINISTRATIVE", 50.0, 25.0
    return str(evidence_type or "MANAGEMENT_CAPITAL_EVIDENCE").upper(), False, "GOVERNANCE_ADMINISTRATIVE", 45.0, 20.0


def management_rows_to_events(rows: Iterable[Mapping[str, Any]], *, as_of: Any = None) -> pd.DataFrame:
    events: list[dict[str, Any]] = []
    for row in rows:
        if not _strict(row, as_of=as_of, max_age_days=730):
            continue
        ticker = normalize_ticker(row.get("ticker"))
        if not ticker:
            continue
        evidence_type = str(row.get("evidence_type") or "MANAGEMENT_CAPITAL_EVIDENCE").upper()
        action = str(row.get("role_or_action") or "").upper()
        category, eligible, role, materiality, bridge = _management_event_policy(evidence_type, action)
        person = str(row.get("person_or_holder") or "").strip()
        title_parts = [part for part in (evidence_type.replace("_", " "), person, action.replace("_", " ")) if part]
        title = " - ".join(title_parts)
        evidence_date = _timestamp(row.get("evidence_date") or row.get("observed_at"))
        source_url = str(row.get("source_url") or "")
        events.append({
            "ticker": ticker,
            "published_at": evidence_date.isoformat(),
            "event_date": evidence_date.date().isoformat(),
            "title": title,
            "summary": title,
            "publisher": str(row.get("source_family") or "GOVERNED_MANAGEMENT_CAPITAL_EVIDENCE"),
            "url": source_url,
            "source_tier": _source_tier(row.get("source_family"), source_url),
            "materiality_score": materiality,
            "financial_bridge_score": bridge,
            "category": category,
            "event_role": role,
            "narrative_eligible": bool(eligible),
            "collection_provider": "GOVERNED_MANAGEMENT_CAPITAL_EVIDENCE",
            "source_verified": True,
            "source_quorum_verified": True,
            "source_quorum_count": int(_finite(row.get("source_quorum_count"), 0)),
            "entity_match_verified": True,
            "evidence_confidence": _finite(row.get("evidence_confidence"), np.nan),
            "governed_evidence_bridge_version": GOVERNED_EVIDENCE_BRIDGE_VERSION,
            "governed_evidence_state": "STRICT_DIRECTIONAL_EVENT_CONSUMED" if eligible else "STRICT_ADMINISTRATIVE_EVIDENCE_AUDIT_ONLY",
        })
    return pd.DataFrame(events)


def load_governed_evidence(
    config: DatabaseConfig,
    tickers: Iterable[str] | None = None,
    *,
    as_of: Any = None,
) -> dict[str, pd.DataFrame]:
    wanted = {normalize_ticker(value) for value in (tickers or []) if normalize_ticker(value)}
    forward_rows, forward_audit = _read_strict_rows(config, "cak_forward_evidence")
    management_rows, management_audit = _read_strict_rows(config, "cak_management_capital_evidence")
    if wanted:
        forward_rows = [row for row in forward_rows if normalize_ticker(row.get("ticker")) in wanted]
        management_rows = [row for row in management_rows if normalize_ticker(row.get("ticker")) in wanted]
    forward_events = forward_rows_to_events(forward_rows, as_of=as_of)
    management_events = management_rows_to_events(management_rows, as_of=as_of)
    audit = pd.DataFrame([*forward_audit, *management_audit])
    if not audit.empty:
        audit["governed_evidence_bridge_version"] = GOVERNED_EVIDENCE_BRIDGE_VERSION
    return {
        "official_forward_events": forward_events,
        "management_capital_events": management_events,
        "audit": audit,
    }


__all__ = [
    "GOVERNED_EVIDENCE_BRIDGE_VERSION",
    "forward_rows_to_events",
    "load_governed_evidence",
    "management_rows_to_events",
    "persistent_forward_payload_is_strict",
]
