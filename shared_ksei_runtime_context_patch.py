from __future__ import annotations

"""Complete EMIR KSEI profiles from the canonical Shared Evidence Hub.

The monthly KSEI archive is already ingested once into the Shared Hub by the
shared evidence pipeline.  EMIR should consume that canonical persisted copy
rather than depend on another live archive download during every scan.  Existing
higher-resolution per-security KSEI facts remain authoritative; canonical
monthly facts only complete missing/incoherent scripless/local/foreign fields.

KSEI registration composition is not regulatory free float and not beneficial
ownership.  No score, rank, gate, recommendation, or authorization changes live
here.
"""

from functools import wraps
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend
import ksei_monthly_field_completion_patch as completion

PATCH_VERSION = "1.0.0-shared-canonical-ksei-emir"
CATEGORY = "ksei-komposisi"
TABLE = "evidence_ownership_snapshots"
CACHE_TTL_SECONDS = 12 * 60 * 60
_CACHE: pd.DataFrame = pd.DataFrame()
_CACHE_AT = 0.0


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _latest_canonical_profiles(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    by_ticker: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if str(row.get("category") or "") != CATEGORY:
            continue
        if not bool(row.get("source_verified")):
            continue
        if str(row.get("validation_state") or "").upper() != "VALID":
            continue
        ticker = _ticker(row.get("ticker"))
        classification = str(row.get("holder_classification") or "")
        if not ticker or classification not in {
            "KSEI_SECURITY_NUMBER", "KSEI_SCRIPLESS_TOTAL", "KSEI_LOCAL_TOTAL", "KSEI_FOREIGN_TOTAL"
        }:
            continue
        report_date = str(row.get("report_date") or "")
        current = by_ticker.setdefault(ticker, {"report_date": report_date, "facts": {}, "source_url": ""})
        if report_date > str(current.get("report_date") or ""):
            current.update({"report_date": report_date, "facts": {}, "source_url": ""})
        if report_date == str(current.get("report_date") or ""):
            current["facts"][classification] = row
            if row.get("source_url"):
                current["source_url"] = str(row.get("source_url"))

    output: list[dict[str, Any]] = []
    for ticker, bundle in by_ticker.items():
        facts = bundle["facts"]
        issued = _finite(facts.get("KSEI_SECURITY_NUMBER", {}).get("shares_held"))
        scripless_pct = _finite(facts.get("KSEI_SCRIPLESS_TOTAL", {}).get("ownership_percentage"))
        local_pct = _finite(facts.get("KSEI_LOCAL_TOTAL", {}).get("ownership_percentage"))
        foreign_pct = _finite(facts.get("KSEI_FOREIGN_TOTAL", {}).get("ownership_percentage"))
        output.append({
            "ticker": ticker,
            "total_shares": issued if np.isfinite(issued) else np.nan,
            "scripless_pct": scripless_pct if np.isfinite(scripless_pct) else np.nan,
            "local_pct": local_pct if np.isfinite(local_pct) else np.nan,
            "foreign_pct": foreign_pct if np.isfinite(foreign_pct) else np.nan,
            "ksei_source_url": str(bundle.get("source_url") or ""),
            "ksei_source_verified": True,
            "ksei_observed_on": str(bundle.get("report_date") or ""),
            "ksei_monthly_holding_composition_state": completion.COMPOSITION_STATE,
            "ksei_composition_completion_state": "SHARED_CANONICAL_MONTHLY_CONTEXT",
        })
    return pd.DataFrame(output)


def read_shared_profiles(tickers: Iterable[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    global _CACHE, _CACHE_AT
    wanted = {_ticker(value) for value in (tickers or []) if _ticker(value)}
    now = time.monotonic()
    if isinstance(_CACHE, pd.DataFrame) and not _CACHE.empty and now - _CACHE_AT <= CACHE_TTL_SECONDS:
        local = _CACHE if not wanted else _CACHE.loc[_CACHE["ticker"].isin(wanted)]
        return local.copy().reset_index(drop=True), {"state": "PROCESS_CACHE_HIT", "rows": len(local)}

    config = HubConfig.from_environment(client_id="EMIR")
    if not config.ready:
        return pd.DataFrame(), {"state": "SHARED_HUB_UNAVAILABLE", **config.status()}
    try:
        rows = SupabaseEvidenceBackend(config).read_rows(TABLE, {}, limit=50000)
    except Exception as exc:
        return pd.DataFrame(), {"state": "READ_FAIL_SOFT", "error": f"{type(exc).__name__}: {str(exc)[:240]}"}
    profiles = _latest_canonical_profiles(rows)
    if not profiles.empty:
        _CACHE = profiles.copy()
        _CACHE_AT = now
    local = profiles if not wanted else profiles.loc[profiles["ticker"].isin(wanted)]
    return local.copy().reset_index(drop=True), {
        "state": "SHARED_CANONICAL_KSEI",
        "rows": len(local),
        "semantics": "OFFICIAL_KSEI_REGISTRATION_COMPOSITION_NOT_REGULATORY_FREE_FLOAT",
    }


def _wrap_fetch_many(module: Any) -> None:
    original = getattr(module, "fetch_many_ksei_profiles", None)
    if not callable(original) or getattr(original, "__shared_canonical_ksei_v1__", False):
        return

    @wraps(original)
    def wrapped(tickers: Iterable[str], max_workers: int = 2):
        requested = list(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value)))
        profiles, actions, audit = original(requested, max_workers=max_workers)
        frame = profiles.copy() if isinstance(profiles, pd.DataFrame) else pd.DataFrame()
        shared, shared_audit = read_shared_profiles(requested)
        completed, changed = completion._supplement_profiles(frame, requested, shared)

        audit_frames = [item for item in (audit,) if isinstance(item, pd.DataFrame) and not item.empty]
        audit_frames.append(pd.DataFrame([{
            "provider": "SHARED_CANONICAL_KSEI",
            "status": str(shared_audit.get("state") or "UNKNOWN"),
            "items": int(changed),
            "rows": int(shared_audit.get("rows") or 0),
            "detail": str(shared_audit.get("semantics") or shared_audit.get("error") or ""),
        }]))
        audit_out = pd.concat(audit_frames, ignore_index=True, sort=False)
        return completed, actions, audit_out

    wrapped.__shared_canonical_ksei_v1__ = True
    setattr(module, "fetch_many_ksei_profiles", wrapped)


def install() -> dict[str, str]:
    import autonomous_enrichment
    import persistent_cache

    _wrap_fetch_many(autonomous_enrichment)
    _wrap_fetch_many(persistent_cache)
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "source": "SHARED_CANONICAL_KSEI_MONTHLY_CONTEXT",
        "existing_per_security": "PRESERVED_WHEN_COMPLETE",
        "regulatory_free_float": "NOT_INFERRED",
        "beneficial_ownership": "NOT_INFERRED",
        "authorization": "UNCHANGED",
    }


__all__ = ["PATCH_VERSION", "install", "read_shared_profiles", "_latest_canonical_profiles"]
