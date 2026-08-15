from __future__ import annotations

"""Prevent false resumable-scan pauses when the KSEI DB guard preserves OLD.

The database intentionally refuses to replace a durable KSEI cache row with an
unresolved/placeholder profile.  PostgREST can still report the upsert request
as successful, while exact SHA readback correctly sees the preserved OLD row.
That is an expected guard outcome, not a cache-corruption event.

This runtime patch mirrors the database validity contract.  When a refresh would
be rejected by that guard and a hash-valid durable cache row already exists, the
scanner reuses the durable row and suppresses only that rejected write.  All
other cache writes and exact SHA verification remain strict.
"""

from functools import wraps
import importlib
from typing import Any, Mapping

import pandas as pd

PATCH_VERSION = "1.0.0"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "verified"}


def _positive_number(value: Any) -> bool:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(parsed) and float(parsed) > 0.0)


def _ksei_profile_is_db_valid(payload: Any) -> bool:
    """Mirror ``guard_ksei_source_cache_profile`` for one KSEI cache payload."""
    if not isinstance(payload, Mapping):
        return False
    profiles = payload.get("profiles")
    profile = profiles[0] if isinstance(profiles, list) and profiles and isinstance(profiles[0], Mapping) else {}
    company_name = str(profile.get("company_name") or "").strip().lower()
    security_status = str(profile.get("security_status") or "").strip().upper()
    claimed_verified = _truthy(profile.get("ksei_source_verified"))
    return bool(
        claimed_verified
        and company_name not in {"", "isin code", "security name", "issuer", "undefined", "null", "none"}
        and _positive_number(profile.get("total_shares"))
        and security_status
        and not security_status.startswith("UNKNOWN")
    )


def _write_would_be_guarded(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    if str(row.get("family") or "").upper() != "KSEI":
        return False
    return not _ksei_profile_is_db_valid(row.get("payload"))


def _replace_ticker_records(frame: pd.DataFrame, ticker: str, records: Any) -> pd.DataFrame:
    """Replace one ticker's provider records with the durable cache records."""
    local = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if not local.empty and "ticker" in local.columns:
        local = local[local["ticker"].astype(str) != str(ticker)].copy()
    replacement = pd.DataFrame(records if isinstance(records, list) else [])
    if not replacement.empty:
        if "ticker" not in replacement.columns:
            replacement["ticker"] = str(ticker)
        else:
            replacement["ticker"] = replacement["ticker"].fillna(str(ticker)).astype(str)
        local = pd.concat([local, replacement], ignore_index=True, sort=False) if not local.empty else replacement
    return local.reset_index(drop=True)


def _mark_guard_fallback(audit: pd.DataFrame, ticker: str) -> pd.DataFrame:
    local = audit.copy() if isinstance(audit, pd.DataFrame) else pd.DataFrame()
    detail = (
        "KSEI refresh returned an unresolved/placeholder profile; "
        "database guard preserves the existing hash-valid cache row."
    )
    if local.empty:
        return pd.DataFrame([{
            "ticker": str(ticker),
            "provider": "SUPABASE_KSEI_CACHE",
            "status": "STALE_CACHE_FALLBACK_GUARD",
            "cache_state": "STALE_CACHE_FALLBACK_GUARD",
            "detail": detail,
        }])
    if "ticker" not in local.columns:
        local["ticker"] = ""
    mask = local["ticker"].astype(str).eq(str(ticker))
    if not bool(mask.any()):
        extra = {column: None for column in local.columns}
        extra.update({
            "ticker": str(ticker),
            "provider": "SUPABASE_KSEI_CACHE",
            "status": "STALE_CACHE_FALLBACK_GUARD",
            "cache_state": "STALE_CACHE_FALLBACK_GUARD",
            "detail": detail,
        })
        local = pd.concat([local, pd.DataFrame([extra])], ignore_index=True, sort=False)
        return local
    for column, value in {
        "provider": "SUPABASE_KSEI_CACHE",
        "status": "STALE_CACHE_FALLBACK_GUARD",
        "cache_state": "STALE_CACHE_FALLBACK_GUARD",
        "detail": detail,
    }.items():
        if column not in local.columns:
            local[column] = ""
        local.loc[mask, column] = value
    return local


def _build_guarded_fetch(cache_module: Any, original_fetch: Any):
    @wraps(original_fetch)
    def guarded_fetch(
        config: Any,
        tickers: Any,
        *,
        max_workers: int = 4,
        now: Any = None,
        force_refresh: bool = False,
        last_scan_id: str = "",
    ):
        profiles, actions, audit, writes = original_fetch(
            config,
            tickers,
            max_workers=max_workers,
            now=now,
            force_refresh=force_refresh,
            last_scan_id=last_scan_id,
        )
        writes = list(writes or [])
        guarded_rows = [
            row for row in writes
            if isinstance(row, Mapping) and _write_would_be_guarded(row)
        ]
        if not guarded_rows or not bool(getattr(config, "ready", False)):
            return profiles, actions, audit, writes

        guarded_tickers = list(dict.fromkeys(
            str(row.get("ticker") or "") for row in guarded_rows if str(row.get("ticker") or "")
        ))
        if not guarded_tickers:
            return profiles, actions, audit, writes

        try:
            cached = cache_module.read_source_cache(config, guarded_tickers, "KSEI")
        except Exception:
            # A cache-read failure is not silently downgraded.  Keep the original
            # write so the existing exact verification/fail-safe path still runs.
            return profiles, actions, audit, writes

        preserved: set[str] = set()
        for ticker in guarded_tickers:
            existing = cached.get(ticker) if isinstance(cached, Mapping) else None
            if not isinstance(existing, Mapping) or not cache_module._row_hash_valid(existing):
                continue
            payload = existing.get("payload") if isinstance(existing.get("payload"), Mapping) else {}
            old_profiles = payload.get("profiles") if isinstance(payload.get("profiles"), list) else []
            old_actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
            if not old_profiles and not old_actions:
                continue

            profiles = _replace_ticker_records(profiles, ticker, old_profiles)
            actions = _replace_ticker_records(actions, ticker, old_actions)
            audit = _mark_guard_fallback(audit, ticker)
            preserved.add(ticker)

        if preserved:
            writes = [
                row for row in writes
                if not (
                    isinstance(row, Mapping)
                    and str(row.get("family") or "").upper() == "KSEI"
                    and str(row.get("ticker") or "") in preserved
                    and _write_would_be_guarded(row)
                )
            ]
        return profiles, actions, audit, writes

    guarded_fetch.__ksei_checkpoint_guard_v1__ = True
    return guarded_fetch


def install() -> None:
    """Patch both cache module and resumable-scan imported function reference."""
    try:
        cache_module = importlib.import_module("persistent_cache")
        scan_module = importlib.import_module("resumable_scan")
    except Exception:
        return

    current = getattr(cache_module, "fetch_ksei_cache_first", None)
    if not callable(current):
        return
    if getattr(current, "__ksei_checkpoint_guard_v1__", False):
        setattr(scan_module, "fetch_ksei_cache_first", current)
        return

    guarded = _build_guarded_fetch(cache_module, current)
    setattr(cache_module, "fetch_ksei_cache_first", guarded)
    setattr(scan_module, "fetch_ksei_cache_first", guarded)


__all__ = [
    "PATCH_VERSION",
    "_ksei_profile_is_db_valid",
    "_write_would_be_guarded",
    "_build_guarded_fetch",
    "install",
]
