from __future__ import annotations

"""Use verified KSEI issuer names to improve scan-time forward entity matching."""

from functools import wraps
from typing import Any, Iterable

import pandas as pd

from live_forward_evidence import collect_live_forward_evidence

PATCH_VERSION = "1.0.0"


def _symbols(values: Iterable[Any]) -> list[str]:
    from data_providers import normalize_ticker
    return list(dict.fromkeys(normalize_ticker(value) for value in values if normalize_ticker(value)))


def _fresh_forward(cache_module: Any, config: Any, names: list[str]) -> set[str]:
    try:
        rows = cache_module.read_source_cache(config, names, "FORWARD_RESEARCH")
    except Exception:
        return set()
    now = pd.Timestamp.now(tz="UTC")
    fresh: set[str] = set()
    for ticker, row in rows.items():
        try:
            if not cache_module._row_hash_valid(row):
                continue
        except Exception:
            continue
        valid_until = pd.to_datetime(row.get("valid_until"), errors="coerce", utc=True)
        if pd.notna(valid_until) and valid_until >= now:
            symbols = _symbols([ticker])
            if symbols:
                fresh.add(symbols[0])
    return fresh


def _company_names_from_ksei(cache_module: Any, config: Any, names: list[str]) -> dict[str, str]:
    try:
        rows = cache_module.read_source_cache(config, names, "KSEI")
    except Exception:
        return {}
    output: dict[str, str] = {}
    for ticker, row in rows.items():
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        profiles = payload.get("profiles") if isinstance(payload.get("profiles"), list) else []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            company = str(profile.get("company_name") or "").strip()
            symbol = _symbols([profile.get("ticker") or ticker])
            if company and symbol:
                output[symbol[0]] = company
                break
    return output


def install(resumable_module: Any, cache_module: Any, integrity_module: Any) -> None:
    original = getattr(resumable_module, "load_cached_news", None)
    if not callable(original) or getattr(original, "__company_identity_forward_v1__", False):
        return

    @wraps(original)
    def wrapped(config: Any, tickers: Iterable[str]):
        names = _symbols(tickers)
        if names and getattr(config, "ready", False):
            fresh = _fresh_forward(cache_module, config, names)
            missing = [ticker for ticker in names if ticker not in fresh]
            if missing:
                company_names = _company_names_from_ksei(cache_module, config, missing)
                events, audit = collect_live_forward_evidence(
                    missing,
                    company_names=company_names,
                    lookback_days=180,
                    max_workers=12,
                    timeout=5.0,
                )
                if isinstance(audit, pd.DataFrame) and not audit.empty:
                    try:
                        integrity_module._persist_forward_research(
                            config,
                            "LIVE_FORWARD_COMPANY_ENTITY",
                            events if isinstance(events, pd.DataFrame) else pd.DataFrame(),
                            audit,
                            cache_module,
                        )
                    except Exception:
                        pass
        return original(config, names)

    wrapped.__company_identity_forward_v1__ = True
    setattr(resumable_module, "load_cached_news", wrapped)


def install_runtime() -> None:
    try:
        import persistent_cache
        import resumable_scan
        import runtime_integrity_patch
    except Exception:
        return
    install(resumable_scan, persistent_cache, runtime_integrity_patch)


__all__ = ["PATCH_VERSION", "install", "install_runtime"]
