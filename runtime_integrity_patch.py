from __future__ import annotations

"""Hot-reload-safe evidence/ranking/runtime integrity hooks for Emir scanner."""

from functools import wraps
from typing import Any, Iterable
import pandas as pd

from evidence_governance import ProviderNegativeCache, apply_three_rank_contract

PATCH_VERSION = "1.0.0"
_NEGATIVE_CACHE = ProviderNegativeCache()


def _status_failure_class(value: Any) -> str:
    text = str(value or "").upper()
    if "429" in text or "RATE" in text:
        return "RATE_LIMIT"
    if "TIMEOUT" in text or "TIMED OUT" in text:
        return "TIMEOUT"
    if "404" in text or "NOT_FOUND" in text or "NOT FOUND" in text:
        return "NOT_FOUND"
    if "403" in text or "401" in text or "AUTH" in text:
        return "AUTH"
    if "PARSE" in text or "DECODE" in text:
        return "PARSE"
    if "500" in text or "502" in text or "503" in text or "SERVER" in text:
        return "SERVER"
    if "EMPTY" in text or "NO_DATA" in text or "NO DATA" in text:
        return "EMPTY"
    return "OTHER"


def _symbols(values: Iterable[Any]) -> list[str]:
    from data_providers import normalize_ticker
    return list(dict.fromkeys(normalize_ticker(value) for value in values if normalize_ticker(value)))


def _append_skip_audit(audit: pd.DataFrame, skipped: list[str], provider: str) -> pd.DataFrame:
    if not skipped:
        return audit
    extra = pd.DataFrame({
        "ticker": skipped,
        "provider": provider,
        "status": "NEGATIVE_CACHE_SKIP",
        "items": 0,
        "detail": "Provider-specific transient/negative result cached; retry deferred by TTL.",
    })
    if audit is None or audit.empty:
        return extra
    return pd.concat([audit, extra], ignore_index=True, sort=False)


def _wrap_fundamentals(module: Any) -> None:
    original = getattr(module, "fetch_many_fundamentals", None)
    if not callable(original) or getattr(original, "__negative_cache_v1__", False):
        return

    @wraps(original)
    def wrapped(tickers: Iterable[str], max_workers: int = 3):
        requested = _symbols(tickers)
        allowed = [t for t in requested if not _NEGATIVE_CACHE.should_skip("YFINANCE", "FUNDAMENTAL", t)]
        skipped = [t for t in requested if t not in allowed]
        if allowed:
            snapshots, audit = original(allowed, max_workers=max_workers)
        else:
            snapshots, audit = pd.DataFrame(), pd.DataFrame()
        success = set()
        if isinstance(snapshots, pd.DataFrame) and not snapshots.empty and "ticker" in snapshots.columns:
            success = set(_symbols(snapshots["ticker"].tolist()))
        for ticker in allowed:
            if ticker in success:
                _NEGATIVE_CACHE.record_success("YFINANCE", "FUNDAMENTAL", ticker)
                continue
            detail = ""
            if isinstance(audit, pd.DataFrame) and not audit.empty and "ticker" in audit.columns:
                rows = audit[audit["ticker"].astype(str).str.upper().eq(ticker.upper())]
                if not rows.empty:
                    detail = " ".join(rows.astype(str).tail(1).iloc[0].tolist())
            _NEGATIVE_CACHE.record_failure("YFINANCE", "FUNDAMENTAL", ticker, _status_failure_class(detail))
        return snapshots, _append_skip_audit(audit, skipped, "YFINANCE_FUNDAMENTAL_NEGATIVE_CACHE")

    wrapped.__negative_cache_v1__ = True
    setattr(module, "fetch_many_fundamentals", wrapped)


def _wrap_ksei(module: Any) -> None:
    original = getattr(module, "fetch_many_ksei_profiles", None)
    if not callable(original) or getattr(original, "__negative_cache_v1__", False):
        return

    @wraps(original)
    def wrapped(tickers: Iterable[str], max_workers: int = 2):
        requested = _symbols(tickers)
        allowed = [t for t in requested if not _NEGATIVE_CACHE.should_skip("KSEI", "SECURITY_PROFILE", t)]
        skipped = [t for t in requested if t not in allowed]
        if allowed:
            profiles, actions, audit = original(allowed, max_workers=max_workers)
        else:
            profiles, actions, audit = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        success = set()
        if isinstance(profiles, pd.DataFrame) and not profiles.empty and "ticker" in profiles.columns:
            verified = profiles.get("ksei_source_verified", pd.Series(False, index=profiles.index)).fillna(False).astype(bool)
            success = set(_symbols(profiles.loc[verified, "ticker"].tolist()))
        for ticker in allowed:
            if ticker in success:
                _NEGATIVE_CACHE.record_success("KSEI", "SECURITY_PROFILE", ticker)
                continue
            detail = ""
            if isinstance(audit, pd.DataFrame) and not audit.empty and "ticker" in audit.columns:
                rows = audit[audit["ticker"].astype(str).str.upper().eq(ticker.upper())]
                if not rows.empty:
                    detail = " ".join(rows.astype(str).tail(1).iloc[0].tolist())
            _NEGATIVE_CACHE.record_failure("KSEI", "SECURITY_PROFILE", ticker, _status_failure_class(detail))
        return profiles, actions, _append_skip_audit(audit, skipped, "KSEI_SECURITY_PROFILE_NEGATIVE_CACHE")

    wrapped.__negative_cache_v1__ = True
    setattr(module, "fetch_many_ksei_profiles", wrapped)


def _wrap_dashboard(module: Any) -> None:
    original = getattr(module, "enrich_dashboard_scores", None)
    if not callable(original) or getattr(original, "__three_rank_contract_v1__", False):
        return

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        out = original(*args, **kwargs)
        if isinstance(out, pd.DataFrame) and not out.empty:
            out = apply_three_rank_contract(out)
            out["ranking_contract_version"] = PATCH_VERSION
        return out

    wrapped.__three_rank_contract_v1__ = True
    setattr(module, "enrich_dashboard_scores", wrapped)


def install(expected_release: str = "") -> dict[str, Any]:
    import autonomous_enrichment
    import top3_dashboard_legacy
    import top3_dashboard

    _wrap_fundamentals(autonomous_enrichment)
    _wrap_ksei(autonomous_enrichment)
    _wrap_dashboard(top3_dashboard_legacy)
    _wrap_dashboard(top3_dashboard)
    return {
        "patch_version": PATCH_VERSION,
        "release": expected_release,
        "ranking_contract": "RAW_RESEARCH|GUARDED_DECISION_PRIORITY|PRODUCTION_REAL_MONEY",
        "negative_cache": "PROVIDER_SPECIFIC_FUNDAMENTAL_AND_KSEI_ONLY",
    }


__all__ = ["PATCH_VERSION", "install"]
