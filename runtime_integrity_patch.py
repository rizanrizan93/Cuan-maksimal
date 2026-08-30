from __future__ import annotations

"""Hot-reload-safe evidence/ranking/runtime integrity hooks for Emir scanner."""

from functools import wraps
from html import escape
from typing import Any, Iterable
import re
import time

import pandas as pd

from evidence_governance import ProviderNegativeCache
from live_forward_evidence import collect_live_forward_evidence

PATCH_VERSION = "1.1.0-live-forward"
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


def _wrap_dashboard_scores(module: Any) -> None:
    original = getattr(module, "enrich_dashboard_scores", None)
    if not callable(original) or getattr(original, "__three_rank_contract_v1__", False):
        return

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        out = original(*args, **kwargs)
        if not isinstance(out, pd.DataFrame) or out.empty:
            return out
        import final_decision
        if final_decision.is_final_decision_snapshot(out):
            return out.copy(deep=True)
        return final_decision.finalize_decision_snapshot(out)

    wrapped.__three_rank_contract_v1__ = True
    setattr(module, "enrich_dashboard_scores", wrapped)


def _wrap_dashboard_renderer(module: Any) -> None:
    original = getattr(module, "render_top3_dashboard_html", None)
    cost_block = getattr(module, "_cost_block", None)
    if not callable(original) or not callable(cost_block) or getattr(original, "__cost_placement_v2__", False):
        return

    @wraps(original)
    def wrapped(top3: pd.DataFrame, *args: Any, **kwargs: Any) -> str:
        html = original(top3, *args, **kwargs)
        # Remove all legacy-injected cost blocks first. The old replace(..., 1)
        # loop re-used a marker that it inserted again, causing blocks #2/#3 to
        # be placed inside card #1.
        html = re.sub(r'<div class="es-cost-basis">.*?</div>', "", html, flags=re.DOTALL)
        cursor = 0
        for _, row in top3.iterrows():
            evidence_type = str(row.get("broker_inventory_evidence_type") or "OHLCV_PROXY")
            flow_note = "DIRECT BROKER EVIDENCE" if "DIRECT" in evidence_type else "OHLCV PROXY — BUKAN IDENTITAS BROKER"
            marker = f"</div><p>{escape(flow_note)}</p>"
            index = html.find(marker, cursor)
            if index < 0:
                continue
            replacement = "</div>" + cost_block(row) + f"<p>{escape(flow_note)}</p>"
            html = html[:index] + replacement + html[index + len(marker):]
            cursor = index + len(replacement)
        return html

    wrapped.__cost_placement_v2__ = True
    setattr(module, "render_top3_dashboard_html", wrapped)


def _cached_forward_research(config: Any, tickers: list[str], cache_module: Any) -> tuple[pd.DataFrame, set[str]]:
    events: list[dict[str, Any]] = []
    fresh: set[str] = set()
    if not getattr(config, "ready", False):
        return pd.DataFrame(), fresh
    try:
        rows = cache_module.read_source_cache(config, tickers, "FORWARD_RESEARCH")
    except Exception:
        return pd.DataFrame(), fresh
    now = pd.Timestamp.now(tz="UTC")
    for ticker, row in rows.items():
        try:
            if not cache_module._row_hash_valid(row):
                continue
        except Exception:
            continue
        valid_until = pd.to_datetime(row.get("valid_until"), errors="coerce", utc=True)
        checked_at = pd.to_datetime(row.get("checked_at"), errors="coerce", utc=True)
        if pd.isna(valid_until) or valid_until < now or pd.isna(checked_at):
            continue
        symbol = _symbols([ticker])
        if not symbol:
            continue
        fresh.add(symbol[0])
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        for event in payload.get("events", []) if isinstance(payload.get("events"), list) else []:
            if isinstance(event, dict):
                events.append(dict(event))
    return pd.DataFrame(events), fresh


def _persist_forward_research(config: Any, scan_id: str, events: pd.DataFrame, audits: pd.DataFrame, cache_module: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not getattr(config, "ready", False) or not isinstance(audits, pd.DataFrame) or audits.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, audit in audits.iterrows():
        ticker = _symbols([audit.get("ticker")])
        if not ticker:
            continue
        symbol = ticker[0]
        ticker_events = []
        if isinstance(events, pd.DataFrame) and not events.empty and "ticker" in events.columns:
            ticker_events = events.loc[events["ticker"].astype(str).str.upper().eq(symbol.upper())].to_dict("records")
        payload = {"events": ticker_events, "audit": audit.to_dict(), "collection_version": PATCH_VERSION}
        rows.append(cache_module.build_source_cache_row(
            symbol,
            "FORWARD_RESEARCH",
            payload,
            provider="GOOGLE_NEWS_RSS_FORWARD",
            status=str(audit.get("state") or "CHECKED"),
            checked_at=audit.get("checked_at"),
            ttl_hours=24.0,
            latest_observed_at=audit.get("checked_at"),
            last_scan_id=str(scan_id or ""),
        ))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    return cache_module.persist_verify_cache_bundle(config, scan_id=str(scan_id or "FORWARD_RESEARCH"), ohlcv_rows=[], source_rows=rows)


def _wrap_load_cached_news(resumable_module: Any, cache_module: Any) -> None:
    original = getattr(resumable_module, "load_cached_news", None)
    if not callable(original) or getattr(original, "__live_forward_research_v1__", False):
        return

    @wraps(original)
    def wrapped(config: Any, tickers: Iterable[str]):
        names = _symbols(tickers)
        existing, audit = original(config, names)
        cached_forward, fresh = _cached_forward_research(config, names, cache_module)
        missing = [ticker for ticker in names if ticker not in fresh]
        live_events = pd.DataFrame()
        live_audit = pd.DataFrame()
        persist_write = persist_verify = pd.DataFrame()
        if missing:
            live_events, live_audit = collect_live_forward_evidence(missing, lookback_days=120, max_workers=12, timeout=4.0)
            if not live_events.empty:
                live_events["ticker"] = live_events["ticker"].map(lambda value: _symbols([value])[0] if _symbols([value]) else "")
            scan_id = ""
            try:
                scan_id = str(getattr(config, "scan_id", "") or "")
            except Exception:
                scan_id = ""
            persist_write, persist_verify = _persist_forward_research(config, scan_id, live_events, live_audit, cache_module)
        frames = [frame for frame in (existing, cached_forward, live_events) if isinstance(frame, pd.DataFrame) and not frame.empty]
        combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        if not combined.empty:
            dedupe = [column for column in ("ticker", "title", "url") if column in combined.columns]
            if dedupe:
                combined = combined.drop_duplicates(dedupe, keep="first").reset_index(drop=True)
        audits = [frame for frame in (audit, live_audit, persist_write, persist_verify) if isinstance(frame, pd.DataFrame) and not frame.empty]
        audit_out = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
        checked = len(fresh) + (live_audit.loc[pd.to_numeric(live_audit.get("coverage_pct", 0), errors="coerce").fillna(0).gt(0), "ticker"].nunique() if not live_audit.empty else 0)
        audit_out = pd.concat([audit_out, pd.DataFrame([{
            "provider": "LIVE_FORWARD_RESEARCH",
            "status": "COLLECTION_COMPLETE" if checked >= len(names) else "COLLECTION_PARTIAL",
            "items": len(combined),
            "detail": f"checked={checked}/{len(names)}; research-only live forward discovery persisted separately from strict official evidence",
        }])], ignore_index=True, sort=False)
        return combined, audit_out

    wrapped.__live_forward_research_v1__ = True
    setattr(resumable_module, "load_cached_news", wrapped)


def _wrap_cache_checkpoint_retry(resumable_module: Any, cache_module: Any) -> None:
    original = getattr(resumable_module, "persist_verify_cache_bundle", None)
    if not callable(original) or getattr(original, "__checkpoint_retry_v1__", False):
        return

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        last = original(*args, **kwargs)
        try:
            _, verification = last
        except Exception:
            return last
        if cache_module.cache_commit_succeeded(verification):
            return last
        # A common free-tier race is rows_expected == rows_verified while the
        # exact hash readback catches an immediately superseded/upserted row.
        # Re-run the idempotent write+readback twice before declaring PAUSED.
        summary = verification.iloc[0] if isinstance(verification, pd.DataFrame) and not verification.empty else pd.Series(dtype=object)
        expected = int(pd.to_numeric(pd.Series([summary.get("rows_expected")]), errors="coerce").fillna(0).iloc[0])
        observed = int(pd.to_numeric(pd.Series([summary.get("rows_verified")]), errors="coerce").fillna(0).iloc[0])
        if expected <= 0 or observed != expected:
            return last
        for delay in (0.15, 0.40):
            time.sleep(delay)
            candidate = original(*args, **kwargs)
            try:
                _, verify_candidate = candidate
            except Exception:
                continue
            last = candidate
            if cache_module.cache_commit_succeeded(verify_candidate):
                return candidate
        return last

    wrapped.__checkpoint_retry_v1__ = True
    setattr(resumable_module, "persist_verify_cache_bundle", wrapped)


def install(expected_release: str = "") -> dict[str, Any]:
    import autonomous_enrichment
    import persistent_cache
    import resumable_scan
    import top3_dashboard_legacy
    import top3_dashboard

    _wrap_fundamentals(autonomous_enrichment)
    _wrap_ksei(autonomous_enrichment)
    _wrap_dashboard_scores(top3_dashboard_legacy)
    _wrap_dashboard_scores(top3_dashboard)
    _wrap_dashboard_renderer(top3_dashboard)
    _wrap_load_cached_news(resumable_scan, persistent_cache)
    _wrap_cache_checkpoint_retry(resumable_scan, persistent_cache)
    return {
        "patch_version": PATCH_VERSION,
        "release": expected_release,
        "ranking_contract": "RAW_RESEARCH|GUARDED_DECISION_PRIORITY|PRODUCTION_REAL_MONEY",
        "negative_cache": "PROVIDER_SPECIFIC_FUNDAMENTAL_AND_KSEI_ONLY",
        "smart_money_cost_placement": "ONE_BLOCK_PER_CARD",
        "live_forward_collection": "FULL_DEEP_REVIEW_UNIVERSE_24H_CACHE",
        "cache_checkpoint_retry": "TWO_IDEMPOTENT_RETRIES_ON_FULL_ROW_READBACK_MISMATCH",
    }


__all__ = ["PATCH_VERSION", "install"]
