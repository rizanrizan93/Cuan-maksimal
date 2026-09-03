from __future__ import annotations

"""EMIR runtime integration for shared/cache-first fundamental collection."""

from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from shared_fundamental_runtime import (
    PLUANG_FINANCIALS_URL, PLUANG_FUNDAMENTALS_URL, PLUANG_RESOLVE_URL,
    YAHOO_FINANCIALS_URL, YAHOO_SUMMARY_URL, SharedFundamentalRuntime,
    bare_ticker, canonicalize_metric_rows, jk_ticker, normalize_pluang_payloads,
    normalize_yahoo_payloads,
)


PATCH_VERSION = "1.0.0-phase5.6-shared-fundamental"
STRUCTURED_REFRESH_LIMIT_PER_CHUNK = 8
_INSTALLED = False


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _coalesce_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame.copy() for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty and "ticker" in frame.columns]
    if not usable:
        return pd.DataFrame()
    output = usable[0].copy()
    output["ticker"] = output["ticker"].map(jk_ticker)
    output = output.drop_duplicates("ticker", keep="last").set_index("ticker")
    for fallback in usable[1:]:
        local = fallback.copy(); local["ticker"] = local["ticker"].map(jk_ticker); local = local.drop_duplicates("ticker", keep="last").set_index("ticker")
        columns = list(dict.fromkeys([*output.columns, *local.columns]))
        output = output.reindex(columns=columns)
        local = local.reindex(columns=columns)
        output = output.combine_first(local)
    return output.reset_index()


def _freshness(period: Any) -> tuple[str, float]:
    stamp = pd.to_datetime(period, errors="coerce")
    if pd.isna(stamp):
        return "UNKNOWN_PERIOD", np.nan
    current = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()
    value = pd.Timestamp(stamp)
    if value.tzinfo is not None:
        value = value.tz_localize(None)
    age = max(0.0, float((current - value.normalize()).days))
    return ("CURRENT_QUARTERLY_PERIOD" if age <= 120 else "AGING_QUARTERLY_PERIOD" if age <= 190 else "STALE_QUARTERLY_PERIOD"), age


def _proxy_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    from autonomous_enrichment import recalibrate_cached_fundamental_snapshot

    metrics = dict(item.get("proxy_metrics") or {})
    official = dict(item.get("official_metrics") or {})

    def proxy_value(name: str) -> Any:
        value = metrics.get(name)
        # canonicalize_metric_rows exposes official facts as fallback only when a
        # normalized/public value is absent. Do not relabel those official YTD facts
        # as a public TTM proxy; the official adapter handles them separately.
        if name in official and _finite(value) is not None and _finite(official.get(name)) is not None:
            if abs(float(value) - float(official[name])) <= max(1e-9, abs(float(official[name])) * 1e-12):
                return np.nan
        return value

    period = item.get("proxy_period_end")
    observed = item.get("proxy_observed_at") or item.get("official_observed_at") or ""
    fresh_state, age = _freshness(period)
    payload: dict[str, Any] = {
        "ticker": jk_ticker(item.get("ticker")),
        "revenue_latest": proxy_value("revenue"),
        "net_income_latest": proxy_value("net_income"),
        "operating_cash_flow_latest": proxy_value("operating_cash_flow"),
        "free_cash_flow_proxy_latest": proxy_value("free_cash_flow") if _finite(proxy_value("free_cash_flow")) is not None else proxy_value("free_cash_flow_proxy"),
        "cash_latest": proxy_value("cash"), "debt_latest": proxy_value("total_debt"),
        "total_liabilities_latest": proxy_value("total_liabilities"), "total_assets_latest": proxy_value("total_assets"),
        "equity_latest": proxy_value("equity"), "current_assets_latest": proxy_value("current_assets"),
        "current_liabilities_latest": proxy_value("current_liabilities"),
        "revenue_growth_pct": metrics.get("revenue_growth_pct", metrics.get("revenue_growth_yoy_pct")),
        "earnings_growth_pct": metrics.get("earnings_growth_pct", metrics.get("earnings_growth_yoy_pct")),
        "revenue_growth_qoq_pct": np.nan, "earnings_growth_qoq_pct": np.nan,
        "revenue_growth_yoy_pct": metrics.get("revenue_growth_yoy_pct", metrics.get("revenue_growth_pct")),
        "earnings_growth_yoy_pct": metrics.get("earnings_growth_yoy_pct", metrics.get("earnings_growth_pct")),
        "revenue_growth_ytd_yoy_pct": np.nan, "earnings_growth_ytd_yoy_pct": np.nan,
        "growth_basis_state": "SHARED_SOURCE_BACKED_GROWTH",
        "fundamental_growth_consistency_state": "SHARED_SINGLE_GROWTH_BASIS",
        "fundamental_growth_consistency_score": 72.0,
        "net_margin_pct": metrics.get("net_margin_pct"), "operating_margin_pct": metrics.get("operating_margin_pct"),
        "net_margin_ttm_pct": metrics.get("net_margin_pct"), "operating_margin_ttm_pct": metrics.get("operating_margin_pct"),
        "roe_ttm_pct": metrics.get("roe_pct"), "roa_ttm_pct": metrics.get("roa_pct"),
        "der_ratio": metrics.get("debt_equity", metrics.get("interest_bearing_debt_to_equity")),
        "interest_bearing_debt_to_equity": metrics.get("interest_bearing_debt_to_equity", metrics.get("debt_equity")),
        "total_liabilities_to_equity": metrics.get("total_liabilities_to_equity"),
        "net_debt_to_equity": metrics.get("net_debt_to_equity"), "current_ratio": metrics.get("current_ratio"),
        "cash_to_debt_ratio": metrics.get("cash_to_debt_ratio"), "ocf_conversion_ratio": metrics.get("ocf_conversion_ratio", metrics.get("cash_conversion_ttm")),
        "fundamental_latest_period": period or "", "fundamental_income_period": period or "", "fundamental_balance_period": period or "",
        "fundamental_cashflow_period": period or "" if _finite(proxy_value("operating_cash_flow")) is not None else "",
        "fundamental_period_alignment_state": "SHARED_PERIOD_OBSERVED" if period else "SHARED_PERIOD_UNKNOWN",
        "fundamental_period_freshness_state": fresh_state, "fundamental_period_age_days": age,
        "fundamental_observed_at": observed, "fundamental_availability_state": "POINT_IN_TIME_OBSERVED" if observed else "AVAILABILITY_TIMESTAMP_UNVERIFIED",
        "fundamental_provenance_state": "SHARED_FACTUAL_EVIDENCE_PROXY",
        "fundamental_official_source_coverage_pct": 0.0,
        "fundamental_public_source_quality_pct": 88.0,
        "fundamental_cache_schema_version": "5",
    }
    recalibrated = recalibrate_cached_fundamental_snapshot(payload)
    ocf = _finite(payload.get("operating_cash_flow_latest")); fcf = _finite(payload.get("free_cash_flow_proxy_latest"))
    recalibrated["fundamental_cashflow_state"] = (
        "SHARED_OCF_FCF_AVAILABLE_PERIOD_BASIS" if ocf is not None and fcf is not None
        else "SHARED_OCF_AVAILABLE_PERIOD_BASIS" if ocf is not None
        else "SHARED_CASHFLOW_NOT_AVAILABLE"
    )
    recalibrated["fundamental_provenance_state"] = "SHARED_FACTUAL_EVIDENCE_PROXY"
    recalibrated["shared_fundamental_runtime_version"] = PATCH_VERSION
    return recalibrated


def _official_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(item.get("official_metrics") or {})
    period = item.get("official_period_end")
    if not metrics or not period:
        return {}
    payload = {
        "ticker": jk_ticker(item.get("ticker")), "idx_official_source_verified": True,
        "idx_official_period_end": period, "idx_official_observed_at": item.get("official_observed_at") or "",
        "idx_official_availability_state": "POINT_IN_TIME_OBSERVED_SHARED_HUB",
        "idx_official_revenue": metrics.get("revenue"), "idx_official_net_income": metrics.get("net_income"),
        "idx_official_ocf": metrics.get("operating_cash_flow"), "idx_official_fcf_proxy": metrics.get("free_cash_flow_proxy"),
        "idx_official_cash": metrics.get("cash"), "idx_official_interest_bearing_debt_proxy": metrics.get("total_debt"),
        "idx_official_liabilities": metrics.get("total_liabilities"), "idx_official_assets": metrics.get("total_assets"),
        "idx_official_equity": metrics.get("equity"), "idx_official_current_assets": metrics.get("current_assets"),
        "idx_official_current_liabilities": metrics.get("current_liabilities"),
        "idx_official_revenue_growth_yoy_pct": metrics.get("revenue_growth_yoy_pct"),
        "idx_official_earnings_growth_yoy_pct": metrics.get("earnings_growth_yoy_pct"),
        "idx_official_net_margin_pct": metrics.get("net_margin_pct"), "idx_official_operating_margin_pct": metrics.get("operating_margin_pct"),
        "idx_official_ocf_conversion_ratio": metrics.get("ocf_conversion_ratio"),
        "idx_official_interest_bearing_debt_to_equity": metrics.get("interest_bearing_debt_to_equity"),
        "idx_official_total_liabilities_to_equity": metrics.get("total_liabilities_to_equity"),
        "idx_official_net_debt_to_equity": metrics.get("net_debt_to_equity"), "idx_official_current_ratio": metrics.get("current_ratio"),
        "idx_official_cash_to_debt_ratio": metrics.get("cash_to_debt_ratio"),
        "idx_official_coverage_pct": item.get("official_coverage_pct", 0.0),
        "idx_official_cashflow_state": "IDX_OFFICIAL_OCF_AVAILABLE" if _finite(metrics.get("operating_cash_flow")) is not None else "IDX_OFFICIAL_CASHFLOW_MISSING",
        "idx_official_source_url": "", "idx_official_provenance_state": "SHARED_HUB_EXACT_OFFICIAL_FACTS",
        "shared_fundamental_runtime_version": PATCH_VERSION,
    }
    return payload


def _bundle_frames(bundle: Mapping[str, Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    proxy = [_proxy_payload(item) for item in bundle.values() if item.get("proxy_metrics")]
    official = [_official_payload(item) for item in bundle.values() if item.get("official_metrics")]
    return pd.DataFrame([row for row in proxy if row]), pd.DataFrame([row for row in official if row])


def _read_shared(tickers: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    runtime = SharedFundamentalRuntime("EMIR")
    bundle, meta = runtime.read_bundle(tickers)
    proxy, official = _bundle_frames(bundle)
    return proxy, official, meta


def _collect_structured_local(ticker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use ZAPI during a scan even when cross-database Shared Hub secrets are absent."""
    runtime = SharedFundamentalRuntime("EMIR")
    code = bare_ticker(ticker); observed = datetime.now(timezone.utc).isoformat(); attempts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        resolved = runtime._zapi_get(PLUANG_RESOLVE_URL, {"code": code}); stock_id = resolved.get("stockId")
        if not stock_id:
            raise RuntimeError("NO_MATCH")
        params = {"code": code, "stockId": stock_id}
        fundamentals = runtime._zapi_get(PLUANG_FUNDAMENTALS_URL, params)
        financials = runtime._zapi_get(PLUANG_FINANCIALS_URL, {**params, "period": "quarterly"})
        rows = normalize_pluang_payloads(code, fundamentals, financials, observed_at=observed)
        attempts.append({"provider":"PLUANG","state":"OK","rows":len(rows)})
    except Exception as exc:
        attempts.append({"provider":"PLUANG","state":type(exc).__name__,"detail":str(exc)[:120]})
    if not rows:
        try:
            symbol = jk_ticker(code); summary = runtime._zapi_get(YAHOO_SUMMARY_URL, {"symbol": symbol}); statements = {
                family: runtime._zapi_get(YAHOO_FINANCIALS_URL, {"symbol":symbol,"statement":family,"period":"quarterly"})
                for family in ("income","balance","cashflow")
            }
            rows = normalize_yahoo_payloads(symbol, summary, statements, observed_at=observed)
            attempts.append({"provider":"YAHOO_ZAPI","state":"OK","rows":len(rows)})
        except Exception as exc:
            attempts.append({"provider":"YAHOO_ZAPI","state":type(exc).__name__,"detail":str(exc)[:120]})
    if not rows:
        return {}, {"state":"STRUCTURED_PROVIDERS_EXHAUSTED","attempts":attempts}
    bundle = canonicalize_metric_rows(rows); item = bundle.get(code, {})
    payload = _proxy_payload(item) if item else {}
    if runtime.ready:
        try:
            runtime._persist(rows)
        except Exception:
            pass
    return payload, {"state":"STRUCTURED_REFRESHED" if payload else "STRUCTURED_EMPTY","attempts":attempts,"rows":len(rows)}


def _shared_good(payload: Mapping[str, Any], pc: Any, now: Any) -> bool:
    period = pd.to_datetime(payload.get("fundamental_latest_period"), errors="coerce")
    coverage = _finite(payload.get("fundamental_coverage_pct")) or 0.0
    quality = _finite(payload.get("fundamental_data_quality_score")) or 0.0
    return bool(pd.notna(period) and coverage >= 45.0 and quality >= 35.0 and not pc._fundamental_payload_reporting_lagged(payload, now))


def _official_good(payload: Mapping[str, Any], pc: Any, now: Any) -> bool:
    if not bool(payload.get("idx_official_source_verified")):
        return False
    period = pd.to_datetime(payload.get("idx_official_period_end"), errors="coerce")
    expected = pc._expected_quarter_end(now)
    if pd.isna(period):
        return False
    return bool(pd.isna(expected) or pd.Timestamp(period).normalize() >= pd.Timestamp(expected).normalize())


def _publish_emir_frame(frame: pd.DataFrame | None) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return
    runtime = SharedFundamentalRuntime("EMIR")
    if not runtime.ready:
        return
    units = {"revenue_growth_pct":"PERCENT","earnings_growth_pct":"PERCENT","roe_pct":"PERCENT","roa_pct":"PERCENT","net_margin_pct":"PERCENT","operating_margin_pct":"PERCENT","debt_equity":"RATIO","current_ratio":"RATIO","operating_cash_flow":"CURRENCY_NATIVE","free_cash_flow":"CURRENCY_NATIVE"}
    for _, row in frame.drop_duplicates("ticker", keep="last").iterrows():
        provenance = str(row.get("fundamental_provenance_state") or "")
        if "SHARED_FACTUAL_EVIDENCE" in provenance:
            continue
        metrics: dict[str, Any] = {}
        for source, target in (("revenue_growth_pct","revenue_growth_pct"),("earnings_growth_pct","earnings_growth_pct"),("roe_ttm_pct","roe_pct"),("roa_ttm_pct","roa_pct"),("net_margin_ttm_pct","net_margin_pct"),("operating_margin_ttm_pct","operating_margin_pct"),("interest_bearing_debt_to_equity","debt_equity"),("current_ratio","current_ratio"),("operating_cash_flow_ttm","operating_cash_flow"),("free_cash_flow_proxy_ttm","free_cash_flow")):
            value = _finite(row.get(source))
            if value is not None: metrics[target] = value
        if not metrics:
            continue
        try:
            runtime.publish_metrics(str(row.get("ticker")), metrics, provider="EMIR_NORMALIZED_RUNTIME", source_families=provenance or "EMIR_SOURCE_BACKED_FUNDAMENTAL", observed_at=row.get("fundamental_observed_at") or datetime.now(timezone.utc).isoformat(), period_end=row.get("fundamental_latest_period"), units=units)
        except Exception:
            pass


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import persistent_cache as pc

    for name in ("fetch_fundamental_cache_first", "fetch_idx_official_fundamental_cache_first", "load_cached_fundamentals", "load_cached_idx_official_fundamentals"):
        original_name = f"_phase56_original_{name}"
        if not hasattr(pc, original_name):
            setattr(pc, original_name, getattr(pc, name))

    orig_fetch = pc._phase56_original_fetch_fundamental_cache_first
    orig_fetch_official = pc._phase56_original_fetch_idx_official_fundamental_cache_first
    orig_load = pc._phase56_original_load_cached_fundamentals
    orig_load_official = pc._phase56_original_load_cached_idx_official_fundamentals

    def fetch_fundamental_cache_first(config: Any, tickers: Iterable[str], *, max_workers: int = 3, now: Any = None, force_refresh: bool = False, last_scan_id: str = ""):
        symbols = list(dict.fromkeys(pc.normalize_ticker(t) for t in tickers if pc.normalize_ticker(t)))
        cached = pc.read_source_cache(config, symbols, "FUNDAMENTAL")
        safe_local: list[str] = []
        if not force_refresh:
            for symbol in symbols:
                row = cached.get(symbol); payload = row.get("payload") if row and isinstance(row.get("payload"), dict) else {}
                if row and pc._row_hash_valid(row) and pc._fundamental_payload_compatible(payload) and pc._source_row_fresh(row, now) and not pc._fundamental_payload_reporting_lagged(payload, now):
                    safe_local.append(symbol)
        shared_proxy, _, shared_meta = _read_shared(symbols)
        shared_map = shared_proxy.set_index("ticker").to_dict(orient="index") if not shared_proxy.empty else {}
        shared_good = [] if force_refresh else [symbol for symbol in symbols if symbol not in safe_local and _shared_good(shared_map.get(symbol, {}), pc, now)]
        unresolved = [symbol for symbol in symbols if symbol not in safe_local and symbol not in shared_good]

        structured_rows: list[dict[str, Any]] = []; structured_audits: list[dict[str, Any]] = []; structured_writes: list[dict[str, Any]] = []
        structured_targets = unresolved[:STRUCTURED_REFRESH_LIMIT_PER_CHUNK]
        structured_ok: set[str] = set()
        for symbol in structured_targets:
            payload, meta = _collect_structured_local(symbol)
            if payload and (_finite(payload.get("fundamental_coverage_pct")) or 0) >= 35:
                payload["ticker"] = symbol; structured_rows.append(payload); structured_ok.add(symbol)
                structured_audits.append({"ticker":symbol,"provider":"ZAPI_STRUCTURED_FUNDAMENTAL","status":"REFRESHED","items":1,"detail":str(meta.get("attempts"))[:500],"cache_state":"REFRESHED"})
                structured_writes.append(pc.build_source_cache_row(symbol,"FUNDAMENTAL",payload,provider="ZAPI_STRUCTURED_FUNDAMENTAL",status="OK",checked_at=now,ttl_hours=pc.FUNDAMENTAL_CACHE_TTL_HOURS,last_scan_id=last_scan_id))
            else:
                structured_audits.append({"ticker":symbol,"provider":"ZAPI_STRUCTURED_FUNDAMENTAL","status":"NO_ITEMS","items":0,"detail":str(meta.get("attempts"))[:500],"cache_state":"PROVIDER_FAILED"})
        provider_targets = [symbol for symbol in unresolved if symbol not in structured_ok]

        parts: list[pd.DataFrame] = []; audits: list[pd.DataFrame] = []; writes: list[dict[str, Any]] = list(structured_writes)
        if safe_local:
            f, a, w = orig_fetch(config, safe_local, max_workers=max_workers, now=now, force_refresh=False, last_scan_id=last_scan_id); parts.append(f); audits.append(a); writes.extend(w)
        if provider_targets:
            f, a, w = orig_fetch(config, provider_targets, max_workers=max_workers, now=now, force_refresh=force_refresh, last_scan_id=last_scan_id); parts.append(f); audits.append(a); writes.extend(w)
        if structured_rows:
            parts.insert(0, pd.DataFrame(structured_rows))
        if shared_good:
            selected = shared_proxy[shared_proxy["ticker"].isin(shared_good)].copy(); parts.append(selected)
            for symbol in shared_good:
                payload = shared_map[symbol]; writes.append(pc.build_source_cache_row(symbol,"FUNDAMENTAL",payload,provider="SHARED_EVIDENCE_HUB",status="OK",checked_at=now,ttl_hours=pc.FUNDAMENTAL_CACHE_TTL_HOURS,last_scan_id=last_scan_id))
            audits.append(pd.DataFrame([{"ticker":symbol,"provider":"SHARED_EVIDENCE_HUB","status":"CACHE_HIT","items":1,"detail":f"provider avoided; shared={shared_meta.get('state')}","cache_state":"CACHE_HIT"} for symbol in shared_good]))
        if structured_audits:
            audits.append(pd.DataFrame(structured_audits))
        frame = _coalesce_frames(*parts)
        audit = pd.concat([a for a in audits if isinstance(a,pd.DataFrame) and not a.empty], ignore_index=True, sort=False) if any(isinstance(a,pd.DataFrame) and not a.empty for a in audits) else pd.DataFrame()
        _publish_emir_frame(frame)
        return frame, audit, writes

    def fetch_official_cache_first(config: Any, tickers: Iterable[str], *, max_workers: int = 2, now: Any = None, force_refresh: bool = False, last_scan_id: str = ""):
        symbols = list(dict.fromkeys(pc.normalize_ticker(t) for t in tickers if pc.normalize_ticker(t)))
        cached = pc.read_source_cache(config, symbols, "IDX_FUNDAMENTAL"); safe_local: list[str] = []
        if not force_refresh:
            for symbol in symbols:
                row=cached.get(symbol); payload=row.get("payload") if row and isinstance(row.get("payload"),dict) else {}
                if row and pc._row_hash_valid(row) and pc._source_row_fresh(row,now) and bool(payload.get("idx_official_source_verified")):
                    safe_local.append(symbol)
        _, shared_official, shared_meta = _read_shared(symbols); smap=shared_official.set_index("ticker").to_dict(orient="index") if not shared_official.empty else {}
        shared_good=[] if force_refresh else [s for s in symbols if s not in safe_local and _official_good(smap.get(s,{}),pc,now)]
        provider_targets=[s for s in symbols if s not in safe_local and s not in shared_good]
        parts=[]; audits=[]; writes=[]
        if safe_local:
            f,a,w=orig_fetch_official(config,safe_local,max_workers=max_workers,now=now,force_refresh=False,last_scan_id=last_scan_id); parts.append(f);audits.append(a);writes.extend(w)
        if provider_targets:
            f,a,w=orig_fetch_official(config,provider_targets,max_workers=max_workers,now=now,force_refresh=force_refresh,last_scan_id=last_scan_id);parts.append(f);audits.append(a);writes.extend(w)
        if shared_good:
            selected=shared_official[shared_official["ticker"].isin(shared_good)].copy();parts.insert(0,selected)
            for symbol in shared_good:
                payload=smap[symbol];writes.append(pc.build_source_cache_row(symbol,"IDX_FUNDAMENTAL",payload,provider="SHARED_EVIDENCE_HUB_OFFICIAL",status="OK",checked_at=now,ttl_hours=pc.IDX_OFFICIAL_FUNDAMENTAL_TTL_HOURS,last_scan_id=last_scan_id))
            audits.append(pd.DataFrame([{"ticker":s,"provider":"SHARED_EVIDENCE_HUB_OFFICIAL","status":"CACHE_HIT","items":1,"detail":f"IDX download avoided; shared={shared_meta.get('state')}","cache_state":"CACHE_HIT"} for s in shared_good]))
        frame=_coalesce_frames(*parts); audit=pd.concat([a for a in audits if isinstance(a,pd.DataFrame) and not a.empty],ignore_index=True,sort=False) if any(isinstance(a,pd.DataFrame) and not a.empty for a in audits) else pd.DataFrame(); return frame,audit,writes

    def load_fundamentals(config: Any, tickers: Iterable[str]):
        base,audit=orig_load(config,tickers); shared,_,meta=_read_shared(tickers); merged=_coalesce_frames(base,shared)
        if not shared.empty:
            extra=pd.DataFrame([{"ticker":"*","provider":"SHARED_EVIDENCE_HUB","status":"CACHE_LOAD","items":len(shared),"detail":f"shared={meta.get('state')}","cache_state":"CACHE_LOAD"}]); audit=pd.concat([audit,extra],ignore_index=True,sort=False) if not audit.empty else extra
        return merged,audit

    def load_official(config: Any, tickers: Iterable[str]):
        base,audit=orig_load_official(config,tickers);_,shared,meta=_read_shared(tickers);merged=_coalesce_frames(base,shared)
        if not shared.empty:
            extra=pd.DataFrame([{"ticker":"*","provider":"SHARED_EVIDENCE_HUB_OFFICIAL","status":"CACHE_LOAD","items":len(shared),"detail":f"shared={meta.get('state')}","cache_state":"CACHE_LOAD"}]);audit=pd.concat([audit,extra],ignore_index=True,sort=False) if not audit.empty else extra
        return merged,audit

    pc.fetch_fundamental_cache_first=fetch_fundamental_cache_first
    pc.fetch_idx_official_fundamental_cache_first=fetch_official_cache_first
    pc.load_cached_fundamentals=load_fundamentals
    pc.load_cached_idx_official_fundamentals=load_official
    _INSTALLED=True


__all__=["PATCH_VERSION","STRUCTURED_REFRESH_LIMIT_PER_CHUNK","install"]
