from __future__ import annotations

import copy
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import resumable_scan as rs
from persistence import DatabaseConfig


def synthetic_frame(seed: int, bars: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-05-01", periods=bars)
    returns = rng.normal(0.0007, 0.018, bars)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.004, bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.018, bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.018, bars))
    volume = rng.integers(500_000, 5_000_000, bars)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def main() -> None:
    tickers = [f"T{i:03d}.JK" for i in range(300)]
    universe = [{"ticker": ticker, "company_name": f"Issuer {i}", "sector": "TEST"} for i, ticker in enumerate(tickers)]
    frames_store = {ticker: synthetic_frame(i + 1) for i, ticker in enumerate(tickers)}
    frames_store["^JKSE"] = synthetic_frame(999)
    ksei_profiles_store: dict[str, dict] = {}
    ksei_actions_store: dict[str, list[dict]] = defaultdict(list)
    news_store: dict[str, list[dict]] = defaultdict(list)
    fundamental_store: dict[str, dict] = {}
    chunk_rows: list[dict] = []
    calls = defaultdict(list)

    official_store = {}
    job = {
        "scan_id": "resume300",
        "universe_hash": "hash",
        "scanner_version": "1.6.3",
        "status": "CREATED",
        "current_stage": "BENCHMARK",
        "current_offset": 0,
        "current_chunk": 0,
        "chunk_size": 20,
        "total_tickers": 300,
        "processed_tickers": 0,
        "failed_tickers": 0,
        "progress_pct": 0.0,
        "scan_mode": "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
        "result_status": "PENDING",
        "universe": universe,
        "settings": {
            "scan_mode": "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
            "period": "5y",
            "completed_only": True,
            "workers": 3,
            "deep_limit": 30,
        "deep_review_scope": "ALL_ELIGIBLE",
            "news_per_ticker": 4,
            "use_google_news": True,
            "use_yahoo_news": True,
            "auto_ksei": True,
            "auto_fundamental": True,
            "auto_idx_official_fundamental": True,
            "official_fundamental_limit": 60,
            "force_cache_refresh": False,
            "capital": 5_000_000,
            "risk_pct": 1.0,
            "max_position_cap_pct": 20.0,
            "calibration_mode": "SHADOW_ONLY",
        },
        "shortlist": [],
        "failures": {},
        "result_summary": {},
        "last_error": "",
    }

    def update(_config, _scan_id, patch):
        nonlocal job
        job = {**job, **copy.deepcopy(dict(patch))}
        return copy.deepcopy(job)

    def record(_config, **kwargs):
        payload = copy.deepcopy(kwargs)
        payload["payload"] = copy.deepcopy(kwargs.get("payload") or {})
        chunk_rows.append(payload)
        return payload

    def fetch_ohlcv(_config, requested, **kwargs):
        requested = list(requested)
        calls["OHLCV"].append(requested)
        frames = {ticker: frames_store[ticker] for ticker in requested if ticker in frames_store}
        audit = pd.DataFrame([{"ticker": ticker, "provider": "FIXTURE", "status": "CACHE_HIT" if ticker in frames else "ERROR"} for ticker in requested])
        writes = [{"ticker": ticker, "content_sha256": "x"} for ticker in requested if ticker in frames]
        return frames, audit, writes

    def persist_cache(_config, **kwargs):
        count = len(kwargs.get("ohlcv_rows") or []) + len(kwargs.get("source_rows") or [])
        write = pd.DataFrame([{"table": "__SUMMARY__", "rows_attempted": count, "rows_written": count, "state": "CACHE_WRITE_ALL"}])
        verify = pd.DataFrame([{"table": "__SUMMARY__", "rows_expected": count, "rows_verified": count, "state": "CACHE_DATABASE_COMMITTED"}])
        return write, verify

    def load_ohlcv(_config, requested, **kwargs):
        requested = list(requested)
        frames = {ticker: frames_store[ticker] for ticker in requested if ticker in frames_store}
        return frames, pd.DataFrame([{"ticker": ticker, "provider": "FIXTURE_CACHE", "status": "CACHE_LOAD"} for ticker in requested])

    def fetch_ksei(_config, requested, **kwargs):
        requested = list(requested); calls["KSEI"].append(requested)
        profiles = []
        for ticker in requested:
            item = {"ticker": ticker, "company_name": f"Company {ticker}", "sector": "TEST", "provider_state": "OK"}
            ksei_profiles_store[ticker] = item; profiles.append(item)
        audit = pd.DataFrame([{"ticker": ticker, "provider": "KSEI_FIXTURE", "status": "OK"} for ticker in requested])
        writes = [{"cache_key": f"KSEI:{ticker}", "content_sha256": "x"} for ticker in requested]
        return pd.DataFrame(profiles), pd.DataFrame(), audit, writes

    def load_ksei(_config, requested):
        profiles = [ksei_profiles_store[ticker] for ticker in requested if ticker in ksei_profiles_store]
        return pd.DataFrame(profiles), pd.DataFrame(), pd.DataFrame()

    def fetch_news(_config, local, **kwargs):
        requested = local["ticker"].tolist(); calls["NEWS"].append(requested)
        events = []
        for ticker in requested:
            item = {"ticker": ticker, "published_at": pd.Timestamp("2026-08-01", tz="UTC"), "title": f"Expansion {ticker}", "summary": "capacity expansion", "publisher": "Fixture", "url": f"https://example.com/{ticker}", "source_tier": "MEDIA", "source_verified": True, "materiality_score": 70, "financial_bridge_score": 60, "top_down_catalyst_score": 60, "industry_translation_score": 60, "issuer_alignment_score": 60, "category": "PROJECT_CAPACITY"}
            news_store[ticker].append(item); events.append(item)
        audit = pd.DataFrame([{"ticker": ticker, "provider": "NEWS_FIXTURE", "status": "OK"} for ticker in requested])
        writes = [{"cache_key": f"NEWS:{ticker}", "content_sha256": "x"} for ticker in requested]
        return pd.DataFrame(events), audit, writes

    def load_news(_config, requested):
        events = [item for ticker in requested for item in news_store.get(ticker, [])]
        return pd.DataFrame(events), pd.DataFrame()

    def fetch_fundamental(_config, requested, **kwargs):
        requested = list(requested); calls["FUNDAMENTAL"].append(requested)
        rows = []
        for ticker in requested:
            item = {"ticker": ticker, "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE", "fundamental_coverage_pct": 80, "fundamental_data_quality_score": 80, "fundamental_conversion_score": 72, "fundamental_cashflow_state": "CASHFLOW_TTM_MISSING", "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK", "revenue_growth_pct": 20, "earnings_growth_pct": 25}
            fundamental_store[ticker] = item; rows.append(item)
        audit = pd.DataFrame([{"ticker": ticker, "provider": "FUND_FIXTURE", "status": "OK"} for ticker in requested])
        writes = [{"cache_key": f"FUNDAMENTAL:{ticker}", "content_sha256": "x"} for ticker in requested]
        return pd.DataFrame(rows), audit, writes

    def load_fundamental(_config, requested):
        rows = [fundamental_store[ticker] for ticker in requested if ticker in fundamental_store]
        return pd.DataFrame(rows), pd.DataFrame()


    def fetch_idx_fundamental(_config, requested, **kwargs):
        requested=list(requested); calls["IDX_FUNDAMENTAL"].append(requested)
        rows=[]
        for ticker in requested:
            item={"ticker":ticker,"idx_official_source_verified":True,"idx_official_period_end":"2026-06-30","idx_official_coverage_pct":90,"idx_official_ocf":100,"idx_official_fcf_proxy":80,"idx_official_cashflow_state":"IDX_OFFICIAL_YTD_OCF_FCF_AVAILABLE"}
            official_store[ticker]=item; rows.append(item)
        audit=pd.DataFrame([{"ticker":t,"provider":"IDX_FIXTURE","status":"OK","items":1} for t in requested])
        writes=[{"cache_key":f"IDX_FUNDAMENTAL:{t}","content_sha256":"x"} for t in requested]
        return pd.DataFrame(rows),audit,writes

    def load_idx_fundamental(_config, requested):
        rows=[official_store[t] for t in requested if t in official_store]
        return pd.DataFrame(rows),pd.DataFrame()

    def persist_memory(_config, **kwargs):
        n=len(kwargs.get("rows") or [])
        return pd.DataFrame([{"state":"RESEARCH_MEMORY_WRITTEN","rows_written":n,"rows_attempted":n}]), pd.DataFrame([{"state":"RESEARCH_MEMORY_VERIFIED_EXACT","rows_expected":n,"rows_verified":n}])

    def load_chunks(_config, _scan_id):
        return pd.DataFrame(chunk_rows)

    def persist_result(_config, **kwargs):
        radar = kwargs["radar"]
        write = pd.DataFrame([
            {"table": "__SUMMARY__", "rows_attempted": len(radar), "rows_written": len(radar), "state": "WRITE_ALL_TABLES"},
            {"table": "cak_radar_snapshots", "rows_attempted": len(radar), "rows_written": len(radar), "state": "WRITTEN"},
        ])
        verify = pd.DataFrame([{"table": "__SUMMARY__", "state": "VERIFIED_ALL_TABLES", "verification_pct": 100.0}])
        commit = pd.DataFrame([{"state": "SCAN_COMPLETED_FULL_PERSISTENCE"}])
        return write, verify, commit

    rs.update_scan_job = update
    rs.record_job_chunk = record
    rs.fetch_ohlcv_cache_first = fetch_ohlcv
    rs.persist_verify_cache_bundle = persist_cache
    rs.cache_commit_succeeded = lambda verify: True
    rs.load_cached_ohlcv_frames = load_ohlcv
    rs.fetch_ksei_cache_first = fetch_ksei
    rs.load_cached_ksei = load_ksei
    rs.fetch_news_cache_first = fetch_news
    rs.load_cached_news = load_news
    rs.fetch_fundamental_cache_first = fetch_fundamental
    rs.load_cached_fundamentals = load_fundamental
    rs.fetch_idx_official_fundamental_cache_first = fetch_idx_fundamental
    rs.load_cached_idx_official_fundamentals = load_idx_fundamental
    rs.persist_verify_research_memory = persist_memory
    rs.load_job_chunks = load_chunks
    rs.persist_verify_scan_best_effort = persist_result

    config = DatabaseConfig(True, "https://fixture.supabase.co", "sb_secret_fixture", key_type="SECRET")
    step_count = 0
    final_result = None
    while job["status"] not in {"COMPLETED", "COMPLETED_PARTIAL_PERSISTENCE"}:
        job, report, result = rs.process_next_job_step(config, copy.deepcopy(job), now=pd.Timestamp("2026-08-06 12:00", tz="Asia/Jakarta"))
        step_count += 1
        if step_count == 7:
            # Simulate browser/session loss. Only JSON-safe database state survives.
            job = json.loads(json.dumps(job, default=str))
        if result is not None:
            final_result = result
        if step_count > 80:
            raise RuntimeError(f"Job did not finish: {job}")

    assert final_result is not None
    assert len(final_result["radar"]) == 300
    assert int(final_result["radar"]["deep_review_state"].eq("DEEP_REVIEWED").sum()) == 300
    # v1.9.3 regression: cached fundamentals must survive reconcile -> radar join.
    radar = final_result["radar"]
    assert "fundamental_conversion_score" in radar.columns
    assert int(pd.to_numeric(radar["fundamental_conversion_score"], errors="coerce").notna().sum()) == 300
    assert int((pd.to_numeric(radar["fundamental_coverage_pct"], errors="coerce") >= 35).sum()) == 300
    assert int((pd.to_numeric(radar["fundamental_data_quality_score"], errors="coerce") >= 35).sum()) == 300
    assert int(radar.get("next_leader_eligible", pd.Series(False, index=radar.index)).fillna(False).sum()) > 0
    assert isinstance(final_result.get("top3_summary"), list)
    assert job["progress_pct"] == 100.0
    assert int((job.get("result_summary") or {}).get("deep_reviewed") or 0) == 300
    assert isinstance((job.get("result_summary") or {}).get("top3"), list)
    assert len(calls["OHLCV"]) == 16  # benchmark + 15 universe chunks
    assert max(len(chunk) for chunk in calls["OHLCV"]) <= 20
    assert sum(len(chunk) for chunk in calls["KSEI"]) == 300
    assert sum(len(chunk) for chunk in calls["NEWS"]) == 300
    assert sum(len(chunk) for chunk in calls["FUNDAMENTAL"]) == 300
    assert sum(len(chunk) for chunk in calls["IDX_FUNDAMENTAL"]) == 60
    assert max(len(chunk) for chunk in calls["KSEI"] + calls["NEWS"] + calls["FUNDAMENTAL"] + calls["IDX_FUNDAMENTAL"]) <= 20
    assert step_count <= 70
    print({
        "state": "PASS",
        "tickers": 300,
        "steps": step_count,
        "ohlcv_calls": len(calls["OHLCV"]),
        "shortlist_ksei": sum(len(chunk) for chunk in calls["KSEI"]),
        "shortlist_news": sum(len(chunk) for chunk in calls["NEWS"]),
        "shortlist_fundamental": sum(len(chunk) for chunk in calls["FUNDAMENTAL"]),
        "resume_after_disconnect": True,
        "radar_rows": len(final_result["radar"]),
        "deep_reviewed": int(final_result["radar"]["deep_review_state"].eq("DEEP_REVIEWED").sum()),
        "top3_persisted_in_job_summary": len((job.get("result_summary") or {}).get("top3") or []),
    })


if __name__ == "__main__":
    main()
