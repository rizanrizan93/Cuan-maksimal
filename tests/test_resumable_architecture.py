from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_migration_v7_contains_job_and_chunk_tables():
    sql = (ROOT / "database" / "migration_v7.sql").read_text()
    assert "create table if not exists public.cak_scan_jobs" in sql.lower()
    assert "create table if not exists public.cak_scan_job_chunks" in sql.lower()
    assert "references public.cak_scan_jobs(scan_id)" in sql.lower()
    assert "grant select, insert, update, delete" in sql.lower()


def test_resumable_pipeline_does_not_claim_background_worker():
    source = (ROOT / "resumable_scan.py").read_text()
    assert "does not create a background worker" in source.lower()
    assert "next session resumes from the last committed checkpoint" in source.lower()


def test_ksei_news_and_fundamental_are_shortlist_stages():
    source = (ROOT / "resumable_scan.py").read_text()
    assert '"KSEI_SHORTLIST"' in source
    assert '"NEWS_SHORTLIST"' in source
    assert '"FUNDAMENTAL_SHORTLIST"' in source
    assert "stage_tickers = [\"^JKSE\"] if stage == \"BENCHMARK\" else tickers if stage == \"OHLCV\" else shortlist" in source


def test_app_recovers_active_job_before_requiring_upload():
    source = (ROOT / "app.py").read_text()
    recovery = source.index("find_unique_active_job(db_config")
    upload_stop = source.index('if universe_file is None:')
    assert recovery < upload_stop
    assert 'st.session_state["emir_active_scan_id"] = recovered_scan_id' in source
    assert 'str(recovered_job.get("status") or "") in {"CREATED", "RUNNING"}' in source


def test_pause_and_auto_continue_are_persisted_to_job_status():
    source = (ROOT / "app.py").read_text()
    assert '{"status": "PAUSED", "last_error": ""}' in source
    assert '{"status": "RUNNING", "last_error": ""}' in source
    assert "update_scan_job(" in source


def test_daily_deep_provider_budget_is_shortlist_bounded_but_final_fundamentals_reuse_full_cache():
    source = (ROOT / "resumable_scan.py").read_text()
    assert '"DAILY_RECALL_80": 80' in source
    assert "DAILY_RECALL_PRIMARY_LIMIT = 55" in source
    assert "DAILY_RECALL_MAX = 80" in source
    assert 'stage_tickers = shortlist' in source
    assert 'stage_tickers = shortlist[: max(1, int(settings.get("official_fundamental_limit") or 400))]' in source
    assert "load_cached_fundamentals(config, tickers)" in source
    assert "load_cached_idx_official_fundamentals(config, tickers)" in source
    assert "for ticker in tickers:" in source


def test_fast_ranking_recall_lane_is_cache_only():
    source = (ROOT / "resumable_scan.py").read_text()
    fast_block = source[source.index('if stage == "FAST_RANKING":'):source.index('if stage == "FINALIZE":')]
    assert "load_cached_fundamentals(" in fast_block
    assert "fetch_fundamental_cache_first(" not in fast_block
    assert "fetch_ksei_cache_first(" not in fast_block
    assert "fetch_news_cache_first(" not in fast_block


def test_market_feature_freshness_is_not_anchored_to_stale_benchmark():
    source = (ROOT / "resumable_scan.py").read_text()
    assert "expected_session_date=benchmark_last_date" not in source
    assert "benchmark_last_date = benchmark.index.max()" not in source
    assert "load_cached_market_features(config, tickers, now=now)" in source
    assert "load_cached_market_features(\n            config, chunk, now=now, force_refresh=force_refresh,\n        )" in source


def test_stale_ohlcv_fallback_cannot_become_current_market_features():
    source = (ROOT / "resumable_scan.py").read_text()
    assert 'current_statuses = {"CACHE_HIT", "INCREMENTAL_REFRESH", "COLD_REFRESH"}' in source
    assert "if frame.empty or ticker not in current_tickers:" in source
    assert "STALE_CACHE_FALLBACK remains auditable evidence" in source
