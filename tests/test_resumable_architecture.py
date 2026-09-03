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
