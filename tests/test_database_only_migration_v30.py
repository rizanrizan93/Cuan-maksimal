from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "database" / "migration_v30_emir_block_idx_database_only.sql").read_text(encoding="utf-8")


def test_all_useful_verified_block_idx_endpoints_are_catalogued():
    paths = {
        "/primary/ExchangeMember/GetBroker",
        "/primary/ListedCompany/GetCompanyProfiles",
        "/primary/ListedCompany/GetCompanyProfilesDetail",
        "/primary/ListedCompany/GetFinancialReport",
        "/primary/ListedCompany/GetProfileAnnouncement",
        "/primary/TradingSummary/GetStockSummary",
        "/primary/TradingSummary/GetIndexSummary",
        "/primary/TradingSummary/GetBrokerSummary",
        "/primary/NewsAnnouncement/GetAllAnnouncement",
        "/primary/NewsAnnouncement/GetUma",
        "/primary/NewsAnnouncement/GetSuspension",
        "/primary/ListingActivity/GetIssuedHistory",
    }
    assert all(path in SQL for path in paths)


def test_database_only_ranking_has_fundamental_smart_money_and_execution_gates():
    required = {
        "growth_score", "profitability_score", "balance_score", "cashflow_score",
        "smart_score", "structure_score", "foreign_positive20>=10", "adtv20>=1000000000",
        "rr1>=1.8", "not overextended", "active_suspension", "recent_dilution",
        "EMIR_BLOCK_IDX_FEATURE_V2", "DATABASE_ONLY",
    }
    assert all(token in SQL for token in required)


def test_storage_and_scheduler_contracts_are_bounded():
    assert "interval '6 months'" in SQL
    assert "latest eight filings per issuer" in SQL
    assert "end_time<now()-interval '14 days'" in SQL
    assert "45 10 * * 1-5" in SQL
    assert "45 11 * * 1-5" in SQL
    assert "cak_prune_storage_v2" in SQL


def test_privileged_functions_are_not_executable_by_client_roles():
    assert SQL.count("revoke all on function") >= 8
    assert "from public,anon,authenticated" in SQL


def test_top3_evidence_bundle_joins_database_evidence():
    sql = (ROOT / "database" / "migration_v32_top3_evidence_bundle.sql").read_text(encoding="utf-8").lower()
    assert "cak_load_latest_idx_top3_v3" in sql
    assert "cak_idx_company_snapshot" in sql
    assert "cak_idx_ownership_snapshot" in sql
    assert "cak_idx_events" in sql
    assert "grant execute" in sql and "service_role" in sql


def test_top3_execution_levels_are_idx_tick_executable():
    sql = (ROOT / "database" / "migration_v33_idx_tick_executable_top3.sql").read_text(encoding="utf-8").lower()
    assert "cak_idx_tick_size_v1" in sql
    assert "cak_idx_floor_to_tick_v1" in sql
    assert "price_fraction_state','idx_tick_executable'" in sql
    for threshold in ("p_price < 200", "p_price < 500", "p_price < 2000", "p_price < 5000"):
        assert threshold in sql


def test_database_deep_900_reports_missing_required_evidence():
    sql = (ROOT / "database" / "migration_v35_database_deep_900_gap_report.sql").read_text(encoding="utf-8").lower()
    assert "target_universe',900" in sql
    assert "limit least(greatest(p_limit,1),900)" in sql
    assert "official_fundamental_metrics" in sql
    assert "official_ownership" in sql
    assert "execution_blocked_missing_critical_evidence" in sql
    assert "from public,anon,authenticated" in sql


def test_xbrl_pipeline_is_bounded_idempotent_and_service_role_only():
    sql = (ROOT / "database" / "migration_v37_xbrl_evidence_pipeline.sql").read_text(encoding="utf-8").lower()
    assert "cak_idx_fundamental_ingestion_queue" in sql
    assert "for update skip locked" in sql
    assert "limit least(greatest(p_limit,1),12)" in sql
    assert "source_file_path like '/portals/%'" in sql
    assert "source_file_size" in sql and "25000000" in sql
    assert "cak_idx_finalize_fundamental_evidence_v1" in sql
    assert "cak_refresh_idx_ranking_v2(v_rank_date)" in sql
    assert "interval '6 months'" in sql
    assert "from public,anon,authenticated" in sql
    assert "to service_role" in sql


def test_gap_report_exposes_worker_and_permanent_unavailability_states():
    sql = (ROOT / "database" / "migration_v38_gap_report_ingestion_state.sql").read_text(encoding="utf-8").lower()
    assert "fundamental_queue_pending" in sql
    assert "fundamental_parse_failed" in sql
    assert "officially_unavailable_fundamental" in sql
    assert "non_ordinary_instrument_excluded" in sql
    assert "execution_blocked_official_filing_unavailable" in sql


def test_edge_worker_only_downloads_official_bounded_xbrl_and_stores_normalized_rows():
    source = (ROOT / "supabase" / "functions" / "emir-xbrl-worker" / "index.ts").read_text(encoding="utf-8")
    assert 'const ATTACHMENT_BASE = "https://block.idx.id"' in source
    assert "MAX_ATTACHMENT_BYTES = 25_000_000" in source
    assert 'job.source_file_path.startsWith("/Portals/")' in source
    assert "ATTACHMENT_REDIRECT_REJECTED" in source
    assert 'from("cak_idx_fundamental_snapshot").upsert' in source
    assert "raw_payload: {" in source
    assert "source_document" not in source
    assert "SUPABASE_SERVICE_ROLE_KEY" in source
    assert "cak_idx_validate_worker_token_v1" in source


def test_xbrl_worker_hotfix_uses_serial_batches_of_three():
    sql = (ROOT / "database" / "migration_v39_xbrl_worker_serial_limit.sql").read_text(encoding="utf-8").lower()
    source = (ROOT / "supabase" / "functions" / "emir-xbrl-worker" / "index.ts").read_text(encoding="utf-8")
    assert "'limit',3" in sql
    assert "Math.min(3" in source
    assert "processPool(jobs, 1" in source


def test_finalizer_restarts_bounded_worker_before_ranking():
    sql = (ROOT / "database" / "migration_v40_finalizer_worker_rescue.sql").read_text(encoding="utf-8").lower()
    assert "stale_worker_reclaimed" in sql
    assert "cak_idx_kick_xbrl_worker_v1()" in sql
    assert "if v_open>0" in sql
    assert "cak_refresh_idx_ranking_v2(v_rank_date)" in sql


def test_xbrl_watchdog_makes_third_stale_attempt_terminal():
    sql = (ROOT / "database" / "migration_v41_xbrl_stalled_job_watchdog.sql").read_text(encoding="utf-8").lower()
    assert "when attempts>=3 then 'parse_failed'" in sql
    assert "cak_idx_watchdog_xbrl_worker_v1" in sql
    assert "*/10 12-15 * * 1-5" in sql
    assert "from public,anon,authenticated" in sql
    assert "to service_role" in sql


def test_xbrl_watchdog_timing_matches_free_plan_wall_clock():
    sql = (ROOT / "database" / "migration_v42_xbrl_watchdog_free_plan_timing.sql").read_text(encoding="utf-8").lower()
    assert "interval '5 minutes'" in sql
    assert "emir-xbrl-watchdog-5m-eod" in sql
    assert "*/5 12-15 * * 1-5" in sql
    assert "when attempts>=3 then 'parse_failed'" in sql


def test_finalizer_reconciles_changed_top900_membership():
    sql = (ROOT / "database" / "migration_v43_xbrl_top900_membership_reconciliation.sql").read_text(encoding="utf-8").lower()
    assert "cak_idx_reconcile_fundamental_queue_v1" in sql
    assert "removed_outside_top900" in sql
    assert "limit 900" in sql
    assert "top900_reconciliation_pending" in sql
    assert "cak_idx_kick_xbrl_worker_v1()" in sql
