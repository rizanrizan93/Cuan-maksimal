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
