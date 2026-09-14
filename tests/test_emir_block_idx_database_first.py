from __future__ import annotations

from datetime import date
from pathlib import Path

from emir_block_idx_eod import (
    BLOCK_IDX_BASE,
    ENDPOINTS,
    endpoint_catalog,
    generic_events,
    normalize_broker_summary,
    normalize_index_summary,
    normalize_stock_summary,
)


def test_verified_endpoint_contract_is_direct_block_idx_only():
    assert BLOCK_IDX_BASE == "https://block.idx.id"
    verified = [row for row in endpoint_catalog() if row["verified"]]
    assert len(verified) >= 12
    assert {row["key"] for row in verified} >= {
        "stock_summary", "index_summary", "broker_summary", "companies",
        "company_profile", "financial_report", "announcements",
        "company_announcements", "uma", "suspension", "issued_history",
        "trading_info",
    }
    assert all(row["path"].startswith("/primary/") for row in verified)
    assert len({row["key"] for row in endpoint_catalog()}) == len(ENDPOINTS)


def test_stock_summary_normalization_keeps_official_foreign_and_microstructure():
    payload = {
        "recordsTotal": 1,
        "data": [{
            "Date": "2026-09-14",
            "StockCode": "TEST",
            "StockName": "Test Emiten",
            "Previous": "100",
            "OpenPrice": "101",
            "High": "108",
            "Low": "99",
            "Close": "106",
            "Volume": "1000000",
            "Value": "105000000",
            "Frequency": "1234",
            "ForeignBuy": "600000",
            "ForeignSell": "250000",
            "ListedShares": "1000000000",
            "TradebleShares": "400000000",
            "Bid": "105",
            "Offer": "106",
            "BidVolume": "50000",
            "OfferVolume": "30000",
        }],
    }
    rows = normalize_stock_summary(
        payload, date(2026, 9, 14),
        "https://block.idx.id/primary/TradingSummary/GetStockSummary?date=20260914",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "TEST"
    assert row["foreign_net"] == 350000
    assert row["bid"] == 105
    assert row["offer"] == 106
    assert row["source_verified"] is True
    assert len(row["payload_hash"]) == 64


def test_requested_date_mismatch_is_rejected():
    payload = {"data": [{"Date": "2026-09-13", "StockCode": "TEST", "Volume": 1, "Value": 1, "Frequency": 1}]}
    assert normalize_stock_summary(payload, date(2026, 9, 14), "https://block.idx.id/x") == []


def test_index_and_market_broker_semantics_remain_separate():
    index = normalize_index_summary(
        {"data": [{"Date": "2026-09-14", "IndexCode": "COMPOSITE", "Close": 8000}]},
        date(2026, 9, 14), "https://block.idx.id/index",
    )
    broker = normalize_broker_summary(
        {"data": [{"Date": "2026-09-14", "IDFirm": "YP", "FirmName": "Broker", "Value": 10, "Volume": 2, "Frequency": 1}]},
        date(2026, 9, 14), "https://block.idx.id/broker",
    )
    assert index[0]["index_code"] == "COMPOSITE"
    assert broker[0]["semantic_scope"] == "MARKET_WIDE_NO_TICKER_BUY_SELL_SPLIT"
    assert "ticker" not in broker[0]


def test_risk_event_normalizer_distinguishes_reopening():
    payload = {"Results": [
        {"Kode": "TEST", "Date": "2026-09-14", "Info_Type": "SPT", "Judul": "Suspensi", "Data_Download": "/a.pdf"},
        {"Kode": "TEST", "Date": "2026-09-15", "Info_Type": "UPT", "Judul": "Pembukaan", "Data_Download": "/b.pdf"},
    ]}
    rows = generic_events(payload, "SUSPENSION", "https://block.idx.id/risk", date(2026, 9, 15))
    assert [row["event_type"] for row in rows] == ["SUSPEND", "UNSUSPEND"]
    assert all(row["ticker"] == "TEST" for row in rows)


def test_database_contract_has_fail_closed_top3_and_no_cte_scope_leak():
    sql = Path("database/migration_v30_emir_block_idx_database_first.sql").read_text()
    assert "CAK_IDX_TOP3_EXECUTION" in sql.upper()
    assert "RR_BELOW_1_8" in sql
    assert "ACTIVE_SUSPENSION" in sql
    assert "OFFICIAL_FUNDAMENTAL_COVERAGE" in sql
    assert "MARKET_WIDE_NO_TICKER_BUY_SELL_SPLIT" in sql
    # Top-3 must read the persisted rank geometry; a previous implementation
    # incorrectly attempted to reference a CTE after its statement ended.
    top3 = sql.split("insert into public.cak_idx_top3_execution", 1)[1]
    assert "from scored" not in top3.lower()
    assert "from public.cak_idx_rank_daily r" in top3


def test_workflow_is_after_close_idempotent_and_dedicated():
    workflow = Path(".github/workflows/emir-block-idx-eod.yml").read_text()
    assert 'cron: "30 10,11 * * 1-5"' in workflow
    assert "cancel-in-progress: false" in workflow
    assert "vbtpwpmkfxzqeuvztcmz.supabase.co" in workflow
    assert "CAK_SCAN_DATABASE_ONLY" in workflow
    assert "--mode" in workflow
