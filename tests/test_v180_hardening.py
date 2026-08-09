from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import autonomous_enrichment as ae
from narrative_flow_engine import calculate_sector_context, score_narrative_events
from top3_dashboard import enrich_dashboard_scores, select_next_leaders
from research_memory import build_research_memory_rows

ROOT = Path(__file__).resolve().parents[1]


def _statement(rows, dates):
    return pd.DataFrame.from_dict(rows, orient="index", columns=dates).astype(float)


def _fake_ticker(*, ni_prior=50.0, debt=100.0, liabilities=300.0, equity=1000.0, cash=300.0, with_cashflow=True, trailing_only=False):
    dates = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"])
    income = _statement({
        "Total Revenue": [250, 230, 220, 210, 200],
        "Net Income": [100, 90, 80, 70, ni_prior],
        "Operating Income": [120, 110, 100, 90, 80],
    }, dates)
    balance = _statement({
        "Stockholders Equity": [equity]*5,
        "Total Debt": [debt]*5,
        "Total Liabilities Net Minority Interest": [liabilities]*5,
        "Total Assets": [equity+liabilities]*5,
        "Current Assets": [500]*5,
        "Current Liabilities": [250]*5,
        "Cash Cash Equivalents And Short Term Investments": [cash]*5,
    }, dates)
    cashflow = pd.DataFrame()
    if with_cashflow and not trailing_only:
        cashflow = _statement({
            "Operating Cash Flow": [110, 100, 95, 90, 85],
            "Capital Expenditure": [-20, -18, -17, -16, -15],
        }, dates)
    ttm = pd.DataFrame()
    if with_cashflow and trailing_only:
        ttm = _statement({"Operating Cash Flow": [395], "Capital Expenditure": [-71]}, pd.to_datetime(["2026-06-30"]))
    return SimpleNamespace(
        quarterly_income_stmt=income,
        quarterly_balance_sheet=balance,
        quarterly_cash_flow=cashflow,
        ttm_cash_flow=ttm,
    )


def test_fundamental_extreme_base_effect_is_capped(monkeypatch):
    monkeypatch.setattr(ae, "yf", SimpleNamespace(Ticker=lambda _: _fake_ticker(ni_prior=2.0)))
    monkeypatch.setattr(ae, "_pace_autonomous_request", lambda: None)
    snap, audit = ae.fetch_yfinance_fundamental_snapshot("BISI.JK")
    assert "EARNINGS_BASE_EFFECT_EXTREME" in snap["fundamental_growth_quality_state"]
    assert snap["fundamental_conversion_score"] <= 78.0
    assert snap["fundamental_score_cap"] <= 78.0
    assert audit["status"] in {"OK", "PARTIAL"}


def test_fundamental_missing_cashflow_is_fail_closed(monkeypatch):
    monkeypatch.setattr(ae, "yf", SimpleNamespace(Ticker=lambda _: _fake_ticker(with_cashflow=False)))
    monkeypatch.setattr(ae, "_pace_autonomous_request", lambda: None)
    snap, _ = ae.fetch_yfinance_fundamental_snapshot("TEST.JK")
    assert snap["fundamental_cashflow_state"] == "CASHFLOW_TTM_MISSING"
    assert snap["fundamental_score_cap"] <= 72.0
    assert snap["fundamental_conversion_score"] <= 72.0


def test_fundamental_trailing_cashflow_fallback(monkeypatch):
    monkeypatch.setattr(ae, "yf", SimpleNamespace(Ticker=lambda _: _fake_ticker(with_cashflow=True, trailing_only=True)))
    monkeypatch.setattr(ae, "_pace_autonomous_request", lambda: None)
    snap, _ = ae.fetch_yfinance_fundamental_snapshot("MARK.JK")
    assert snap["fundamental_cashflow_state"] == "OCF_FCF_TTM_AVAILABLE"
    assert np.isfinite(snap["operating_cash_flow_ttm"])
    assert np.isfinite(snap["free_cash_flow_proxy_ttm"])


def test_extreme_leverage_caps_fundamental(monkeypatch):
    monkeypatch.setattr(ae, "yf", SimpleNamespace(Ticker=lambda _: _fake_ticker(debt=2200, liabilities=3300, equity=1000, cash=50)))
    monkeypatch.setattr(ae, "_pace_autonomous_request", lambda: None)
    snap, _ = ae.fetch_yfinance_fundamental_snapshot("DGWG.JK")
    assert snap["fundamental_leverage_risk_state"] == "EXTREME_LEVERAGE"
    assert snap["fundamental_score_cap"] <= 58.0
    assert snap["fundamental_conversion_score"] <= 58.0


def test_narrative_cross_ticker_contamination_is_filtered():
    now = pd.Timestamp("2026-08-08", tz="UTC")
    events = pd.DataFrame([
        {"ticker":"BISI.JK", "published_at":now, "title":"Chart dan Harga Saham TLDN — IDX:TLDN", "summary":"TLDN rally", "publisher":"TradingView", "url":"https://example.com/tldn"},
        {"ticker":"BISI.JK", "published_at":now, "title":"BISI catat pertumbuhan penjualan dan laba", "summary":"BISI revenue earnings improve", "publisher":"News", "url":"https://example.com/bisi"},
    ])
    result = score_narrative_events(events, as_of=now, issuer_context={"company_name":"PT BISI INTERNATIONAL Tbk", "sector":"Consumer Non-Cyclicals"})
    assert result["narrative_event_count"] == 1
    assert result["narrative_relevance_filtered_count"] == 1
    assert "BISI" in result["narrative_latest_title"]


def test_sector_rrg_cross_section_produces_rotation_states():
    rows=[]; universe=[]
    specs={"A":(8,2),"B":(-3,4),"C":(5,-3),"D":(-6,-4)}
    for sector,(rs,mom) in specs.items():
        for i in range(3):
            ticker=f"{sector}{i}.JK"
            rows.append({"ticker":ticker,"feature_state":"OK","last_price":110,"ema50":100,"relative_strength60_pct":rs+i*0.1,"relative_strength_momentum_pct":mom+i*0.1,"smart_money_score":60+i})
            universe.append({"ticker":ticker,"sector":sector})
    result=calculate_sector_context(pd.DataFrame(rows), pd.DataFrame(universe))
    states={result[f"{sector}0.JK"]["sector_rrg_state"] for sector in specs}
    assert len(states) >= 3
    assert all(result[f"{sector}0.JK"]["sector_context_method"] == "RRG_PROXY_CROSS_SECTIONAL_CENTERED_NOT_OFFICIAL_RRG" for sector in specs)


def test_next_leader_is_independent_from_execution_state():
    radar=pd.DataFrame([
        {"ticker":"ELSA.JK","company_name":"Elnusa","sector":"Energy","emir_decision_state":"EMIR_DATA_INTEGRITY_BLOCK","fundamental_conversion_score":82,"fundamental_coverage_pct":80,"fundamental_data_quality_score":80,"fundamental_cashflow_state":"OCF_FCF_TTM_AVAILABLE","fundamental_official_source_coverage_pct":0,"fundamental_leverage_risk_state":"BALANCE_SHEET_CAPACITY_OK","story_runway_score":75,"financial_conversion_score":75,"issuer_alignment_score":65,"sector_leadership_score":72,"smart_money_score":65,"broker_inventory_score":60,"market_structure_score":75,"liquidity_score":75,"ownership_score":55,"distribution_score":20,"emir_final_score":45,"deep_review_state":"DEEP_REVIEWED"},
        {"ticker":"JRPT.JK","company_name":"JRPT","sector":"Property","emir_decision_state":"EMIR_WATCH_INVENTORY_COLLECTION","fundamental_conversion_score":60,"fundamental_coverage_pct":75,"fundamental_data_quality_score":70,"fundamental_cashflow_state":"OCF_FCF_TTM_AVAILABLE","fundamental_official_source_coverage_pct":0,"fundamental_leverage_risk_state":"BALANCE_SHEET_CAPACITY_OK","story_runway_score":45,"financial_conversion_score":50,"issuer_alignment_score":55,"sector_leadership_score":50,"smart_money_score":72,"broker_inventory_score":70,"market_structure_score":80,"liquidity_score":40,"ownership_score":50,"distribution_score":0,"emir_final_score":50,"deep_review_state":"DEEP_REVIEWED"},
    ])
    enriched=enrich_dashboard_scores(radar)
    leaders=select_next_leaders(enriched, limit=2)
    assert leaders.iloc[0]["ticker"] == "ELSA.JK"
    assert bool(enriched.loc[enriched.ticker.eq("ELSA.JK"), "next_leader_eligible"].iloc[0]) is True
    assert int(enriched.loc[enriched.ticker.eq("ELSA.JK"), "next_leader_universe_rank"].iloc[0]) == 1


def test_research_memory_builds_versioned_evidence():
    events=pd.DataFrame([{"ticker":"MARK.JK","published_at":"2026-08-08T00:00:00Z","title":"MARK capacity expansion","url":"https://issuer.example","source_tier":"ISSUER","source_verified":True}])
    auto=pd.DataFrame([{"ticker":"MARK.JK","evidence_type":"PUBLIC_FUNDAMENTAL_PROXY","observed_at":"2026-08-08T00:00:00Z","fundamental_latest_period":"2026-06-30","fundamental_conversion_score":80.0,"source_verified":False}])
    rows=build_research_memory_rows("scan-x", events, auto)
    assert {r["family"] for r in rows} == {"NARRATIVE_EVENT", "PUBLIC_FUNDAMENTAL_PROXY"}
    assert all(len(r["memory_id"]) == 64 and len(r["content_sha256"]) == 64 for r in rows)
    assert any(r["effective_period"] == "2026-06-30" for r in rows)


def test_database_v8_migration_contains_research_memory():
    sql=(ROOT/"database/migration_v8.sql").read_text().lower()
    assert "create table if not exists public.cak_research_memory" in sql
    assert "content_sha256" in sql and "last_scan_id" in sql


def test_fundamental_research_memory_fallback_when_provider_fails(monkeypatch):
    import persistent_cache as pc
    payload = {
        "ticker":"ELSA.JK", "fundamental_cache_schema_version":"4",
        "revenue_growth_qoq_pct":5.0,"revenue_growth_yoy_pct":9.0,"earnings_growth_qoq_pct":8.0,"earnings_growth_yoy_pct":29.0,
        "roe_ttm_pct":12.0,"roa_ttm_pct":7.0,"interest_bearing_debt_to_equity":0.2,"total_liabilities_to_equity":0.5,
        "net_debt_to_equity":-0.1,"current_ratio":1.5,"cash_to_debt_ratio":2.0,
        "fundamental_period_alignment_state":"ALIGNED","fundamental_cashflow_state":"OCF_FCF_TTM_AVAILABLE",
        "fundamental_data_quality_score":80.0,"fundamental_score_cap":88.0,
            "fundamental_growth_consistency_state":"QUARTER_AND_YTD_CONFIRMED","fundamental_growth_consistency_score":100.0,
            "revenue_growth_ytd_yoy_pct":9.0,"earnings_growth_ytd_yoy_pct":29.0,"fundamental_provenance_state":"YFINANCE_PUBLIC_FINANCIAL_STATEMENT_PROXY_NOT_OFFICIAL_FILING",
    }
    monkeypatch.setattr(pc, "read_source_cache", lambda *a, **k: {})
    monkeypatch.setattr(pc, "load_latest_research_memory", lambda *a, **k: {"ELSA.JK":[{"payload":payload}]})
    monkeypatch.setattr(pc, "fetch_many_fundamentals", lambda *a, **k: (pd.DataFrame(), pd.DataFrame([{"ticker":"ELSA.JK","provider":"YFINANCE_FUNDAMENTALS","status":"ERROR","items":0,"detail":"network"}])))
    cfg=SimpleNamespace(ready=True,url="x",key="x")
    snapshots,audit,writes=pc.fetch_fundamental_cache_first(cfg,["ELSA.JK"],now="2026-08-08T00:00:00Z")
    assert len(snapshots)==1
    assert audit.iloc[-1]["status"] == "RESEARCH_MEMORY_FALLBACK"
    assert writes == []


def test_research_memory_persist_exact_readback(monkeypatch):
    import research_memory as rm
    rows=build_research_memory_rows("scan-x", pd.DataFrame(), pd.DataFrame([{"ticker":"MARK.JK","evidence_type":"PUBLIC_FUNDAMENTAL_PROXY","fundamental_latest_period":"2026-06-30","fundamental_cache_schema_version":"4"}]))
    monkeypatch.setattr(rm, "database_status", lambda cfg: {"bridge_version":"1.8.0","schema_version":"emir_autonomous_schema_v8","database_mode":"SUPABASE_REST","database_key_type":"SECRET","write_policy":"X"})
    monkeypatch.setattr(rm, "_post_payload_in_chunks", lambda cfg, **kwargs: len(kwargs["payload"]))
    class Resp:
        def json(self):
            return [{"memory_id":r["memory_id"],"content_sha256":r["content_sha256"]} for r in rows]
    monkeypatch.setattr(rm, "_request", lambda *a, **k: Resp())
    cfg=SimpleNamespace(ready=True,url="x",key="x")
    write,verify=rm.persist_verify_research_memory(cfg,scan_id="scan-x",rows=rows)
    assert write.iloc[0]["state"] == "RESEARCH_MEMORY_WRITTEN"
    assert verify.iloc[0]["state"] == "RESEARCH_MEMORY_VERIFIED_EXACT"
