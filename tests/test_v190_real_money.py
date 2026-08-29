from __future__ import annotations

import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autonomous_enrichment import reconcile_fundamental_snapshot
from idx_official_fundamentals import candidate_reporting_periods, parse_idx_xbrl_instance
from narrative_flow_engine import blend_market_context, build_emir_profile
from research_memory import build_research_memory_rows


def _xbrl_fixture() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8"?>
    <xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:id="http://www.idx.co.id/xbrl/taxonomy/2024-01-01">
      <xbrli:context id="CurrentYearDuration"><xbrli:entity><xbrli:identifier scheme="x">TEST</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>2026-01-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate></xbrli:period></xbrli:context>
      <xbrli:context id="PriorYearDuration"><xbrli:entity><xbrli:identifier scheme="x">TEST</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-06-30</xbrli:endDate></xbrli:period></xbrli:context>
      <xbrli:context id="CurrentYearInstant"><xbrli:entity><xbrli:identifier scheme="x">TEST</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2026-06-30</xbrli:instant></xbrli:period></xbrli:context>
      <id:SalesAndRevenue contextRef="CurrentYearDuration" unitRef="IDR">1200</id:SalesAndRevenue>
      <id:SalesAndRevenue contextRef="PriorYearDuration" unitRef="IDR">1000</id:SalesAndRevenue>
      <id:ProfitLossAttributableToParentEntity contextRef="CurrentYearDuration" unitRef="IDR">180</id:ProfitLossAttributableToParentEntity>
      <id:ProfitLossAttributableToParentEntity contextRef="PriorYearDuration" unitRef="IDR">120</id:ProfitLossAttributableToParentEntity>
      <id:OperatingProfitLoss contextRef="CurrentYearDuration" unitRef="IDR">240</id:OperatingProfitLoss>
      <id:NetCashFlowsReceivedFromUsedInOperatingActivities contextRef="CurrentYearDuration" unitRef="IDR">220</id:NetCashFlowsReceivedFromUsedInOperatingActivities>
      <id:PaymentsForAcquisitionOfPropertyPlantAndEquipment contextRef="CurrentYearDuration" unitRef="IDR">50</id:PaymentsForAcquisitionOfPropertyPlantAndEquipment>
      <id:Assets contextRef="CurrentYearInstant" unitRef="IDR">4000</id:Assets>
      <id:Liabilities contextRef="CurrentYearInstant" unitRef="IDR">1200</id:Liabilities>
      <id:Equity contextRef="CurrentYearInstant" unitRef="IDR">2800</id:Equity>
      <id:CurrentAssets contextRef="CurrentYearInstant" unitRef="IDR">1600</id:CurrentAssets>
      <id:CurrentLiabilities contextRef="CurrentYearInstant" unitRef="IDR">800</id:CurrentLiabilities>
      <id:CashAndCashEquivalents contextRef="CurrentYearInstant" unitRef="IDR">500</id:CashAndCashEquivalents>
      <id:ShortTermBankLoans contextRef="CurrentYearInstant" unitRef="IDR">200</id:ShortTermBankLoans>
    </xbrli:xbrl>'''


def test_candidate_period_august_prefers_tw2():
    periods = candidate_reporting_periods(pd.Timestamp("2026-08-08", tz="Asia/Jakarta"))
    assert periods[0] == (2026, "TW2")
    assert (2026, "TW1") in periods


def test_idx_xbrl_extracts_yoy_ocf_and_fcf():
    row = parse_idx_xbrl_instance("TEST.JK", _xbrl_fixture(), source_url="https://idx.test/instance.zip", period_label="TW2")
    assert row["idx_official_source_verified"] is True
    assert row["idx_official_period_end"] == "2026-06-30"
    assert row["idx_official_revenue_growth_yoy_pct"] == 20.0
    assert row["idx_official_earnings_growth_yoy_pct"] == 50.0
    assert row["idx_official_ocf"] == 220
    assert row["idx_official_fcf_proxy"] == 170
    assert row["idx_official_cashflow_state"] == "IDX_OFFICIAL_YTD_OCF_FCF_AVAILABLE"


def test_idx_xbrl_sums_official_productive_asset_capex_tags():
    fixture = _xbrl_fixture().replace(
        b"PaymentsForAcquisitionOfPropertyPlantAndEquipment",
        b"PaymentsForAcquisitionOfPropertyAndEquipment",
    ).replace(
        b"</xbrli:xbrl>",
        b'<id:PaymentsForAcquisitionOfIntangibleAssets contextRef="CurrentYearDuration" unitRef="IDR">20</id:PaymentsForAcquisitionOfIntangibleAssets></xbrli:xbrl>',
    )
    row = parse_idx_xbrl_instance(
        "TEST.JK", fixture, source_url="https://idx.test/instance.zip", period_label="TW2"
    )
    assert row["idx_official_capex_proxy"] == -70
    assert row["idx_official_fcf_proxy"] == 150


def test_official_reconciliation_overrides_proxy_mismatch_and_unlocks_cashflow():
    proxy = {
        "ticker":"BISI.JK", "fundamental_latest_period":"2026-06-30",
        "revenue_latest":2000, "net_income_latest":900, "revenue_growth_yoy_pct":81.19,
        "earnings_growth_yoy_pct":4494.42, "fundamental_data_quality_score":76,
        "fundamental_coverage_pct":80, "fundamental_profitability_score":70,
        "fundamental_cashflow_state":"CASHFLOW_TTM_MISSING",
    }
    official = parse_idx_xbrl_instance("BISI.JK", _xbrl_fixture(), source_url="https://idx.test/instance.zip", period_label="TW2")
    out = reconcile_fundamental_snapshot(proxy, official, now="2026-08-08")
    assert out["revenue_growth_yoy_pct"] == 20.0
    assert out["earnings_growth_yoy_pct"] == 50.0
    assert out["fundamental_authority_state"] == "IDX_OFFICIAL_XBRL_PRIMARY"
    assert out["fundamental_cross_source_state"] == "OFFICIAL_OVERRIDES_PROXY_MISMATCH"
    assert "OCF_FCF_AVAILABLE" in out["fundamental_cashflow_state"]
    assert out["fundamental_official_source_coverage_pct"] >= 80
    assert out["fundamental_conversion_score"] <= out["fundamental_score_cap"]


def test_blended_market_context_is_selective_when_benchmark_soft_but_breadth_healthy():
    b={"market_regime":"RISK_OFF","market_context_score":44,"market_context_coverage_pct":100,"market_distribution_score":63,"market_trend_score":45}
    u={"market_regime":"SELECTIVE","market_context_score":62,"market_context_coverage_pct":85,"market_breadth_above_ema50_pct":61,"market_flow_score":60}
    out=blend_market_context(b,u)
    assert out["market_regime"] == "SELECTIVE"
    assert out["market_context_provenance_state"] == "BLENDED_IHSG_AND_UNIVERSE_BREADTH"


def test_blended_market_context_remains_risk_off_when_both_weak():
    b={"market_regime":"RISK_OFF","market_context_score":30,"market_context_coverage_pct":100,"market_distribution_score":80}
    u={"market_regime":"RISK_OFF","market_context_score":32,"market_context_coverage_pct":85,"market_breadth_above_ema50_pct":20}
    assert blend_market_context(b,u)["market_regime"] == "RISK_OFF"


def _profile_inputs(direct: bool):
    features={
        "feature_state":"OK","smart_money_score":82,"smart_money_coverage_pct":100,"market_structure_score":82,"market_structure_mode":"CONTINUATION_SETUP",
        "trend_score":84,"liquidity_score":80,"distribution_score":5,"crowding_score":28,"execution_friction_score":8,"price_stage":"BASE_TRANSITION",
        "absorption_score":80,"last_price":1000,"ema20":970,"high20":1020,"low20":930,"atr14":30,"previous_high20":1140,"prior_high20":1160,"prior_high55":1240,"prior_high120":1320,"prior_high252":1450,"adtv20_idr":1_000_000_000,"gap_risk_score":0,
        "ohlcv_integrity_state":"VALID","corporate_action_anomaly_flag":False,
    }
    narrative={"narrative_score":78,"narrative_coverage_pct":90,"narrative_state":"MATERIAL_THESIS_CONFIRMED","narrative_verified_source_count":1,"narrative_independent_story_count":2,"financial_conversion_score":75,"issuer_alignment_score":80,"issuer_alignment_coverage_pct":90,"story_runway_score":80,"top_down_catalyst_score":75,"industry_translation_score":75,"retail_adoption_stage":"PRE_RETAIL"}
    broker={"broker_inventory_score":75,"broker_inventory_coverage_pct":80,"broker_inventory_shift_state":"COLLECTION","retail_cannibalisation_risk":0}
    orderbook={"orderbook_trigger_score":75,"orderbook_coverage_pct":90 if direct else 65,"precise_trigger_price":1020,"orderbook_provenance_state":"DIRECT_SOURCE_VERIFIED" if direct else "OHLCV_EOD_MICROSTRUCTURE_PROXY_NOT_LIVE_DEPTH"}
    integrity={"idx_integrity_score":90,"idx_integrity_coverage_pct":90,"idx_integrity_hard_block":False,"idx_integrity_unknown_critical_count":0,"idx_integrity_provenance_state":"DIRECT_SOURCE_VERIFIED" if direct else "AUTO_PUBLIC_KSEI_AND_REGULATORY_NEWS","corporate_action_review_cleared":True}
    fundamental={"fundamental_conversion_score":78,"fundamental_coverage_pct":90,"fundamental_data_quality_score":90,"fundamental_official_source_coverage_pct":85,"fundamental_observed_at":"2026-08-08T09:00:00Z","fundamental_availability_state":"POINT_IN_TIME_OBSERVED","fundamental_cashflow_state":"IDX_OFFICIAL_YTD_OCF_FCF_AVAILABLE","fundamental_cashflow_quality_state":"CASHFLOW_POSITIVE_CONVERTING","operating_cash_flow_ttm":220,"free_cash_flow_proxy_ttm":170,"ocf_conversion_ratio":0.82,"fundamental_period_freshness_state":"CURRENT_QUARTERLY_PERIOD","fundamental_growth_consistency_state":"QUARTER_AND_YTD_CONFIRMED","fundamental_leverage_risk_state":"BALANCE_SHEET_CAPACITY_OK","fundamental_state":"FUTURE_FUNDAMENTAL_SUPPORTIVE"}
    market={"market_regime":"SELECTIVE","market_context_score":65,"market_context_coverage_pct":100}
    sector={"sector_leadership_score":70,"sector_context_coverage_pct":100,"sector_rrg_state":"LEADING","sector_relative_strength_pct":5,"sector_strength_momentum_pct":2}
    return features,narrative,broker,orderbook,integrity,fundamental,market,sector


def test_guarded_real_money_eod_proxy_never_authorizes_position():
    features,narrative,broker,orderbook,integrity,fundamental,market,sector=_profile_inputs(False)
    p=build_emir_profile(ticker="TEST.JK",features=features,narrative=narrative,broker=broker,ownership={"ownership_score":70,"ownership_coverage_pct":70},orderbook=orderbook,market=market,sector=sector,integrity=integrity,fundamental=fundamental,deep_reviewed=True,capital_mode="GUARDED_REAL_MONEY",risk_budget_pct=1.5,max_position_cap_pct=20)
    assert p["auto_eod_ready"] is True
    assert p["production_ready"] is False
    assert p["position_cap_pct"] == 0
    assert p["risk_budget_pct"] <= 0.75
    assert p["entry_authorization_state"] != "SCANNER_AUTHORIZED_DIRECT_VERIFIED"


def test_guarded_direct_verified_can_authorize_but_cap_and_risk_are_bounded():
    features,narrative,broker,orderbook,integrity,fundamental,market,sector=_profile_inputs(True)
    p=build_emir_profile(ticker="TEST.JK",features=features,narrative=narrative,broker=broker,ownership={"ownership_score":70,"ownership_coverage_pct":70},orderbook=orderbook,market=market,sector=sector,integrity=integrity,fundamental=fundamental,deep_reviewed=True,capital_mode="GUARDED_REAL_MONEY",risk_budget_pct=2,max_position_cap_pct=20)
    assert p["real_money_ready"] is True
    assert p["production_ready"] is True
    assert p["position_cap_pct"] <= 10
    assert p["risk_budget_pct"] <= 0.75


def test_official_fundamental_enters_durable_research_memory_as_official():
    official=pd.DataFrame([{"ticker":"OMED.JK","evidence_type":"IDX_OFFICIAL_FUNDAMENTAL","idx_official_period_end":"2026-06-30","idx_official_source_url":"https://idx.test/o.zip","idx_official_source_verified":True,"source_verified":True}])
    rows=build_research_memory_rows("scan-1",pd.DataFrame(),official)
    assert len(rows)==1
    assert rows[0]["family"]=="IDX_OFFICIAL_FUNDAMENTAL"
    assert rows[0]["effective_period"]=="2026-06-30"
    assert rows[0]["official_source"] is True
    assert rows[0]["source_verified"] is True
