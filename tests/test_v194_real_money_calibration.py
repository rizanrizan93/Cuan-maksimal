from __future__ import annotations

import pandas as pd

from narrative_flow_engine import build_emir_profile
from top3_dashboard import enrich_dashboard_scores, select_real_money_top3


def _profile(*, verified_story: bool = False, strong_public_story: bool = True, liquidity: float = 72.0, capital_mode: str = "GUARDED_REAL_MONEY", risk_budget_pct: float = 0.5):
    features = {
        "feature_state": "OK", "smart_money_score": 78, "smart_money_coverage_pct": 100,
        "market_structure_score": 82, "market_structure_mode": "CONTINUATION_SETUP",
        "trend_score": 86, "liquidity_score": liquidity, "distribution_score": 8,
        "crowding_score": 28, "execution_friction_score": 6, "price_stage": "BASE_TRANSITION",
        "absorption_score": 78, "last_price": 1100, "ema20": 1040, "high20": 1140,
        "low20": 950, "atr14": 44,
        "previous_high20": 1260, "prior_high20": 1280,
        "prior_high55": 1360, "prior_high120": 1480, "prior_high252": 1620,
        "adtv20_idr": 19_000_000_000,
        "gap_risk_score": 0, "ohlcv_integrity_state": "VALID", "corporate_action_anomaly_flag": False,
    }
    narrative = {
        "narrative_score": 55 if strong_public_story else 35,
        "narrative_coverage_pct": 74 if strong_public_story else 45,
        "narrative_state": "WEAK_OR_UNCONVERTED",
        "narrative_verified_source_count": 1 if verified_story else 0,
        "narrative_official_source_count": 0,
        "narrative_event_count": 19 if strong_public_story else 2,
        "narrative_independent_story_count": 14 if strong_public_story else 1,
        "narrative_source_independence_score": 73 if strong_public_story else 20,
        "narrative_materiality_score": 33 if strong_public_story else 20,
        "narrative_contradiction_score": 0,
        "financial_conversion_score": 42, "issuer_alignment_score": 55,
        "issuer_alignment_coverage_pct": 74, "story_runway_score": 52,
        "top_down_catalyst_score": 40, "industry_translation_score": 63,
        "retail_adoption_stage": "EARLY_AWARENESS",
    }
    broker = {
        "broker_inventory_score": 70, "broker_inventory_coverage_pct": 72,
        "broker_inventory_shift_state": "COLLECTION", "retail_cannibalisation_risk": 0,
    }
    orderbook = {
        "orderbook_trigger_score": 60, "orderbook_coverage_pct": 60, "precise_trigger_price": 1140,
        "orderbook_provenance_state": "OHLCV_EOD_MICROSTRUCTURE_PROXY_NOT_LIVE_DEPTH",
    }
    integrity = {
        "idx_integrity_score": 88, "idx_integrity_coverage_pct": 60, "idx_integrity_hard_block": False,
        "idx_integrity_unknown_critical_count": 0,
        "idx_integrity_provenance_state": "AUTO_PUBLIC_KSEI_AND_REGULATORY_NEWS",
        "corporate_action_review_cleared": True,
    }
    fundamental = {
        "fundamental_conversion_score": 76, "fundamental_coverage_pct": 83.5,
        "fundamental_data_quality_score": 80.5, "fundamental_official_source_coverage_pct": 0,
        "fundamental_cashflow_state": "CASHFLOW_TTM_MISSING",
        "fundamental_period_freshness_state": "CURRENT_QUARTERLY_PERIOD",
        "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK",
        "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
    }
    market = {"market_regime": "SELECTIVE", "market_context_score": 50, "market_context_coverage_pct": 94}
    sector = {
        "sector_leadership_score": 55, "sector_context_coverage_pct": 100,
        "sector_rrg_state": "WEAKENING", "sector_relative_strength_pct": 6, "sector_strength_momentum_pct": -11,
    }
    return build_emir_profile(
        ticker="MARK.JK", features=features, narrative=narrative, broker=broker,
        ownership={"ownership_score": 50, "ownership_coverage_pct": 34}, orderbook=orderbook,
        market=market, sector=sector, integrity=integrity, fundamental=fundamental,
        deep_reviewed=True, capital_mode=capital_mode, risk_budget_pct=risk_budget_pct,
        max_position_cap_pct=7.5,
    )


def test_strong_public_narrative_is_manual_condition_not_hard_blocker():
    row = _profile(verified_story=False, strong_public_story=True)
    assert "NO_VERIFIED_NARRATIVE_OR_OFFICIAL_EVENT" not in row["real_money_block_reasons"]
    assert "PUBLIC_NARRATIVE_MANUAL_VERIFY" in row["real_money_manual_conditions"]
    assert row["real_money_narrative_evidence_tier"] == "PUBLIC_PROXY_ACCEPTED_MANUAL"
    assert row["real_money_candidate"] is True
    assert row["real_money_entry_candidate"] is True
    assert row["real_money_gate_state"] == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    assert row["real_money_ready"] is False
    assert row["risk_budget_pct"] <= 0.50
    assert row["guarded_position_cap_after_manual_confirmation_pct"] <= 3.0


def test_weak_public_narrative_remains_blocked():
    row = _profile(verified_story=False, strong_public_story=False)
    assert "NO_VERIFIED_NARRATIVE_OR_OFFICIAL_EVENT" in row["real_money_block_reasons"]
    assert row["real_money_candidate"] is False
    assert row["real_money_gate_state"] == "REAL_MONEY_BLOCKED"


def test_low_liquidity_still_blocks_even_with_good_public_proxy():
    row = _profile(verified_story=False, strong_public_story=True, liquidity=20)
    assert "LIQUIDITY_LT_45" in row["real_money_block_reasons"]
    assert row["real_money_entry_candidate"] is False


def test_real_money_top3_excludes_wait_and_manual_rows_without_authorization():
    rows = pd.DataFrame([
        {
            "ticker": "ELSA.JK", "real_money_candidate": True, "real_money_entry_candidate": False,
            "real_money_ready": False, "emir_decision_state": "EMIR_WAIT_REACCUMULATION",
            "next_leader_score": 70, "fundamental_conversion_score": 76, "fundamental_coverage_pct": 83,
            "fundamental_data_quality_score": 80, "story_runway_score": 70, "financial_conversion_score": 60,
            "sector_leadership_score": 60, "smart_money_score": 75, "broker_inventory_score": 70,
            "market_structure_score": 85, "liquidity_score": 80, "ownership_score": 50,
            "distribution_score": 10, "emir_conviction_score": 70, "dashboard_silent_accum_score": 75,
            "dashboard_flow_score": 72, "rr_tp1": 2.0,
        },
        {
            "ticker": "MARK.JK", "real_money_candidate": True, "real_money_entry_candidate": True,
            "real_money_ready": False, "emir_decision_state": "EMIR_WATCH_INVENTORY_COLLECTION",
            "next_leader_score": 60, "fundamental_conversion_score": 76, "fundamental_coverage_pct": 83,
            "fundamental_data_quality_score": 80, "story_runway_score": 52, "financial_conversion_score": 42,
            "sector_leadership_score": 55, "smart_money_score": 58, "broker_inventory_score": 57,
            "market_structure_score": 81, "liquidity_score": 72, "ownership_score": 50,
            "distribution_score": 2, "emir_conviction_score": 48, "dashboard_silent_accum_score": 62,
            "dashboard_flow_score": 55, "rr_tp1": 1.9,
        },
        {
            "ticker": "OMED.JK", "real_money_candidate": True, "real_money_entry_candidate": True,
            "real_money_ready": False, "emir_decision_state": "EMIR_WATCH_INVENTORY_COLLECTION",
            "next_leader_score": 64, "fundamental_conversion_score": 76, "fundamental_coverage_pct": 83,
            "fundamental_data_quality_score": 80, "story_runway_score": 51, "financial_conversion_score": 42,
            "sector_leadership_score": 53, "smart_money_score": 78, "broker_inventory_score": 75,
            "market_structure_score": 77, "liquidity_score": 55, "ownership_score": 50,
            "distribution_score": 27, "emir_conviction_score": 44, "dashboard_silent_accum_score": 70,
            "dashboard_flow_score": 72, "rr_tp1": 1.9,
        },
        {
            "ticker": "BISI.JK", "real_money_candidate": False, "real_money_entry_candidate": False,
            "real_money_ready": False, "emir_decision_state": "EMIR_WATCH_INVENTORY_COLLECTION",
            "next_leader_score": 50, "fundamental_conversion_score": 76, "fundamental_coverage_pct": 83,
            "fundamental_data_quality_score": 76, "story_runway_score": 25, "financial_conversion_score": 26,
            "sector_leadership_score": 52, "smart_money_score": 81, "broker_inventory_score": 65,
            "market_structure_score": 81, "liquidity_score": 20, "ownership_score": 50,
            "distribution_score": 0, "emir_conviction_score": 48, "dashboard_silent_accum_score": 67,
            "dashboard_flow_score": 71, "rr_tp1": 1.9,
        },
    ])
    enriched = enrich_dashboard_scores(rows)
    out = select_real_money_top3(enriched, limit=3)
    assert out.empty


def test_manual_real_money_candidate_displays_half_percent_risk_even_in_research_scan():
    row = _profile(verified_story=False, strong_public_story=True, capital_mode="RESEARCH", risk_budget_pct=1.0)
    assert row["real_money_entry_candidate"] is True
    assert row["real_money_ready"] is False
    assert row["real_money_gate_state"] == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    assert row["risk_budget_pct"] <= 0.50
