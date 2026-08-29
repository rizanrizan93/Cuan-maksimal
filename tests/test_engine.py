from __future__ import annotations

import sys
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_providers import (  # noqa: E402
    _sanitize_ohlcv,
    completed_session_frame,
    parse_universe_frame,
)
from narrative_flow_engine import (  # noqa: E402
    PUBLIC_FORMULA_REGISTRY,
    aggregate_broker_summary,
    build_emir_profile,
    calculate_market_context,
    calculate_market_features,
    calculate_sector_context,
    classify_lifecycle,
    formula_registry_frame,
    parse_idx_integrity,
    parse_orderbook_evidence,
    parse_ownership,
    build_outcome_calibration,
    round_idx,
    score_narrative_events,
)


def synthetic_frame(n: int = 320, seed: int = 7, trend: float = 0.0012) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2025-01-01", periods=n)
    returns = rng.normal(trend, 0.012, n)
    close = 500 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.02, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    volume[-20:] *= np.linspace(1.0, 1.8, 20)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)


def strong_inputs() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    stock = synthetic_frame(trend=0.002)
    bench = synthetic_frame(seed=9, trend=0.0002)
    features = calculate_market_features(stock, bench, as_of=stock.index[-1])
    features.update({
        "smart_money_score": 82.0,
        "smart_money_coverage_pct": 100.0,
        "trend_score": 85.0,
        "liquidity_score": 80.0,
        "distribution_score": 8.0,
        "crowding_score": 40.0,
        "price_stage": "MARKUP",
        "absorption_score": 80.0,
        "market_structure_score": 80.0,
        "market_structure_mode": "CONTINUATION",
        "continuation_price_flow_score": 80.0,
    })
    narrative = {
        "narrative_score": 80.0,
        "narrative_coverage_pct": 85.0,
        "narrative_state": "MATERIAL_THESIS_CONFIRMED",
        "narrative_event_count": 4,
        "narrative_category": "PROJECT_CAPACITY",
        "narrative_latest_title": "Expansion",
        "narrative_materiality_score": 82.0,
        "financial_conversion_score": 78.0,
        "issuer_alignment_score": 80.0,
        "issuer_alignment_coverage_pct": 85.0,
        "retail_adoption_stage": "PRE_RETAIL",
        "narrative_verified_source_count": 1,
        "narrative_official_source_count": 1,
        "narrative_risk_flags": "NO_MAJOR_NARRATIVE_RISK",
        "conversion_path": "REVENUE → MARGIN → EARNINGS",
        "thesis_statement": "Expansion converts to earnings.",
        "story_runway_score": 82.0,
        "top_down_catalyst_score": 80.0,
        "industry_translation_score": 80.0,
    }
    broker = {
        "broker_inventory_score": 80.0,
        "broker_inventory_coverage_pct": 90.0,
        "broker_summary_score": 75.0,
        "broker_summary_coverage_pct": 90.0,
        "broker_inventory_shift_state": "COLLECTION_PERSISTING",
        "retail_exit_score": 80.0,
        "retail_cannibalisation_risk": 10.0,
    }
    ownership = {"ownership_score": 75.0, "ownership_coverage_pct": 90.0, "effective_free_float_pct": 20.0}
    market = {"market_regime": "RISK_ON", "market_context_score": 80.0, "market_context_coverage_pct": 100.0}
    sector = {
        "sector_leadership_score": 75.0,
        "sector_context_coverage_pct": 100.0,
        "sector_state": "LEADING",
        "sector_rrg_state": "LEADING",
    }
    integrity = {
        "idx_integrity_score": 92.0,
        "idx_integrity_coverage_pct": 100.0,
        "idx_integrity_state": "IDX_INTEGRITY_CLEAR",
        "idx_integrity_hard_block": False,
        "idx_integrity_provenance_state": "DIRECT_SOURCE_VERIFIED",
        "corporate_action_review_cleared": True,
    }
    return features, narrative, broker, ownership, market, sector, integrity


def strong_fundamental():
    return {
        "fundamental_conversion_score": 78.0,
        "fundamental_coverage_pct": 85.0,
        "fundamental_data_quality_score": 85.0,
        "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
        "fundamental_provenance_state": "VERIFIED_TEST_FIXTURE",
    }


def test_formula_registry_contract():
    assert len(PUBLIC_FORMULA_REGISTRY) >= 15
    frame = formula_registry_frame()
    assert {"EXPLICIT_PUBLIC", "PUBLIC_SYNTHESIS", "EMPIRICAL_PROXY", "MANUAL_EVIDENCE_REQUIRED"}.issubset(set(frame["provenance_class"]))
    assert frame["formula_id"].is_unique


def test_parse_universe_metadata():
    frame = parse_universe_frame(StringIO(
        "ticker,company_name,sector,theme,macro_theme,secular_trend,catalyst\n"
        "ADMR,Adaro Minerals,Materials,aluminium smelter,downstreaming,energy transition,commercial operation\n"
    ))
    assert frame.iloc[0]["ticker"] == "ADMR.JK"
    assert frame.iloc[0]["sector"] == "Materials"
    assert frame.iloc[0]["macro_theme"] == "downstreaming"


def test_sanitize_yfinance_multiindex():
    base = synthetic_frame(80)
    multi = base.copy()
    multi.columns = pd.MultiIndex.from_tuples([(column, "ADMR.JK") for column in base.columns])
    cleaned = _sanitize_ohlcv(multi)
    assert len(cleaned) == len(base)
    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_completed_session_drops_intraday_bar():
    frame = synthetic_frame(80)
    today = pd.Timestamp("2026-08-03")
    frame.index = pd.bdate_range(end=today, periods=80)
    result = completed_session_frame(frame, now="2026-08-03 14:00:00+07:00", completed_only=True)
    assert result.index.max().date().isoformat() == "2026-07-31"


def test_market_features_and_structure_are_bounded():
    result = calculate_market_features(synthetic_frame(), synthetic_frame(seed=11, trend=0.0003))
    assert result["feature_state"] == "OK"
    for key in (
        "trend_score", "smart_money_score", "absorption_score", "markup_quality_score", "liquidity_score",
        "distribution_score", "crowding_score", "seller_exhaustion_score", "market_structure_score",
        "reversal_score", "continuation_price_flow_score", "sideways_quality_score",
    ):
        assert 0 <= result[key] <= 100
    assert result["market_structure_mode"] in {
        "REVERSAL_SETUP", "CONTINUATION_SETUP", "SIDEWAYS_ACCUMULATION", "NO_CLEAR_STRUCTURE"
    }


def test_market_context():
    result = calculate_market_context(synthetic_frame(trend=0.0015))
    assert result["market_regime"] in {"RISK_ON", "SELECTIVE", "RISK_OFF"}
    assert result["market_context_coverage_pct"] == 100


def test_sector_context_rrg_proxy():
    fast = pd.DataFrame([
        {
            "ticker": "AAA.JK", "feature_state": "OK", "last_price": 110, "ema50": 100,
            "relative_strength60_pct": 12, "relative_strength_momentum_pct": 3, "smart_money_score": 70,
        },
        {
            "ticker": "BBB.JK", "feature_state": "OK", "last_price": 105, "ema50": 100,
            "relative_strength60_pct": 8, "relative_strength_momentum_pct": 2, "smart_money_score": 66,
        },
    ])
    universe = pd.DataFrame([{"ticker": "AAA.JK", "sector": "Energy"}, {"ticker": "BBB.JK", "sector": "Energy"}])
    result = calculate_sector_context(fast, universe)
    assert result["AAA.JK"]["sector_state"] == "LEADING"
    assert result["AAA.JK"]["sector_rrg_state"] == "LEADING"


def test_narrative_scores_top_down_conversion_alignment():
    events = pd.DataFrame([
        {
            "ticker": "ADMR.JK", "published_at": pd.Timestamp.now(tz="UTC") - pd.to_timedelta(2, unit="D"),
            "title": "Adaro Minerals starts commercial operation of aluminium smelter expansion",
            "summary": "Downstream capacity supports industry growth, revenue, margin and cash flow",
            "publisher": "Company Investor Relation", "url": "https://company.example/press-release", "source_tier": "OFFICIAL", "source_verified": True,
            "top_down_catalyst_score": 85, "industry_translation_score": 80,
        },
        {
            "ticker": "ADMR.JK", "published_at": pd.Timestamp.now(tz="UTC") - pd.to_timedelta(4, unit="D"),
            "title": "New aluminium capacity enters commissioning", "publisher": "Public News",
            "url": "https://news.example/article", "source_tier": "PUBLIC_NEWS",
        },
    ])
    result = score_narrative_events(
        events,
        issuer_context={
            "company_name": "Adaro Minerals", "sector": "Materials", "theme": "aluminium smelter",
            "macro_theme": "downstreaming", "secular_trend": "energy transition materials",
            "catalyst": "commercial operation",
        },
    )
    assert result["narrative_event_count"] == 2
    assert result["narrative_score"] > 55
    assert result["financial_conversion_score"] > 50
    assert result["issuer_alignment_score"] > 50
    assert result["story_runway_score"] > 50
    assert "REVENUE" in result["conversion_path"]



def test_syndicated_news_does_not_inflate_independent_story_coverage():
    now = pd.Timestamp("2026-08-03", tz="UTC")
    events = pd.DataFrame([{
        "ticker": "TEST.JK", "published_at": now - pd.to_timedelta(1, unit="D"),
        "title": f"Issuer starts new smelter commercial operation - Publisher {i}",
        "summary": "Capacity supports revenue and earnings",
        "publisher": f"Publisher {i}", "url": f"https://news.example/{i}",
        "source_tier": "PUBLIC_NEWS",
    } for i in range(5)])
    result = score_narrative_events(events, as_of=now, issuer_context={"theme": "smelter", "catalyst": "commercial operation"})
    assert result["narrative_event_count"] == 5
    assert result["narrative_independent_story_count"] == 1
    assert result["narrative_syndication_ratio_pct"] == 80.0
    assert result["narrative_coverage_pct"] < 50

def test_missing_narrative_reduces_fixed_coverage():
    stock = synthetic_frame(trend=0.002)
    features = calculate_market_features(stock, synthetic_frame(seed=9, trend=0.0002), as_of=stock.index[-1])
    profile = build_emir_profile(
        ticker="TEST",
        features=features,
        narrative=score_narrative_events(pd.DataFrame()),
        market={"market_regime": "RISK_ON", "market_context_score": 75, "market_context_coverage_pct": 100},
        deep_reviewed=True,
    )
    assert profile["production_ready"] is False
    assert profile["emir_evidence_coverage_pct"] < 70
    assert profile["emir_decision_state"] in {"EMIR_WAIT_NARRATIVE", "EMIR_EVIDENCE_PENDING", "EMIR_FUNDAMENTAL_EVIDENCE_PENDING", "EMIR_NO_EDGE_YET"}


def test_thesis_ready_waits_for_direct_bid_offer():
    features, narrative, broker, ownership, market, sector, integrity = strong_inputs()
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker, ownership=ownership,
        market=market, sector=sector, integrity=integrity, fundamental=strong_fundamental(), deep_reviewed=True,
    )
    assert profile["thesis_ready"] is True
    assert profile["production_ready"] is False
    assert profile["emir_decision_state"] == "EMIR_THESIS_READY_WAIT_BID_OFFER"
    assert profile["execution_state"] == "THESIS_READY_WAIT_DIRECT_BID_OFFER_TRIGGER"


def test_ready_profile_has_valid_precise_scenario_and_5pct_cap():
    features, narrative, broker, ownership, market, sector, integrity = strong_inputs()
    orderbook = {
        "orderbook_trigger_score": 75.0,
        "orderbook_coverage_pct": 100.0,
        "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
        "precise_trigger_price": float(features["high20"]),
    }
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker, ownership=ownership,
        orderbook=orderbook, market=market, sector=sector, integrity=integrity, fundamental=strong_fundamental(), deep_reviewed=True,
    )
    assert profile["production_ready"] is True
    assert profile["emir_decision_state"] == "EMIR_READY_WITH_PRECISE_TRIGGER"
    assert profile["stop_loss"] < profile["entry_low"] <= profile["entry_high"] < profile["tp1"] < profile["tp2"]
    assert profile["hard_stop_distance_pct"] <= 5.5
    assert profile["trigger_provenance"] == "DIRECT_BID_OFFER_EVIDENCE"


def test_broker_unverified_not_used_for_production():
    frame = pd.DataFrame([{"ticker": "ADMR", "buy_value": 200, "sell_value": 100, "source_verified": False}])
    result = aggregate_broker_summary(frame)["ADMR.JK"]
    assert result["broker_summary_coverage_pct"] == 0
    assert result["broker_summary_provenance_state"] == "UNVERIFIED_NOT_USED_FOR_PRODUCTION"


def test_multi_period_broker_inventory_and_no_owner_inference():
    frame = pd.DataFrame([
        {
            "ticker": "ADMR", "date": "2025-01-01", "buy_value": 200, "sell_value": 80,
            "participant_type": "FUND", "source_verified": True, "crossing_value": 1000,
            "crossing_price": 1400,
        },
        {
            "ticker": "ADMR", "date": "2026-01-01", "buy_value": 220, "sell_value": 90,
            "participant_type": "RETAIL", "retail_proxy_flag": True, "source_verified": True,
        },
    ])
    result = aggregate_broker_summary(frame)["ADMR.JK"]
    assert result["broker_inventory_score"] > 50
    assert result["broker_inventory_coverage_pct"] > 0
    assert result["beneficial_owner_inference_state"] == "NOT_INFERRED_FROM_BROKER_CODE"


def test_orderbook_verified_and_unverified():
    verified = parse_orderbook_evidence(pd.DataFrame([{
        "ticker": "ADMR", "resistance_price": 1500, "offer_lot": 100000, "median_offer_lot": 20000,
        "small_lot_share_pct": 70, "break_seconds": 15, "break_value": 8_000_000_000,
        "source_verified": True,
    }]))["ADMR.JK"]
    assert verified["orderbook_provenance_state"] == "DIRECT_SOURCE_VERIFIED"
    assert verified["precise_trigger_price"] == 1500
    unverified = parse_orderbook_evidence(pd.DataFrame([{"ticker": "ADMR", "source_verified": False}]))["ADMR.JK"]
    assert unverified["orderbook_coverage_pct"] == 0


def test_ownership_effective_float_and_no_guessing():
    result = parse_ownership(pd.DataFrame([{
        "ticker": "ADMR", "free_float_pct": 20, "affiliated_public_holding_pct": 3,
        "owner_alignment_score": 80, "holder_relationship_confidence_pct": 75,
        "source_verified": True,
    }]))["ADMR.JK"]
    assert result["reported_free_float_pct"] == 20
    assert result["effective_free_float_pct"] == 17
    assert result["fake_float_gap_pct"] == 3
    assert parse_ownership(pd.DataFrame()) == {}


def test_lifecycle_story_leads_flow():
    state = classify_lifecycle(
        {"smart_money_score": 45, "distribution_score": 10, "crowding_score": 30, "price_stage": "BASE_TRANSITION"},
        {"narrative_score": 70, "narrative_coverage_pct": 60, "narrative_state": "THESIS_FORMING"},
    )
    assert state == "STORY_LEADS_FLOW"


def test_idx_rounding():
    assert round_idx(199.4, "nearest") == 199
    assert round_idx(501, "up") == 505
    assert round_idx(5101, "up") == 5125


def test_400_ticker_synthetic_contract():
    benchmark = synthetic_frame(seed=100, trend=0.0003)
    rows = []
    for index in range(400):
        features = calculate_market_features(
            synthetic_frame(seed=index + 1, trend=0.0005 + (index % 9) * 0.0001), benchmark
        )
        rows.append(features)
    frame = pd.DataFrame(rows)
    assert len(frame) == 400
    assert frame["smart_money_score"].notna().sum() == 400
    assert frame["market_structure_score"].notna().sum() == 400
    assert frame["feature_state"].eq("OK").all()


def test_idx_integrity_hsc_is_production_block():
    frame = pd.DataFrame([{
        "ticker": "TEST", "source_verified": True, "observed_at": "2026-08-01",
        "listing_board": "MAIN", "hsc_flag": True, "special_monitoring_flag": False,
        "full_call_auction_flag": False, "suspension_flag": False, "uma_flag": False,
        "sanctions_flag": False, "free_float_pct": 18, "over_1pct_disclosure_flag": True,
        "corporate_action_flag": False, "source_url": "https://idx.example/hsc",
    }])
    result = parse_idx_integrity(frame, as_of="2026-08-03 10:00:00+07:00")["TEST.JK"]
    assert result["idx_integrity_hard_block"] is True
    assert result["idx_integrity_state"] == "IDX_INTEGRITY_HARD_BLOCK"
    assert "HIGH_SHAREHOLDING_CONCENTRATION" in result["idx_integrity_block_reasons"]


def test_unresolved_extreme_move_blocks_production_until_corporate_action_cleared():
    features, narrative, broker, ownership, market, sector, integrity = strong_inputs()
    features["corporate_action_anomaly_flag"] = True
    features["ohlcv_integrity_state"] = "CORPORATE_ACTION_REVIEW_REQUIRED"
    integrity["corporate_action_review_cleared"] = False
    orderbook = {
        "orderbook_trigger_score": 75.0, "orderbook_coverage_pct": 100.0,
        "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
        "precise_trigger_price": float(features["high20"]),
    }
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker, ownership=ownership,
        orderbook=orderbook, market=market, sector=sector, integrity=integrity, deep_reviewed=True,
    )
    assert profile["production_ready"] is False
    assert profile["emir_decision_state"] == "EMIR_DATA_INTEGRITY_BLOCK"


def test_outcome_memory_guarded_rejects_adverse_empirical_state():
    outcomes = pd.DataFrame([{
        "ticker": f"T{i}.JK", "outcome_verified": True, "emir_lifecycle": "MOMENTUM_TRIGGERED",
        "market_structure_mode": "CONTINUATION_SETUP", "return_pct": -2 if i < 25 else 1,
        "max_drawdown_pct": -12, "thesis_invalidated": i < 20,
    } for i in range(40)])
    calibration = build_outcome_calibration(outcomes)
    features, narrative, broker, ownership, market, sector, integrity = strong_inputs()
    features["market_structure_mode"] = "CONTINUATION_SETUP"
    orderbook = {
        "orderbook_trigger_score": 75.0, "orderbook_coverage_pct": 100.0,
        "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
        "precise_trigger_price": float(features["high20"]),
    }
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker, ownership=ownership,
        orderbook=orderbook, market=market, sector=sector, integrity=integrity,
        outcome_calibration_map=calibration, calibration_mode="GUARDED", deep_reviewed=True,
    )
    assert profile["outcome_calibration_state"] == "EMPIRICAL_EDGE_REJECTED"
    assert profile["production_ready"] is False
    assert profile["emir_decision_state"] == "EMIR_CALIBRATION_REJECTED"


def test_small_account_capacity_is_not_falsely_blocked():
    features, narrative, broker, ownership, market, sector, integrity = strong_inputs()
    orderbook = {
        "orderbook_trigger_score": 75.0, "orderbook_coverage_pct": 100.0,
        "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
        "precise_trigger_price": float(features["high20"]),
    }
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker, ownership=ownership,
        orderbook=orderbook, market=market, sector=sector, integrity=integrity,
        capital_idr=5_000_000, max_position_cap_pct=20, deep_reviewed=True,
    )
    assert profile["execution_capacity_state"] != "EXECUTION_CAPACITY_BLOCK"
    assert profile["max_safe_position_value_idr"] > profile["requested_position_value_idr"]


def test_unverified_official_label_does_not_satisfy_narrative_provenance_gate():
    now = pd.Timestamp("2026-08-03", tz="UTC")
    events = pd.DataFrame([{
        "ticker": "TEST.JK",
        "published_at": now - pd.to_timedelta(1, unit="D"),
        "title": "Official expansion announcement",
        "summary": "Capacity can support revenue, earnings and cash flow",
        "publisher": "Issuer IR",
        "url": "https://issuer.example/announcement",
        "source_tier": "OFFICIAL",
        "source_verified": False,
    }])
    result = score_narrative_events(
        events,
        as_of=now,
        issuer_context={"theme": "expansion", "catalyst": "commercial operation"},
    )
    assert result["narrative_verified_source_count"] == 0
    assert result["narrative_official_source_count"] == 0
    assert result["narrative_source_provenance_state"] == "UNVERIFIED_PUBLIC_NEWS_ONLY"


def test_core_thesis_cannot_become_production_without_idx_integrity_evidence():
    features, narrative, broker, ownership, market, sector, _ = strong_inputs()
    orderbook = {
        "orderbook_trigger_score": 80.0,
        "orderbook_coverage_pct": 100.0,
        "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
        "precise_trigger_price": float(features["high20"]),
    }
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker,
        ownership=ownership, orderbook=orderbook, market=market, sector=sector,
        integrity={}, fundamental=strong_fundamental(), deep_reviewed=True,
    )
    assert profile["core_thesis_ready"] is True
    assert profile["idx_integrity_ready"] is False
    assert profile["production_ready"] is False
    assert profile["emir_decision_state"] == "EMIR_CORE_THESIS_READY_WAIT_IDX_INTEGRITY"


def test_large_account_is_blocked_when_requested_position_exceeds_liquidity_capacity():
    features, narrative, broker, ownership, market, sector, integrity = strong_inputs()
    features["adtv20_idr"] = 200_000_000.0
    features["liquidity_score"] = 55.0
    orderbook = {
        "orderbook_trigger_score": 80.0,
        "orderbook_coverage_pct": 100.0,
        "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
        "precise_trigger_price": float(features["high20"]),
    }
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker,
        ownership=ownership, orderbook=orderbook, market=market, sector=sector,
        integrity=integrity, capital_idr=1_000_000_000, max_position_cap_pct=20,
        deep_reviewed=True,
    )
    assert profile["execution_capacity_state"] == "EXECUTION_CAPACITY_BLOCK"
    assert profile["production_ready"] is False
    assert profile["position_cap_pct"] == 0


def test_shadow_only_outcome_memory_never_blocks_decision():
    outcomes = pd.DataFrame([{
        "ticker": f"T{i}.JK", "outcome_verified": True,
        "emir_lifecycle": "MOMENTUM_TRIGGERED",
        "market_structure_mode": "CONTINUATION_SETUP",
        "return_pct": -3.0, "max_drawdown_pct": -20.0,
        "thesis_invalidated": True,
    } for i in range(40)])
    calibration = build_outcome_calibration(outcomes)
    features, narrative, broker, ownership, market, sector, integrity = strong_inputs()
    features["market_structure_mode"] = "CONTINUATION_SETUP"
    orderbook = {
        "orderbook_trigger_score": 80.0,
        "orderbook_coverage_pct": 100.0,
        "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
        "precise_trigger_price": float(features["high20"]),
    }
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker,
        ownership=ownership, orderbook=orderbook, market=market, sector=sector,
        integrity=integrity, outcome_calibration_map=calibration,
        calibration_mode="SHADOW_ONLY", deep_reviewed=True,
    )
    assert profile["outcome_calibration_state"] == "EMPIRICAL_EDGE_REJECTED"
    assert profile["emir_decision_state"] != "EMIR_CALIBRATION_REJECTED"


def test_market_context_uses_universe_breadth_proxy_when_benchmark_missing():
    from narrative_flow_engine import calculate_market_context_from_universe

    rows = []
    for i in range(40):
        rows.append({
            "ticker": f"T{i:03d}.JK",
            "feature_state": "OK",
            "last_price": 110 + i,
            "ema50": 100 + i,
            "trend_score": 70,
            "distribution_score": 20,
            "smart_money_score": 65,
            "momentum20_pct": 5,
        })
    result = calculate_market_context_from_universe(pd.DataFrame(rows))
    assert result["market_regime"] in {"RISK_ON", "SELECTIVE", "RISK_OFF"}
    assert result["market_context_provenance_state"] == "UNIVERSE_BREADTH_PROXY_NOT_DIRECT_IHSG"
    assert 55 <= result["market_context_coverage_pct"] <= 85
    assert result["market_proxy_valid_tickers"] == 40


def test_market_context_proxy_fails_closed_with_small_universe():
    from narrative_flow_engine import calculate_market_context_from_universe

    result = calculate_market_context_from_universe(pd.DataFrame([
        {
            "ticker": "AAA.JK", "feature_state": "OK", "last_price": 110,
            "ema50": 100, "trend_score": 70, "distribution_score": 20,
            "smart_money_score": 65, "momentum20_pct": 5,
        }
    ]))
    assert result["market_regime"] == "MARKET_CONTEXT_UNAVAILABLE"
    assert result["market_context_coverage_pct"] == 0


def test_markup_without_narrative_flow_convergence_requires_reaccumulation():
    state = classify_lifecycle(
        {
            "smart_money_score": 70, "distribution_score": 5, "crowding_score": 35,
            "price_stage": "MARKUP",
        },
        {
            "narrative_score": 44, "narrative_coverage_pct": 70,
            "narrative_state": "WEAK_OR_UNCONVERTED", "retail_adoption_stage": "EARLY_AWARENESS",
        },
        {
            "broker_inventory_score": 68,
            "inventory_cycle_phase": "MULTIYEAR_COLLECTION_PERSISTING_PROXY",
            "broker_inventory_shift_state": "MULTIYEAR_COLLECTION_PERSISTING_PROXY",
        },
    )
    assert state == "MARKUP_REACCUMULATION_REQUIRED"


def test_multiyear_inventory_release_is_distribution_not_collection():
    state = classify_lifecycle(
        {"smart_money_score": 70, "distribution_score": 10, "crowding_score": 30, "price_stage": "BASE_TRANSITION"},
        {"narrative_score": 55, "narrative_coverage_pct": 60, "narrative_state": "THESIS_FORMING", "retail_adoption_stage": "EARLY_AWARENESS"},
        {"broker_inventory_score": 68, "inventory_cycle_phase": "MULTIYEAR_INVENTORY_RELEASE_PROXY"},
    )
    assert state == "SMART_MONEY_DISTRIBUTION"


def test_risk_off_leading_sector_exception_is_explicit_and_position_capped():
    features, narrative, broker, ownership, market, sector, integrity = strong_inputs()
    market["market_regime"] = "RISK_OFF"
    sector.update({"sector_rrg_state": "LEADING", "sector_leadership_score": 80, "sector_relative_strength_pct": 12, "sector_strength_momentum_pct": 3})
    # Auto-public integrity/orderbook proxy are not needed to test the macro gate itself.
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker, ownership=ownership,
        orderbook={"orderbook_trigger_score": 70, "orderbook_coverage_pct": 70, "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED", "precise_trigger_price": features["high20"]},
        market=market, sector=sector, integrity=integrity,
        fundamental={"fundamental_conversion_score": 75, "fundamental_coverage_pct": 85},
        deep_reviewed=True, capital_idr=5_000_000,
    )
    assert profile["risk_off_sector_leader_exception"] is True
    assert profile["market_allows_thesis"] is True
    assert profile["position_cap_pct"] <= 5.0
    assert "MARKET_RISK_OFF_SECTOR_LEADER_EXCEPTION_POSITION_CAPPED" in profile["risk_flags"]


def test_missing_fundamental_cannot_be_promoted_to_thesis_or_production():
    features, narrative, broker, ownership, market, sector, integrity = strong_inputs()
    orderbook = {
        "orderbook_trigger_score": 80.0,
        "orderbook_coverage_pct": 100.0,
        "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
        "precise_trigger_price": float(features["high20"]),
    }
    profile = build_emir_profile(
        ticker="TEST", features=features, narrative=narrative, broker=broker,
        ownership=ownership, orderbook=orderbook, market=market, sector=sector,
        integrity=integrity, fundamental={}, deep_reviewed=True,
    )
    assert profile["thesis_ready"] is False
    assert profile["production_ready"] is False
    assert profile["emir_decision_state"] == "EMIR_FUNDAMENTAL_EVIDENCE_PENDING"


def test_parse_user_400_sector_idx_ic_alias():
    frame = parse_universe_frame(StringIO(
        "Ticker,Company,Sector_IDX_IC,Cap_Universe,Sharia_Status,Active_Scan,Universe_Rank\n"
        "MARK,PT Mark Dynamics Indonesia Tbk,Basic Materials,SMALL_MID,YES,YES,10\n"
    ))
    assert frame.iloc[0]["ticker"] == "MARK.JK"
    assert frame.iloc[0]["sector"] == "Basic Materials"
    assert frame.iloc[0]["cap_universe"] == "SMALL_MID"
    assert frame.iloc[0]["sharia_status"] == "YES"


def test_narrative_excludes_events_published_after_historical_as_of():
    as_of = pd.Timestamp("2026-06-30T00:00:00Z")
    events = pd.DataFrame([{
        "ticker": "TEST.JK",
        "published_at": "2026-07-05T00:00:00Z",
        "title": "TEST announces major capacity expansion",
        "summary": "New capacity supports revenue, margin, earnings and cash flow",
        "publisher": "Issuer",
        "url": "https://issuer.example/future-event",
        "source_tier": "ISSUER",
        "source_verified": True,
    }])
    result = score_narrative_events(
        events,
        as_of=as_of,
        issuer_context={"theme": "capacity expansion", "catalyst": "capacity"},
    )
    assert result["narrative_event_count"] == 0
    assert result["narrative_future_event_filtered_count"] == 1
    assert result["narrative_state"] == "NO_ACTIVE_PUBLIC_NARRATIVE"
    assert result["narrative_risk_flags"] == "NO_POINT_IN_TIME_ELIGIBLE_EVENT"
