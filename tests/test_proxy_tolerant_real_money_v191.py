from narrative_flow_engine import build_emir_profile


def _base(fundamental):
    features={"feature_state":"OK","smart_money_score":80,"smart_money_coverage_pct":100,"market_structure_score":80,"market_structure_mode":"CONTINUATION_SETUP","trend_score":82,"liquidity_score":75,"distribution_score":10,"crowding_score":30,"execution_friction_score":10,"price_stage":"BASE_TRANSITION","absorption_score":80,"last_price":1000,"ema20":970,"high20":1020,"low20":930,"atr14":30,"adtv20_idr":1_000_000_000,"gap_risk_score":0,"ohlcv_integrity_state":"VALID","corporate_action_anomaly_flag":False}
    narrative={"narrative_score":78,"narrative_coverage_pct":90,"narrative_state":"MATERIAL_THESIS_CONFIRMED","narrative_verified_source_count":1,"narrative_independent_story_count":2,"financial_conversion_score":75,"issuer_alignment_score":80,"issuer_alignment_coverage_pct":90,"story_runway_score":80,"top_down_catalyst_score":75,"industry_translation_score":75,"retail_adoption_stage":"PRE_RETAIL"}
    broker={"broker_inventory_score":75,"broker_inventory_coverage_pct":80,"broker_inventory_shift_state":"COLLECTION","retail_cannibalisation_risk":0}
    orderbook={"orderbook_trigger_score":75,"orderbook_coverage_pct":90,"precise_trigger_price":1020,"orderbook_provenance_state":"DIRECT_SOURCE_VERIFIED"}
    integrity={"idx_integrity_score":90,"idx_integrity_coverage_pct":90,"idx_integrity_hard_block":False,"idx_integrity_unknown_critical_count":0,"idx_integrity_provenance_state":"DIRECT_SOURCE_VERIFIED","corporate_action_review_cleared":True}
    market={"market_regime":"SELECTIVE","market_context_score":65,"market_context_coverage_pct":100}
    sector={"sector_leadership_score":70,"sector_context_coverage_pct":100,"sector_rrg_state":"LEADING","sector_relative_strength_pct":5,"sector_strength_momentum_pct":2}
    return build_emir_profile(ticker="TEST.JK",features=features,narrative=narrative,broker=broker,ownership={"ownership_score":70,"ownership_coverage_pct":70},orderbook=orderbook,market=market,sector=sector,integrity=integrity,fundamental=fundamental,deep_reviewed=True,capital_mode="GUARDED_REAL_MONEY",risk_budget_pct=0.75,max_position_cap_pct=10)


def test_proxy_fundamental_can_score_and_be_manual_candidate():
    fundamental={"fundamental_conversion_score":74,"fundamental_coverage_pct":83,"fundamental_data_quality_score":82,"fundamental_official_source_coverage_pct":0,"fundamental_cashflow_state":"CASHFLOW_TTM_MISSING","fundamental_period_freshness_state":"CURRENT_QUARTERLY_PERIOD","fundamental_leverage_risk_state":"BALANCE_SHEET_CAPACITY_OK","fundamental_state":"FUTURE_FUNDAMENTAL_SUPPORTIVE"}
    row=_base(fundamental)
    assert row["real_money_candidate"] is True
    assert row["real_money_ready"] is False
    assert row["real_money_gate_state"] == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    assert row["real_money_fundamental_evidence_tier"] == "PUBLIC_PROXY_ACCEPTED_MANUAL"
    assert row["risk_budget_pct"] <= 0.50
    assert row["guarded_position_cap_after_manual_confirmation_pct"] <= 3.0
    assert "PROXY_FUNDAMENTAL_MANUAL_VERIFY" in row["real_money_manual_conditions"]


def test_official_direct_path_can_still_be_ready():
    fundamental={"fundamental_conversion_score":78,"fundamental_coverage_pct":90,"fundamental_data_quality_score":90,"fundamental_official_source_coverage_pct":85,"fundamental_cashflow_state":"IDX_OFFICIAL_YTD_OCF_FCF_AVAILABLE","fundamental_period_freshness_state":"CURRENT_QUARTERLY_PERIOD","fundamental_leverage_risk_state":"BALANCE_SHEET_CAPACITY_OK","fundamental_state":"FUTURE_FUNDAMENTAL_SUPPORTIVE"}
    row=_base(fundamental)
    assert row["real_money_ready"] is True
    assert row["production_ready"] is True
