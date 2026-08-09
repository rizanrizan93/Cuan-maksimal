from __future__ import annotations

from top3_dashboard import calculate_next_leader_score


def _base(**overrides):
    row = {
        "ticker": "TEST.JK",
        "company_name": "PT TEST INDUSTRI Tbk",
        "sector": "Industrials",
        "fundamental_conversion_score": 76,
        "fundamental_coverage_pct": 84.9,
        "fundamental_data_quality_score": 82.1,
        "fundamental_cashflow_state": "CASHFLOW_TTM_MISSING",
        "fundamental_official_source_coverage_pct": 0,
        "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK",
        "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
        "fundamental_period_freshness_state": "CURRENT_QUARTERLY_PERIOD",
        "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED",
        "fundamental_ytd_quarters_count": 2,
        "revenue_growth_yoy_pct": 30,
        "earnings_growth_yoy_pct": 50,
        "revenue_growth_ytd_yoy_pct": 25,
        "earnings_growth_ytd_yoy_pct": 40,
        "story_runway_score": 50,
        "financial_conversion_score": 40,
        "issuer_alignment_score": 50,
        "sector_leadership_score": 56,
        "smart_money_score": 60,
        "broker_inventory_score": 55,
        "market_structure_score": 75,
        "liquidity_score": 60,
        "ownership_score": 50,
        "distribution_score": 20,
    }
    row.update(overrides)
    return row


def test_bank_generic_proxy_cannot_gain_industrial_cashflow_advantage():
    standard = calculate_next_leader_score(_base(
        company_name="PT TEST INDUSTRI Tbk", sector="Industrials",
        fundamental_conversion_score=88, fundamental_data_quality_score=96,
        fundamental_cashflow_state="OCF_FCF_TTM_AVAILABLE",
    ))
    bank = calculate_next_leader_score(_base(
        company_name="BANK TEST INDONESIA Tbk, PT", sector="Financials",
        fundamental_conversion_score=88, fundamental_data_quality_score=96,
        fundamental_cashflow_state="OCF_FCF_TTM_AVAILABLE",
    ))
    assert bank["next_leader_sector_model_state"] == "BANK_GENERIC_PROXY_LIMITED"
    assert "BANK_SPECIFIC_RISK_METRICS_NOT_MODELED" in bank["next_leader_quality_flags"]
    assert bank["next_leader_penalty"] >= standard["next_leader_penalty"] + 8
    assert bank["next_leader_score"] < standard["next_leader_score"]


def test_confirmed_business_growth_can_outrank_execution_distribution_noise():
    impc_like = calculate_next_leader_score(_base(
        revenue_growth_ytd_yoy_pct=29.73, earnings_growth_ytd_yoy_pct=51.03,
        revenue_growth_yoy_pct=33.9, earnings_growth_yoy_pct=70.34,
        story_runway_score=30.3, financial_conversion_score=20,
        issuer_alignment_score=29.6, smart_money_score=48,
        broker_inventory_score=45, distribution_score=48.8,
    ))
    slower = calculate_next_leader_score(_base(
        revenue_growth_ytd_yoy_pct=5.62, earnings_growth_ytd_yoy_pct=56.51,
        revenue_growth_yoy_pct=4.51, earnings_growth_yoy_pct=70.67,
        story_runway_score=42.1, financial_conversion_score=34.4,
        issuer_alignment_score=48.8, smart_money_score=77.8,
        broker_inventory_score=70, distribution_score=20,
    ))
    assert impc_like["next_leader_business_momentum_score"] > slower["next_leader_business_momentum_score"]
    # Strong, broad top-line + earnings growth must not be buried solely by current flow/distribution.
    assert impc_like["next_leader_score"] >= slower["next_leader_score"] - 3


def test_earnings_led_low_topline_gets_review_flag_and_penalty():
    row = calculate_next_leader_score(_base(
        revenue_growth_ytd_yoy_pct=5.62, earnings_growth_ytd_yoy_pct=56.51,
    ))
    assert "EARNINGS_LED_LOW_TOPLINE_REVIEW" in row["next_leader_quality_flags"]
    assert row["next_leader_penalty"] >= 8  # no-official 4 + low-topline 4


def test_moderate_distribution_no_longer_directly_penalizes_next_leader():
    low = calculate_next_leader_score(_base(distribution_score=20))
    moderate = calculate_next_leader_score(_base(distribution_score=50))
    extreme = calculate_next_leader_score(_base(distribution_score=75))
    assert moderate["next_leader_penalty"] == low["next_leader_penalty"]
    assert extreme["next_leader_penalty"] >= low["next_leader_penalty"] + 5


def test_loss_making_growth_is_capped_and_flagged():
    profitable = calculate_next_leader_score(_base(
        revenue_growth_ytd_yoy_pct=45, earnings_growth_ytd_yoy_pct=50,
        net_margin_ttm_pct=12, roe_ttm_pct=18,
    ))
    loss = calculate_next_leader_score(_base(
        revenue_growth_ytd_yoy_pct=45, earnings_growth_ytd_yoy_pct=50,
        net_margin_ttm_pct=-6, roe_ttm_pct=-15,
    ))
    assert "LOSS_MAKING_GROWTH_REVIEW" in loss["next_leader_quality_flags"]
    assert loss["next_leader_business_momentum_score"] <= 60
    assert loss["next_leader_score"] < profitable["next_leader_score"]


def test_broad_based_confirmed_ytd_growth_gets_small_quality_bonus():
    broad = calculate_next_leader_score(_base(
        revenue_growth_ytd_yoy_pct=30, earnings_growth_ytd_yoy_pct=50,
        net_margin_ttm_pct=15, roe_ttm_pct=20,
    ))
    narrow = calculate_next_leader_score(_base(
        revenue_growth_ytd_yoy_pct=8, earnings_growth_ytd_yoy_pct=50,
        net_margin_ttm_pct=15, roe_ttm_pct=20,
    ))
    assert broad["next_leader_business_quality_adjustment"] == 3.0
    assert "BROAD_BASED_YTD_GROWTH_CONFIRMED" in broad["next_leader_quality_flags"]
    assert narrow["next_leader_business_quality_adjustment"] == 0.0


def test_bank_generic_model_is_research_only_for_real_money_top3():
    import pandas as pd
    from top3_dashboard import enrich_dashboard_scores, select_real_money_top3
    row = _base(
        ticker="AMAR.JK", company_name="BANK AMAR INDONESIA Tbk, PT", sector="Financials",
        fundamental_conversion_score=76.9, fundamental_data_quality_score=96.4,
        fundamental_cashflow_state="OCF_FCF_TTM_AVAILABLE",
        real_money_candidate=True, real_money_entry_candidate=True, real_money_ready=False,
        real_money_gate_state="REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED",
        real_money_block_reasons="NONE", emir_conviction_score=60,
        dashboard_silent_accum_score=65, dashboard_flow_score=65, rr_tp1=2.0,
    )
    enriched = enrich_dashboard_scores(pd.DataFrame([row]))
    assert enriched.iloc[0]["real_money_candidate"] in (False, 0)
    assert "BANK_SPECIFIC_RISK_METRICS_NOT_MODELED" in enriched.iloc[0]["real_money_block_reasons"]
    assert select_real_money_top3(enriched).empty
