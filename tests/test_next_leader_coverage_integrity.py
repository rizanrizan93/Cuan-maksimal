from top3_dashboard import calculate_next_leader_score


def _row(coverage: float) -> dict[str, float | str]:
    return {
        "fundamental_conversion_score": 82.0,
        "fundamental_coverage_pct": coverage,
        "fundamental_data_quality_score": 90.0,
        "future_fundamental_score": 78.0,
        "future_fundamental_coverage_pct": coverage,
        "story_runway_score": 80.0,
        "financial_conversion_score": 78.0,
        "narrative_coverage_pct": coverage,
        "issuer_alignment_score": 80.0,
        "issuer_alignment_coverage_pct": coverage,
        "sector_leadership_score": 76.0,
        "sector_context_coverage_pct": coverage,
        "smart_money_score": 75.0,
        "smart_money_coverage_pct": coverage,
        "broker_inventory_score": 74.0,
        "broker_inventory_coverage_pct": coverage,
        "market_structure_score": 76.0,
        "liquidity_score": 80.0,
        "ownership_score": 70.0,
        "ownership_coverage_pct": coverage,
        "revenue_growth_ytd_yoy_pct": 22.0,
        "earnings_growth_ytd_yoy_pct": 28.0,
        "fundamental_ytd_quarters_count": 2,
        "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED",
        "fundamental_period_freshness_state": "CURRENT_QUARTERLY_PERIOD",
        "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
        "fundamental_official_source_coverage_pct": coverage,
        "fundamental_cashflow_quality_state": "CASHFLOW_POSITIVE_CONVERTING",
    }


def test_next_leader_coverage_uses_underlying_evidence_not_column_presence():
    thin = calculate_next_leader_score(_row(10.0))
    complete = calculate_next_leader_score(_row(95.0))

    assert thin["next_leader_model_coverage_pct"] < 25.0
    assert complete["next_leader_model_coverage_pct"] > 90.0
    assert thin["next_leader_score"] < complete["next_leader_score"]
