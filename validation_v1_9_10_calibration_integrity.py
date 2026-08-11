from __future__ import annotations

import math
import numpy as np

from narrative_flow_engine import ENGINE_VERSION, _weighted_fixed
from top3_dashboard import calculate_next_leader_score


def _base_row() -> dict:
    return {
        "fundamental_conversion_score": 80.0,
        "fundamental_coverage_pct": 90.0,
        "fundamental_data_quality_score": 80.0,
        "fundamental_official_source_coverage_pct": 80.0,
        "fundamental_period_freshness_state": "CURRENT_QUARTERLY_PERIOD",
        "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED",
        "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK",
        "fundamental_cashflow_state": "CASHFLOW_TTM_AVAILABLE",
        "fundamental_ytd_quarters_count": 2,
        "revenue_growth_ytd_yoy_pct": 25.0,
        "earnings_growth_ytd_yoy_pct": 30.0,
        "revenue_growth_yoy_pct": 25.0,
        "earnings_growth_yoy_pct": 30.0,
        "net_margin_ttm_pct": 12.0,
        "roe_ttm_pct": 18.0,
        "story_runway_score": 75.0,
        "financial_conversion_score": 75.0,
        "issuer_alignment_score": 75.0,
        "sector_leadership_score": 75.0,
        "smart_money_score": 75.0,
        "broker_inventory_score": 75.0,
        "market_structure_score": 75.0,
        "liquidity_score": 75.0,
        "ownership_score": 75.0,
        "distribution_score": 10.0,
    }


def main() -> None:
    assert ENGINE_VERSION in {"1.9.10-calibration-integrity", "1.9.11-lineage-sector-integrity"}

    score, coverage = _weighted_fixed([(80.0, 1.0, 70.0)])
    assert math.isclose(score, 80.0) and math.isclose(coverage, 70.0), (score, coverage)

    inner_score, inner_cov = _weighted_fixed([(80.0, 1.0, 70.0)])
    outer_score, outer_cov = _weighted_fixed([(inner_score, 1.0, inner_cov)])
    assert math.isclose(outer_score, 80.0) and math.isclose(outer_cov, 70.0), (outer_score, outer_cov)

    score, coverage = _weighted_fixed([(80.0, 0.5, 100.0), (np.nan, 0.5, 0.0)])
    assert math.isclose(score, 80.0) and math.isclose(coverage, 50.0), (score, coverage)

    full = calculate_next_leader_score(_base_row())
    missing_story = _base_row()
    for key in ("story_runway_score", "financial_conversion_score", "issuer_alignment_score"):
        missing_story.pop(key)
    partial = calculate_next_leader_score(missing_story)
    assert partial["next_leader_model_coverage_pct"] < full["next_leader_model_coverage_pct"]
    assert partial["next_leader_quality_pre_confidence"] >= 70.0
    # Missing evidence should reduce confidence/score, but not collapse quality as if 0/100.
    assert full["next_leader_score"] - partial["next_leader_score"] < 18.0, (full, partial)

    missing_flow = _base_row()
    for key in ("smart_money_score", "broker_inventory_score", "market_structure_score", "liquidity_score", "ownership_score"):
        missing_flow.pop(key)
    partial_flow = calculate_next_leader_score(missing_flow)
    assert partial_flow["next_leader_model_coverage_pct"] < full["next_leader_model_coverage_pct"]
    assert partial_flow["next_leader_quality_pre_confidence"] >= 70.0

    print("PASS v1.9.10 calibration integrity")


if __name__ == "__main__":
    main()
