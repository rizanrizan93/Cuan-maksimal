import pandas as pd
from top3_dashboard import enrich_dashboard_scores, select_next_leaders


def _row(ticker, state, fund, story, smart):
    return {
        "ticker": ticker, "company_name": ticker.replace(".JK", ""), "sector": "TEST",
        "emir_decision_state": state, "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
        "fundamental_conversion_score": fund, "fundamental_coverage_pct": 83.5,
        "fundamental_data_quality_score": 80.5, "fundamental_cashflow_state": "CASHFLOW_TTM_MISSING",
        "fundamental_official_source_coverage_pct": 0, "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK",
        "story_runway_score": story, "financial_conversion_score": 42, "issuer_alignment_score": 55,
        "sector_leadership_score": 56, "smart_money_score": smart, "broker_inventory_score": 60,
        "market_structure_score": 70, "liquidity_score": 65, "ownership_score": 50, "distribution_score": 20,
        "emir_final_score": 45, "deep_review_state": "DEEP_REVIEWED",
    }


def test_next_leader_survives_execution_and_real_money_block_states():
    radar = pd.DataFrame([
        _row("OMED.JK", "EMIR_DATA_INTEGRITY_BLOCK", 72, 51, 78),
        _row("ELSA.JK", "EMIR_REJECT_SMART_MONEY_DISTRIBUTION", 72, 54, 64),
        _row("MARK.JK", "EMIR_AVOID_RETAIL_EUPHORIA", 72, 52, 58),
    ])
    leaders = select_next_leaders(enrich_dashboard_scores(radar), limit=20)
    assert len(leaders) == 3
    assert set(leaders["ticker"]) == {"OMED.JK", "ELSA.JK", "MARK.JK"}
    assert leaders["next_leader_eligible"].all()


def test_next_leader_still_rejects_genuinely_missing_fundamental_evidence():
    bad = _row("BAD.JK", "EMIR_NO_EDGE_YET", 0, 70, 80)
    bad["fundamental_state"] = "PROVIDER_FAILED"
    bad["fundamental_coverage_pct"] = 0
    bad["fundamental_data_quality_score"] = 0
    leaders = select_next_leaders(enrich_dashboard_scores(pd.DataFrame([bad])), limit=20)
    assert leaders.empty
