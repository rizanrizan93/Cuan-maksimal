from __future__ import annotations

import numpy as np
import pandas as pd

import autonomous_enrichment as ae
from top3_dashboard import calculate_next_leader_score


def test_absolute_freshness_no_longer_treats_131_day_q1_as_current():
    state, quality, age = ae._period_freshness("2026-03-31", now="2026-08-09T16:00:00+07:00")
    assert 130 <= age <= 132
    assert state == "AGING_QUARTERLY_PERIOD"
    assert quality < 75


def test_calendar_ytd_pair_requires_matching_prior_quarters():
    series = pd.Series(
        [200.0, 100.0, 120.0, 110.0, 100.0, 90.0],
        index=pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"]),
    )
    current, prior, quarters = ae._calendar_ytd_pair(series)
    assert current == 300.0
    assert prior == 190.0
    assert quarters == 2
    assert round(ae._pct_growth(current, prior), 2) == 57.89


def test_turnaround_quarter_positive_but_ytd_negative_is_unconfirmed():
    state, score = ae._growth_consistency_state(35.9, 65.0, -26.0, -35.0, 2)
    assert state == "TURNAROUND_INFLECTION_UNCONFIRMED"
    assert score <= 55


def test_cross_sectional_reference_marks_one_quarter_lag_before_ranking():
    rows = []
    for idx in range(20):
        latest = "2026-06-30" if idx < 5 else "2026-03-31"
        rows.append({
            "ticker": f"T{idx}.JK",
            "fundamental_latest_period": latest,
            "fundamental_coverage_pct": 83.5,
            "fundamental_data_quality_score": 80.5,
            "fundamental_score_cap": 76.0,
            "fundamental_conversion_score": 76.0,
            "fundamental_period_freshness_state": "CURRENT_QUARTERLY_PERIOD",
        })
    out = ae.apply_cross_sectional_fundamental_freshness(pd.DataFrame(rows), now="2026-08-09")
    q2 = out[out["ticker"].eq("T0.JK")].iloc[0]
    q1 = out[out["ticker"].eq("T5.JK")].iloc[0]
    assert q2["fundamental_period_freshness_state"] == "CURRENT_QUARTERLY_PERIOD"
    assert q1["fundamental_period_freshness_state"] == "LAGGING_REPORTING_PERIOD"
    assert q1["fundamental_cross_sectional_reference_period"] == "2026-06-30"
    assert q1["fundamental_period_lag_days"] >= 90
    assert q1["fundamental_conversion_score"] <= 68
    assert q1["fundamental_data_quality_score"] < 70


def _leader_row(**overrides):
    row = {
        "fundamental_conversion_score": 76,
        "fundamental_coverage_pct": 83.5,
        "fundamental_data_quality_score": 80.5,
        "fundamental_cashflow_state": "CASHFLOW_TTM_MISSING",
        "fundamental_official_source_coverage_pct": 0,
        "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK",
        "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
        "fundamental_period_freshness_state": "CURRENT_QUARTERLY_PERIOD",
        "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED",
        "story_runway_score": 50,
        "financial_conversion_score": 42,
        "issuer_alignment_score": 50,
        "sector_leadership_score": 56,
        "smart_money_score": 74,
        "broker_inventory_score": 70,
        "market_structure_score": 80,
        "liquidity_score": 60,
        "ownership_score": 50,
        "distribution_score": 20,
    }
    row.update(overrides)
    return row


def test_next_leader_penalizes_unconfirmed_inflection_and_lagging_period():
    confirmed = calculate_next_leader_score(_leader_row())
    unconfirmed = calculate_next_leader_score(_leader_row(fundamental_growth_consistency_state="TURNAROUND_INFLECTION_UNCONFIRMED"))
    lagging = calculate_next_leader_score(_leader_row(fundamental_period_freshness_state="LAGGING_REPORTING_PERIOD"))
    both = calculate_next_leader_score(_leader_row(
        fundamental_growth_consistency_state="TURNAROUND_INFLECTION_UNCONFIRMED",
        fundamental_period_freshness_state="LAGGING_REPORTING_PERIOD",
    ))
    assert unconfirmed["next_leader_penalty"] >= confirmed["next_leader_penalty"] + 10
    assert lagging["next_leader_penalty"] >= confirmed["next_leader_penalty"] + 10
    assert both["next_leader_score"] < unconfirmed["next_leader_score"]
    assert both["next_leader_score"] < lagging["next_leader_score"]


def test_schema_v5_rejects_old_v4_payload_without_pit_contract():
    import persistent_cache as pc
    payload = {
        "fundamental_cache_schema_version": "4",
        "revenue_growth_qoq_pct": 1.0, "revenue_growth_yoy_pct": 10.0,
        "earnings_growth_qoq_pct": 1.0, "earnings_growth_yoy_pct": 10.0,
        "roe_ttm_pct": 10.0, "roa_ttm_pct": 5.0,
        "interest_bearing_debt_to_equity": 0.2, "total_liabilities_to_equity": 0.4,
        "net_debt_to_equity": 0.1, "current_ratio": 1.5, "cash_to_debt_ratio": 2.0,
        "fundamental_period_alignment_state": "ALIGNED", "fundamental_cashflow_state": "CASHFLOW_TTM_MISSING",
        "fundamental_data_quality_score": 80.0, "fundamental_score_cap": 76.0,
    }
    assert not pc._fundamental_payload_compatible(payload)


def test_yfinance_snapshot_caps_suni_like_latest_quarter_rebound_when_h1_still_negative(monkeypatch):
    from types import SimpleNamespace

    dates = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"])
    # Q2 is +35.9% YoY, but H1 is -26% because Q1 was much weaker than Q1-25.
    q2_25_revenue, q1_25_revenue = 190.0, 248.17
    q2_26_revenue = q2_25_revenue * 1.3589
    h1_26_revenue = (q2_25_revenue + q1_25_revenue) * 0.74
    q1_26_revenue = h1_26_revenue - q2_26_revenue
    q2_25_earnings, q1_25_earnings = 30.0, 60.7
    q2_26_earnings = q2_25_earnings * 1.6503
    h1_26_earnings = (q2_25_earnings + q1_25_earnings) * 0.65
    q1_26_earnings = h1_26_earnings - q2_26_earnings
    income = pd.DataFrame(
        [
            [q2_26_revenue, q1_26_revenue, 180, 170, q2_25_revenue, q1_25_revenue],
            [50, 10, 40, 35, 35, 25],
            [q2_26_earnings, q1_26_earnings, 25, 20, q2_25_earnings, q1_25_earnings],
        ],
        index=["Total Revenue", "Operating Income", "Net Income"], columns=dates,
    )
    balance = pd.DataFrame(
        [[1000]*6, [190]*6, [330]*6, [1500]*6, [600]*6, [220]*6, [180]*6],
        index=["Stockholders Equity", "Total Debt", "Total Liabilities", "Total Assets", "Current Assets", "Current Liabilities", "Cash And Cash Equivalents"],
        columns=dates,
    )
    cashflow = pd.DataFrame([[80, 70, 60, 50, 75, 65], [-15, -14, -13, -12, -14, -13]], index=["Operating Cash Flow", "Capital Expenditure"], columns=dates)
    fake = SimpleNamespace(quarterly_income_stmt=income, quarterly_balance_sheet=balance, quarterly_cash_flow=cashflow)
    monkeypatch.setattr(ae, "yf", SimpleNamespace(Ticker=lambda _: fake))
    monkeypatch.setattr(ae, "_pace_autonomous_request", lambda: None)
    snap, _ = ae.fetch_yfinance_fundamental_snapshot("SUNI.JK")
    assert snap["revenue_growth_yoy_pct"] > 30
    assert snap["revenue_growth_ytd_yoy_pct"] < 0
    assert snap["earnings_growth_yoy_pct"] > 60
    assert snap["earnings_growth_ytd_yoy_pct"] < 0
    assert snap["fundamental_growth_consistency_state"] == "TURNAROUND_INFLECTION_UNCONFIRMED"
    assert snap["fundamental_score_cap"] <= 70
