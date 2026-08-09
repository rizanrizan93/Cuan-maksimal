from io import StringIO
import pandas as pd

from persistent_cache import _fundamental_payload_compatible, _ohlcv_cache_fresh


def test_old_fundamental_cache_rejected():
    old = {"ticker": "MARK.JK", "revenue_growth_pct": 3.38, "fundamental_conversion_score": 73.0}
    assert not _fundamental_payload_compatible(old)


def test_new_fundamental_cache_accepted():
    payload = {
        "fundamental_cache_schema_version": "4",
        "revenue_growth_qoq_pct": 3.0, "revenue_growth_yoy_pct": 23.0,
        "earnings_growth_qoq_pct": 4.0, "earnings_growth_yoy_pct": 19.0,
        "roe_ttm_pct": 25.0, "roa_ttm_pct": 15.0,
        "interest_bearing_debt_to_equity": 0.1, "total_liabilities_to_equity": 0.3,
        "net_debt_to_equity": -0.1, "current_ratio": 2.0, "cash_to_debt_ratio": 3.0,
        "fundamental_period_alignment_state": "ALIGNED", "fundamental_cashflow_state": "OCF_FCF_TTM_AVAILABLE",
        "fundamental_data_quality_score": 85.0, "fundamental_score_cap": 88.0,
        "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED", "fundamental_growth_consistency_score": 100.0,
        "revenue_growth_ytd_yoy_pct": 23.0, "earnings_growth_ytd_yoy_pct": 19.0,
    }
    assert _fundamental_payload_compatible(payload)


def test_recent_but_previous_session_ohlcv_cache_is_not_fresh():
    frame = pd.DataFrame({"Open":[100],"High":[101],"Low":[99],"Close":[100],"Volume":[1000]}, index=pd.to_datetime(["2026-08-06"]))
    row = {"checked_at": "2026-08-07T23:00:00+00:00"}
    assert not _ohlcv_cache_fresh(row, frame, ttl_hours=12, now="2026-08-08 05:30:00+07:00")
