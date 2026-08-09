from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from narrative_flow_engine import build_emir_profile, calculate_sector_context
from resumable_scan import compute_fast_context


def _frame(seed: int, bars: int = 260, end: str = "2026-08-05") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=bars)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, bars)))
    open_ = close * (1 + rng.normal(0, 0.003, bars))
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) * 1.01,
        "Low": np.minimum(open_, close) * 0.99,
        "Close": close,
        "Volume": rng.integers(500_000, 3_000_000, bars),
    }, index=dates)


def test_sector_context_all_nan_relative_strength_is_unknown_without_warning():
    fast = pd.DataFrame([
        {
            "ticker": "A.JK", "feature_state": "OK", "last_price": 100, "ema50": 95,
            "relative_strength60_pct": np.nan, "relative_strength_momentum_pct": np.nan,
            "smart_money_score": 70,
        },
        {
            "ticker": "B.JK", "feature_state": "OK", "last_price": 90, "ema50": 95,
            "relative_strength60_pct": np.nan, "relative_strength_momentum_pct": np.nan,
            "smart_money_score": 60,
        },
    ])
    universe = pd.DataFrame({"ticker": ["A.JK", "B.JK"], "sector": ["TEST", "TEST"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = calculate_sector_context(fast, universe)
    assert result["A.JK"]["sector_rrg_state"] == "UNKNOWN"
    assert result["B.JK"]["sector_rrg_state"] == "UNKNOWN"
    assert not [item for item in caught if "Mean of empty slice" in str(item.message)]


def test_small_universe_benchmark_freshness_uses_coverage_threshold_not_all_tickers():
    tickers = [f"T{i:02d}.JK" for i in range(20)]
    universe = pd.DataFrame({"ticker": tickers, "company_name": "", "sector": "TEST"})
    frames = {ticker: _frame(i + 1) for i, ticker in enumerate(tickers[:18])}
    benchmark = _frame(999, end="2026-08-03")
    result = compute_fast_context(universe, frames, benchmark, as_of=pd.Timestamp("2026-08-06", tz="Asia/Jakarta"))
    freshness = result["benchmark_freshness"]
    assert freshness["universe_reference_count"] == 18
    assert freshness["benchmark_freshness_state"] == "STALE_RELATIVE_TO_UNIVERSE"
    assert freshness["benchmark_usable"] is False


def test_missing_ohlcv_is_data_integrity_block_even_when_not_deep_reviewed():
    features = {
        "feature_state": "INSUFFICIENT_HISTORY",
        "ohlcv_integrity_state": "INSUFFICIENT_HISTORY",
        "market_structure_mode": "NO_CLEAR_STRUCTURE",
        "liquidity_score": 0,
        "distribution_score": 0,
        "crowding_score": 0,
        "execution_friction_score": 0,
        "adtv20_idr": 0,
    }
    profile = build_emir_profile(ticker="FINN.JK", features=features, deep_reviewed=False)
    assert profile["emir_decision_state"] == "EMIR_DATA_INTEGRITY_BLOCK"
    assert profile["action"] == "REFRESH_OR_CLEAR_CORPORATE_ACTION_DATA"
    assert profile["production_ready"] is False
