from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from autonomous_enrichment import _first_date
from narrative_flow_engine import calculate_market_features


def test_yyyymmdd_parser_does_not_emit_dayfirst_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = [_first_date("20260806") for _ in range(50)]
    assert all(pd.notna(value) for value in values)
    assert not caught


def test_market_features_no_mean_of_empty_slice_when_no_pullbacks():
    dates = pd.bdate_range("2025-01-01", periods=260)
    close = np.linspace(100, 200, len(dates))
    frame = pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": 1_000_000,
    }, index=dates)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = calculate_market_features(frame)
    assert result["feature_state"] == "OK"
    assert not [item for item in caught if "Mean of empty slice" in str(item.message)]
