from __future__ import annotations

from pathlib import Path

import pandas as pd

from resumable_scan import position_builder


ROOT = Path(__file__).resolve().parents[1]


def test_position_builder_sizes_selected_breakout_plan_not_legacy_accumulation_plan():
    row = pd.Series({
        "preferred_execution_path": "BREAKOUT_RETEST",
        "execution_entry_low": 1200.0,
        "execution_entry_high": 1200.0,
        "execution_stop_loss": 1140.0,
        "entry_low": 1000.0,
        "entry_high": 1040.0,
        "stop_loss": 980.0,
        "position_cap_pct": 20.0,
    })
    out = position_builder(row, capital=5_000_000.0, risk_pct=1.0)
    assert out["position_sizing_basis"] == "SELECTED_EXECUTION_PATH"
    assert out["position_sizing_path"] == "BREAKOUT_RETEST"
    assert out["position_sizing_entry"] == 1200.0
    assert out["position_sizing_stop"] == 1140.0
    assert out["position_sizing_risk_per_share"] == 60.0


def test_official_filing_event_is_not_dated_at_financial_period_end():
    source = (ROOT / "resumable_scan.py").read_text(encoding="utf-8")
    assert '"published_at": observed_at' in source
    assert '"financial_period_end": period_end' in source
    assert '"published_at": pd.to_datetime(period_end' not in source
    assert 'if pd.isna(observed_at) or "POINT_IN_TIME" not in availability_state' in source