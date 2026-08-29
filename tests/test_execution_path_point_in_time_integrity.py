from __future__ import annotations

from pathlib import Path

import pandas as pd

from narrative_flow_engine import build_execution_plan
from resumable_scan import _point_in_time_event_frame, position_builder


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


def test_central_event_gate_excludes_future_and_undated_rows_before_any_scorer():
    events = pd.DataFrame([
        {"ticker":"A.JK","published_at":"2026-08-01T00:00:00Z","title":"available"},
        {"ticker":"A.JK","published_at":"2026-09-01T00:00:00Z","title":"future"},
        {"ticker":"A.JK","published_at":None,"event_date":None,"title":"undated"},
    ])
    out, excluded = _point_in_time_event_frame(events, as_of="2026-08-14T00:00:00Z")

    assert out["title"].tolist() == ["available"]
    assert excluded == 2
    assert set(out["point_in_time_state"]) == {"AVAILABLE_AS_OF_DECISION"}



def test_synthetic_r_multiple_targets_are_marked_research_only():
    plan = build_execution_plan(
        {
            "last_price": 100.0,
            "atr14": 4.0,
            "ema20": 99.0,
            "high20": 103.0,
            "low20": 94.0,
            "previous_high20": float("nan"),
            "prior_high20": float("nan"),
            "prior_high55": float("nan"),
            "prior_high120": float("nan"),
            "prior_high252": float("nan"),
        },
        ready=True,
        lifecycle="MOMENTUM_TRIGGERED",
        orderbook={
            "precise_trigger_price": 103.0,
            "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
        },
    )

    assert plan["execution_target_basis"] == "R_MULTIPLE_FALLBACK_RESEARCH_ONLY"
    assert plan["execution_targets_structural"] is False


def test_real_money_gate_contains_structural_target_blocker():
    source = (ROOT / "narrative_flow_engine.py").read_text(encoding="utf-8")
    assert 'blockers.append("EXECUTION_TARGETS_NOT_STRUCTURAL")' in source
