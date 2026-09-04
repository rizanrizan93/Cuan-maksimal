from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import phase56_coverage_runtime_integrity_patch as patch


def _strict_event(ticker: str = "TSPC.JK") -> dict:
    return {
        "ticker": ticker,
        "published_at": "2026-07-15T00:00:00+00:00",
        "title": "Verified issuer joint venture specialty pharma",
        "summary": "joint venture specialty pharma expansion",
        "url": "https://example.com/issuer/tspc",
        "source_tier": "ISSUER",
        "source_verified": True,
        "category": "JOINT_VENTURE_SPECIALTY_PHARMA",
    }


def test_radar_only_future_call_reuses_loaded_strict_forward_evidence() -> None:
    def loader(*args, **kwargs):
        return {"official_forward_events": pd.DataFrame([_strict_event()])}

    def calculator(*args, **kwargs):
        events = kwargs.get("events") if "events" in kwargs else args[1]
        return {"event_count": len(events), "titles": events.get("title", pd.Series(dtype=str)).tolist()}

    fake = SimpleNamespace(
        load_verified_direct_evidence=loader,
        calculate_future_fundamental=calculator,
    )
    patch._wrap_direct_loader(fake)
    patch._wrap_future_calculator(fake)

    fake.load_verified_direct_evidence(None, ["TSPC.JK"])
    result = fake.calculate_future_fundamental(ticker="TSPC.JK", events=pd.DataFrame())

    assert result["event_count"] == 1
    assert "Verified issuer joint venture specialty pharma" in result["titles"]


def test_unverified_forward_row_is_not_cached_or_promoted() -> None:
    row = _strict_event()
    row["source_verified"] = False
    patch._cache_strict_forward_events({"official_forward_events": pd.DataFrame([row])})
    assert "TSPC.JK" not in patch._STRICT_FORWARD_BY_TICKER


def test_strict_event_merge_deduplicates_same_ticker_title_url() -> None:
    frame = pd.DataFrame([_strict_event()])
    out = patch._merge_strict_events(frame, frame)
    assert len(out) == 1


class _FakeEngine:
    @staticmethod
    def round_idx(value, direction):
        return float(value)

    @staticmethod
    def idx_tick(value):
        return 1.0


def test_breakout_plan_is_evaluated_after_accumulation_geometry_failure() -> None:
    features = {
        "last_price": 100.0,
        "atr14": 5.0,
        "ema20": 95.0,
        "high20": 105.0,
        "low20": 90.0,
        "previous_high20": 110.0,
        "prior_high55": 120.0,
        "prior_high120": 135.0,
    }
    rescued = patch._breakout_rescue(
        features,
        ready=False,
        lifecycle="EARLY_CONVERGENCE",
        orderbook=None,
        engine=_FakeEngine,
    )

    assert rescued is not None
    assert rescued["preferred_execution_path"] == "BREAKOUT_RETEST"
    assert rescued["execution_geometry_valid"] is True
    assert rescued["breakout_geometry_valid"] is True
    assert rescued["accumulation_geometry_valid"] is False
    assert rescued["execution_state"] == "RESEARCH_SCENARIO_ONLY"
    assert rescued["execution_stop_loss"] < rescued["execution_entry_reference"] < rescued["execution_tp1"] < rescued["execution_tp2"]


def test_zero_atr_remains_no_geometry_not_fabricated() -> None:
    rescued = patch._breakout_rescue(
        {
            "last_price": 100.0,
            "atr14": 0.0,
            "ema20": 95.0,
            "high20": 105.0,
            "low20": 90.0,
        },
        ready=False,
        lifecycle="EARLY_CONVERGENCE",
        orderbook=None,
        engine=_FakeEngine,
    )
    assert rescued is None


def test_patch_does_not_create_authorization_or_ownership_semantics() -> None:
    source = open("phase56_coverage_runtime_integrity_patch.py", encoding="utf-8").read()
    assert '"execution_authorized"' not in source
    assert '"real_money_ready"' not in source
    assert "regulatory_free_float_pct" not in source
    assert "ksei_provider_state" not in source
