
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from data_engine import load_ticker_data, normalize_ticker, map_flow_to_score

from setup_common import (
    _safe_float,
    _safe_text,
    _ensure_technical_columns,
    _pivot_series,
    _bar_age,
    _entry_price_from_zone,
    _rr,
    _min_finite,
    _max_finite,
    _phase_from_trend,
    compute_cycle_features,
    build_macro_liquidity_gate,
    compute_institutional_forward_score,
    compute_relative_strength,
    format_score_delta,
    score_to_grade,
)
from breakout_engine import evaluate_breakout_setup
from pullback_engine import evaluate_pullback_setup
from unicorn_engine import evaluate_unicorn_setup
from reversal_engine import evaluate_reversal_setup


def _setup_priority_rank(kind: str) -> int:
    text = str(kind or "").strip().upper()
    order = {
        "UNICORN": 1,
        "SNIPER": 1,
        "BREAKOUT": 2,
        "PULLBACK": 3,
        "REVERSAL": 4,
        "NONE": 9,
        "NO_SETUP": 9,
    }
    return order.get(text, 8)


def _setup_entry_profile(setup_kind: str, entry_buffer_atr: float, stop_loss_atr: float, target_1_atr: float, target_2_atr: float) -> dict:
    kind = str(setup_kind or "").strip().upper()
    profiles = {
        "BREAKOUT": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.60),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.45),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.00),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.10),
            "entry_bias": 0.22,
            "rr_floor_1": 1.65,
            "rr_floor_2": 2.50,
            "max_risk_pct": 0.070,
        },
        "PULLBACK": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.95),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.85),
            "target_1_atr": max(0.0, float(target_1_atr) * 0.95),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.00),
            "entry_bias": 0.34,
            "rr_floor_1": 1.75,
            "rr_floor_2": 2.80,
            "max_risk_pct": 0.075,
        },
        "UNICORN": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.80),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.70),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.05),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.15),
            "entry_bias": 0.28,
            "rr_floor_1": 1.95,
            "rr_floor_2": 3.00,
            "max_risk_pct": 0.080,
        },
        "SNIPER": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.55),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.60),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.00),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.05),
            "entry_bias": 0.18,
            "rr_floor_1": 2.10,
            "rr_floor_2": 3.25,
            "max_risk_pct": 0.085,
        },
        "REVERSAL": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 1.00),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.90),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.00),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.05),
            "entry_bias": 0.32,
            "rr_floor_1": 1.70,
            "rr_floor_2": 2.70,
            "max_risk_pct": 0.078,
        },
    }
    return profiles.get(kind, {
        "entry_buffer_atr": max(0.0, float(entry_buffer_atr)),
        "stop_loss_atr": max(0.0, float(stop_loss_atr)),
        "target_1_atr": max(0.0, float(target_1_atr)),
        "target_2_atr": max(0.0, float(target_2_atr)),
        "entry_bias": 0.30,
        "rr_floor_1": 1.8,
        "rr_floor_2": 2.8,
        "max_risk_pct": 0.08,
    }).copy()


def _estimate_setup_fill_probability(
    setup_kind: str,
    distance_to_entry_atr: float,
    age_bars: float | int | None,
    setup_valid: bool,
    has_liquidity_sweep: bool = False,
    has_mss: bool = False,
    rr1: float | None = None,
    rr2: float | None = None,
    setup_fresh: bool = False,
    entry_zone_width_atr: float | None = None,
    structure_confluence: float | None = None,
) -> float:
    if not setup_valid:
        return 0.0
    kind = str(setup_kind or "").upper()
    p = 55.0
    if kind in {"SNIPER"}:
        p += 10
    elif kind in {"UNICORN"}:
        p += 7
    elif kind in {"BREAKOUT"}:
        p += 4
    elif kind in {"PULLBACK"}:
        p += 5
    elif kind in {"REVERSAL"}:
        p += 3
    if has_liquidity_sweep:
        p += 8
    if has_mss:
        p += 8
    if setup_fresh:
        p += 8
    if np.isfinite(distance_to_entry_atr):
        if distance_to_entry_atr <= 0.3:
            p += 10
        elif distance_to_entry_atr <= 0.8:
            p += 6
        elif distance_to_entry_atr > 2.0:
            p -= 15
    if np.isfinite(age_bars):
        if age_bars <= 3:
            p += 4
        elif age_bars > 10:
            p -= 6
    if np.isfinite(rr1):
        if rr1 >= 2.0:
            p += 5
        elif rr1 < 1.2:
            p -= 8
    if np.isfinite(rr2):
        if rr2 >= 3.0:
            p += 4
        elif rr2 < 2.0:
            p -= 5
    if np.isfinite(entry_zone_width_atr):
        if entry_zone_width_atr <= 0.4:
            p += 5
        elif entry_zone_width_atr > 1.5:
            p -= 5
    if np.isfinite(structure_confluence):
        p += (float(structure_confluence) - 50.0) * 0.15
    return float(np.clip(p, 0.0, 100.0))


def _setup_distance_to_entry_atr(close: float, entry: float, atr_v: float) -> float:
    if not np.isfinite(close) or not np.isfinite(entry) or not np.isfinite(atr_v) or atr_v <= 0:
        return np.nan
    return float((close - entry) / atr_v)


def _setup_structure_confluence_score(
    entry_price: float,
    atr_v: float,
    entry_zone_low: float | None = None,
    entry_zone_high: float | None = None,
    recent_swing_low: float | None = None,
    recent_swing_high: float | None = None,
) -> float:
    score = 50.0
    if np.isfinite(entry_zone_low) and np.isfinite(entry_zone_high) and np.isfinite(entry_price):
        width = abs(entry_zone_high - entry_zone_low) / max(atr_v, 1e-9)
        if width <= 0.35:
            score += 12
        elif width <= 0.7:
            score += 6
        else:
            score -= 6
        mid = (entry_zone_low + entry_zone_high) / 2.0
        score += max(0.0, 8.0 - abs(entry_price - mid) / max(atr_v, 1e-9) * 10.0)
    if np.isfinite(recent_swing_low) and np.isfinite(entry_price):
        score += 5 if entry_price >= recent_swing_low else -3
    if np.isfinite(recent_swing_high) and np.isfinite(entry_price):
        score += 4 if entry_price <= recent_swing_high else 0
    return float(np.clip(score, 0.0, 100.0))


def _projected_entry_flow(
    setup_kind: str,
    entry_zone_low: float | None = None,
    entry_zone_high: float | None = None,
    support_anchor: float | None = None,
    resistance_anchor: float | None = None,
    breakout_reference: float | None = None,
    fvg_bottom: float | None = None,
    fvg_top: float | None = None,
    breaker_bottom: float | None = None,
    sweep_low: float | None = None,
) -> dict:
    kind = str(setup_kind or "").upper()
    if kind in {"BREAKOUT"}:
        first = "Break retest"
        rebound = "Range expansion"
        zone = "Resistance retest"
        label = "Breakout Retest"
    elif kind in {"PULLBACK"}:
        first = "Pullback to support"
        rebound = "Trend continuation"
        zone = "Support reclaim"
        label = "Pullback Continuation"
    elif kind in {"UNICORN", "SNIPER"}:
        first = "Sweep then mitigate"
        rebound = "FVG rebound"
        zone = "FVG mitigation"
        label = "Unicorn / Sniper"
    elif kind in {"REVERSAL"}:
        first = "Base reclaim"
        rebound = "Reversal expansion"
        zone = "Accumulation reclaim"
        label = "Reversal Accumulation"
    else:
        first = "No setup"
        rebound = "No setup"
        zone = "Wait"
        label = "None"
    retest_anchor = _safe_float(
            breakout_reference if np.isfinite(_safe_float(breakout_reference, np.nan)) else
            fvg_bottom if np.isfinite(_safe_float(fvg_bottom, np.nan)) else
            support_anchor if np.isfinite(_safe_float(support_anchor, np.nan)) else np.nan,
            np.nan
        )
    return {
        "projected_first_leg": first,
        "projected_rebound_leg": rebound,
        "entry_zone_role": zone,
        "entry_zone_label": label,
        "entry_projection_summary": f"{label}: {zone} -> {rebound}",
        "retest_anchor": retest_anchor,
        "sweep_low": sweep_low,
        "projected_entry_anchor": retest_anchor,
        "projected_entry_zone_low": _safe_float(entry_zone_low, np.nan),
        "projected_entry_zone_high": _safe_float(entry_zone_high, np.nan),
        "projected_entry_price": _safe_float(entry_zone_low + (entry_zone_high - entry_zone_low) * 0.5 if np.isfinite(_safe_float(entry_zone_low, np.nan)) and np.isfinite(_safe_float(entry_zone_high, np.nan)) else np.nan, np.nan),
    }



def _project_entry_zone_for_setup(
    setup_kind: str,
    close: float,
    atr_v: float,
    support_anchor: float | None = None,
    resistance_anchor: float | None = None,
    breakout_reference: float | None = None,
    fvg_bottom: float | None = None,
    fvg_top: float | None = None,
    breaker_bottom: float | None = None,
    sweep_low: float | None = None,
) -> dict:
    """Project the next bullish retrace / retest zone from the active setup structure."""
    kind = str(setup_kind or "").upper()
    close = _safe_float(close, np.nan)
    atr_v = _safe_float(atr_v, np.nan)
    if not np.isfinite(close) or close <= 0:
        close = np.nan
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(float(close) * 0.02 if np.isfinite(close) else 1.0, 1.0)

    def _cap_below_price(high_value: float) -> float:
        if not np.isfinite(close):
            return high_value
        cap = close - max(atr_v * 0.05, close * 0.001)
        return float(min(high_value, cap))

    if kind in {"UNICORN", "SNIPER"}:
        anchor = _min_finite([fvg_bottom, breaker_bottom, sweep_low, support_anchor], default=np.nan)
        if not np.isfinite(anchor):
            anchor = close - atr_v * 0.45
        zone_low = max(0.0, min(anchor - atr_v * 0.16, close - atr_v * 0.55))
        zone_high = max(zone_low, min(anchor + atr_v * 0.14, close - atr_v * 0.05))
        entry_bias = 0.34
        label = "Unicorn / Sniper"
        zone_role = "FVG mitigation"
        first_leg = "Sweep then mitigate"
        rebound_leg = "FVG rebound"
        trigger = "Liquidity_Sweep_MSS_FVG_Retest"
    elif kind == "BREAKOUT":
        anchor = _min_finite([breakout_reference, resistance_anchor], default=np.nan)
        if not np.isfinite(anchor):
            anchor = close - atr_v * 0.30
        zone_low = max(0.0, min(anchor - atr_v * 0.12, close - atr_v * 0.45))
        zone_high = max(zone_low, min(anchor + atr_v * 0.10, close - atr_v * 0.05))
        entry_bias = 0.26
        label = "Breakout Retest"
        zone_role = "Resistance retest"
        first_leg = "Break retest"
        rebound_leg = "Range expansion"
        trigger = "Breakout_Retest"
    elif kind == "PULLBACK":
        anchor = _min_finite([support_anchor], default=np.nan)
        if not np.isfinite(anchor):
            anchor = close - atr_v * 0.35
        zone_low = max(0.0, min(anchor - atr_v * 0.14, close - atr_v * 0.50))
        zone_high = max(zone_low, min(anchor + atr_v * 0.12, close - atr_v * 0.05))
        entry_bias = 0.36
        label = "Pullback Continuation"
        zone_role = "Support reclaim"
        first_leg = "Pullback to support"
        rebound_leg = "Trend continuation"
        trigger = "Trend_Pullback_Retest"
    elif kind == "REVERSAL":
        anchor = _min_finite([support_anchor, sweep_low], default=np.nan)
        if not np.isfinite(anchor):
            anchor = close - atr_v * 0.40
        zone_low = max(0.0, min(anchor - atr_v * 0.18, close - atr_v * 0.55))
        zone_high = max(zone_low, min(anchor + atr_v * 0.10, close - atr_v * 0.05))
        entry_bias = 0.32
        label = "Reversal Accumulation"
        zone_role = "Accumulation reclaim"
        first_leg = "Base reclaim"
        rebound_leg = "Reversal expansion"
        trigger = "Liquidity_Sweep_MSS_Reclaim"
    else:
        anchor = close - atr_v * 0.30 if np.isfinite(close) else np.nan
        zone_low = max(0.0, anchor - atr_v * 0.10) if np.isfinite(anchor) else np.nan
        zone_high = max(zone_low, anchor + atr_v * 0.08) if np.isfinite(zone_low) else np.nan
        entry_bias = 0.30
        label = "None"
        zone_role = "Wait"
        first_leg = "No setup"
        rebound_leg = "No setup"
        trigger = "No_signal"

    if np.isfinite(zone_low) and np.isfinite(zone_high):
        zone_high = _cap_below_price(zone_high)
        if zone_high < zone_low:
            zone_high = zone_low + max(atr_v * 0.05, 0.01)
        entry_price = _entry_price_from_zone(zone_low, zone_high, bias=entry_bias)
    else:
        entry_price = np.nan

    if np.isfinite(entry_price) and np.isfinite(close) and entry_price >= close:
        entry_price = min(close - max(atr_v * 0.08, close * 0.002), zone_high if np.isfinite(zone_high) else close)
        entry_price = max(entry_price, zone_low if np.isfinite(zone_low) else entry_price)

    return {
        "entry_zone_low": zone_low,
        "entry_zone_high": zone_high,
        "entry_price": entry_price,
        "entry_trigger": trigger,
        "projected_first_leg": first_leg,
        "projected_rebound_leg": rebound_leg,
        "entry_zone_role": zone_role,
        "entry_zone_label": label,
        "entry_projection_summary": f"{label}: {zone_role} @ {_safe_float(zone_low, np.nan):.2f} - {_safe_float(zone_high, np.nan):.2f} -> {rebound_leg}",
        "projected_entry_price": entry_price,
        "projected_entry_zone": f"{_safe_float(zone_low, np.nan):.2f}-{_safe_float(zone_high, np.nan):.2f}",
        "retest_anchor": anchor,
    }


def _build_setup_lifecycle_snapshot(stock_res: dict, plan: dict | None, plan_reason: str = "") -> dict:

    kind = str((plan or {}).get("setup_kind") or "NONE").upper()
    valid = bool((plan or {}).get("entry_valid", False))
    if valid:
        stage = "ENTRY_ZONE"
        next_action = "MONITOR_ENTRY"
    elif kind != "NONE":
        stage = "WATCHLIST"
        next_action = "WAIT"
    else:
        stage = "NO_SETUP"
        next_action = "WAIT"
    return {
        "setup_lifecycle_stage": stage,
        "setup_validity_ok": valid,
        "setup_validity_reason": plan_reason or (plan or {}).get("plan_reason", "n/a"),
        "setup_next_action": next_action,
    }


def _combine_setup_states(d: pd.DataFrame) -> dict:
    breakout = evaluate_breakout_setup(d)
    pullback = evaluate_pullback_setup(d)
    unicorn = evaluate_unicorn_setup(d)
    reversal = evaluate_reversal_setup(d)
    return {
        "breakout": breakout,
        "pullback": pullback,
        "unicorn": unicorn,
        "reversal": reversal,
    }


def _build_consistent_entry_plan(
    stock_res: dict,
    entry_buffer_atr: float = 0.25,
    stop_loss_atr: float = 1.8,
    target_1_atr: float = 2.2,
    target_2_atr: float = 3.8,
) -> dict:
    d = stock_res.get("df")
    if d is None or getattr(d, "empty", True):
        return {
            "entry_zone_low": np.nan,
            "entry_zone_high": np.nan,
            "entry_price_plan": np.nan,
            "entry_trigger": "No_signal",
            "stop_loss_plan": np.nan,
            "target_1": np.nan,
            "target_2": np.nan,
            "risk_per_share": np.nan,
            "risk_reward_1": np.nan,
            "risk_reward_2": np.nan,
            "upside_to_t1_pct": np.nan,
            "upside_to_t2_pct": np.nan,
            "plan_reason": "No actionable buy signal",
            "setup_kind": "None",
            "setup_variant": "None",
            "breakout_confirmed": False,
            "breakout_reference": np.nan,
            "breakout_invalidation_level": np.nan,
            "pullback_valid": False,
            "pullback_reference": np.nan,
            "pullback_invalidation_level": np.nan,
            "entry_valid": False,
            "entry_mode": "None",
            "tradeability_ok": False,
            "tradeability_score": np.nan,
            "tradeability_reason": "n/a",
            "tradeability_tier": "n/a",
            "tradeability_gate_reason": "n/a",
            "setup_lifecycle_stage": "NO_SETUP",
            "setup_validity_ok": False,
            "setup_validity_reason": "n/a",
            "setup_next_action": "WAIT",
            "setup_age_bars": np.nan,
            "setup_age_limit": np.nan,
            "setup_distance_to_entry_pct": np.nan,
            "setup_rr_1": np.nan,
            "setup_rr_2": np.nan,
            "spread_proxy_20d": np.nan,
            "gap_proxy_20d": np.nan,
            "avg_value_traded_20d": np.nan,
            "tradeability_threshold": np.nan,
            "tradeability_value_floor": np.nan,
            "tradeability_spread_cap": np.nan,
            "tradeability_gap_cap": np.nan,
            "execution_status": "WAIT",
            "execution_status_reason": "n/a",
            "entry_candidate_label": "NONE",
            "candidate_entry_price": np.nan,
            "candidate_stop_price": np.nan,
            "candidate_target_1": np.nan,
            "candidate_target_2": np.nan,
            "candidate_entry_zone_low": np.nan,
            "candidate_entry_zone_high": np.nan,
            "candidate_risk_reward_1": np.nan,
            "candidate_risk_reward_2": np.nan,
            "setup_states": {},
        }

    d = _ensure_technical_columns(d.copy())
    last = d.iloc[-1]
    close = _safe_float(last.get("Close"), np.nan)
    open_price = _safe_float(last.get("Open"), np.nan)
    low_price = _safe_float(last.get("Low"), np.nan)
    atr_v = _safe_float(last.get("ATR14"), np.nan)
    ema20 = _safe_float(last.get("EMA20"), np.nan)
    ema50 = _safe_float(last.get("EMA50"), np.nan)
    ema200 = _safe_float(last.get("EMA200"), np.nan)
    if not np.isfinite(close) or close <= 0:
        return _build_consistent_entry_plan({"df": pd.DataFrame()}, entry_buffer_atr, stop_loss_atr, target_1_atr, target_2_atr)
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close * 0.02, 1.0)

    setup_states = _combine_setup_states(d)
    breakout = setup_states["breakout"]
    pullback = setup_states["pullback"]
    unicorn = setup_states["unicorn"]
    reversal = setup_states["reversal"]

    # primary setup order aligns with the user preference: Unicorn first.
    primary_key = None
    primary_state = None
    for key in ("unicorn", "breakout", "pullback", "reversal"):
        if bool(setup_states[key].get("valid", False)):
            primary_key = key
            primary_state = setup_states[key]
            break

    if primary_state is None:
        plan = {
            "entry_zone_low": np.nan,
            "entry_zone_high": np.nan,
            "entry_price_plan": np.nan,
            "entry_trigger": "No_signal",
            "stop_loss_plan": np.nan,
            "target_1": np.nan,
            "target_2": np.nan,
            "risk_per_share": np.nan,
            "risk_reward_1": np.nan,
            "risk_reward_2": np.nan,
            "upside_to_t1_pct": np.nan,
            "upside_to_t2_pct": np.nan,
            "plan_reason": "No setup valid",
            "setup_kind": "None",
            "setup_variant": "None",
            "breakout_confirmed": False,
            "breakout_reference": breakout.get("breakout_reference", np.nan),
            "breakout_invalidation_level": breakout.get("invalidation_level", np.nan),
            "pullback_valid": False,
            "pullback_reference": pullback.get("pullback_reference", np.nan),
            "pullback_invalidation_level": pullback.get("invalidation_level", np.nan),
            "entry_valid": False,
            "entry_mode": "None",
            "tradeability_ok": False,
            "tradeability_score": np.nan,
            "tradeability_reason": "no_setup",
            "tradeability_tier": "n/a",
            "tradeability_gate_reason": "no_setup",
            "setup_lifecycle_stage": "NO_SETUP",
            "setup_validity_ok": False,
            "setup_validity_reason": "No setup valid",
            "setup_next_action": "WAIT",
            "setup_age_bars": np.nan,
            "setup_age_limit": np.nan,
            "setup_distance_to_entry_pct": np.nan,
            "setup_rr_1": np.nan,
            "setup_rr_2": np.nan,
            "spread_proxy_20d": np.nan,
            "gap_proxy_20d": np.nan,
            "avg_value_traded_20d": np.nan,
            "tradeability_threshold": np.nan,
            "tradeability_value_floor": np.nan,
            "tradeability_spread_cap": np.nan,
            "tradeability_gap_cap": np.nan,
            "execution_status": "WAIT",
            "execution_status_reason": "No setup valid",
            "entry_candidate_label": "NONE",
            "candidate_entry_price": np.nan,
            "candidate_stop_price": np.nan,
            "candidate_target_1": np.nan,
            "candidate_target_2": np.nan,
            "candidate_entry_zone_low": np.nan,
            "candidate_entry_zone_high": np.nan,
            "candidate_risk_reward_1": np.nan,
            "candidate_risk_reward_2": np.nan,
            "setup_states": setup_states,
        }
        return plan

    kind = str(primary_state.get("setup_kind", "None"))
    variant = str(primary_state.get("setup_variant", "None"))
    profile = _setup_entry_profile(kind, entry_buffer_atr, stop_loss_atr, target_1_atr, target_2_atr)

    entry_zone_low = _safe_float(primary_state.get("entry_zone_low"), np.nan)
    entry_zone_high = _safe_float(primary_state.get("entry_zone_high"), np.nan)
    entry_price = _safe_float(primary_state.get("entry_price"), np.nan)
    stop_price = _safe_float(primary_state.get("stop_price"), np.nan)
    target_1 = _safe_float(primary_state.get("target_1"), np.nan)
    target_2 = _safe_float(primary_state.get("target_2"), np.nan)
    support_anchor = _safe_float(primary_state.get("support_anchor"), np.nan)
    resistance_anchor = _safe_float(primary_state.get("resistance_anchor"), np.nan)

    projected_zone = _project_entry_zone_for_setup(
        kind,
        close=close,
        atr_v=atr_v,
        support_anchor=support_anchor,
        resistance_anchor=resistance_anchor,
        breakout_reference=_safe_float(breakout.get("breakout_reference"), np.nan),
        fvg_bottom=_safe_float(unicorn.get("fvg_bottom"), np.nan),
        fvg_top=_safe_float(unicorn.get("fvg_top"), np.nan),
        breaker_bottom=_safe_float(unicorn.get("breaker_bottom"), np.nan),
        sweep_low=_safe_float(unicorn.get("sweep_low"), np.nan),
    )

    if (
        not np.isfinite(entry_zone_low)
        or not np.isfinite(entry_zone_high)
        or not np.isfinite(entry_price)
        or entry_price >= close
        or entry_zone_high >= close
    ):
        # fallback to a projected forward retrace zone under the current close
        entry_zone_low = _safe_float(projected_zone.get("entry_zone_low"), np.nan)
        entry_zone_high = _safe_float(projected_zone.get("entry_zone_high"), np.nan)
        entry_price = _safe_float(projected_zone.get("entry_price"), np.nan)

    if not np.isfinite(entry_zone_low) or not np.isfinite(entry_zone_high):
        # secondary fallback around the structural anchor
        anchor = _min_finite([support_anchor, ema20, ema50], default=close - atr_v * 0.6)
        entry_zone_low = max(0.0, anchor - profile["entry_buffer_atr"] * atr_v)
        entry_zone_high = max(entry_zone_low, min(anchor + atr_v * 0.2, close - max(atr_v * 0.05, close * 0.001)))
        entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=profile["entry_bias"])

    if np.isfinite(entry_price) and entry_price >= close:
        entry_price = min(close - max(atr_v * 0.08, close * 0.002), entry_zone_high if np.isfinite(entry_zone_high) else close - atr_v * 0.08)
        entry_price = max(entry_price, entry_zone_low)

    if not np.isfinite(stop_price) or stop_price >= entry_price:
        stop_price = max(entry_price - atr_v * profile["stop_loss_atr"], 0.0)

    if not np.isfinite(target_1):
        target_1 = max(entry_price + atr_v * profile["target_1_atr"], entry_price * 1.02)
    if not np.isfinite(target_2):
        target_2 = max(target_1 + atr_v * 0.9, entry_price + atr_v * profile["target_2_atr"])

    risk = max(entry_price - stop_price, 1e-9)
    rr1 = (target_1 - entry_price) / risk
    rr2 = (target_2 - entry_price) / risk

    if not np.isfinite(resistance_anchor):
        resistance_anchor = _max_finite([breakout.get("resistance_anchor"), pullback.get("resistance_anchor"), reversal.get("resistance_anchor"), close], default=close)

    entry_valid = bool(primary_state.get("valid", False))
    entry_mode = "Limit_Retest" if entry_valid else "Watch"
    entry_trigger = str(primary_state.get("entry_trigger", "n/a"))
    plan_reason = f"{kind} {variant} valid" if entry_valid else "No setup valid"
    setup_age_bars = _safe_float(primary_state.get("age_bars"), np.nan)
    setup_age_limit = float(max(profile["stop_loss_atr"], profile["target_1_atr"]) * 10)

    setup_rr_1 = float(rr1)
    setup_rr_2 = float(rr2)
    distance_to_entry_pct = float((close / max(entry_price, 1e-9) - 1.0) * 100.0)
    setup_distance_to_entry_atr = float((close - entry_price) / max(atr_v, 1e-9))

    proj = _projected_entry_flow(
        kind,
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        support_anchor=support_anchor,
        resistance_anchor=resistance_anchor,
        breakout_reference=_safe_float(breakout.get("breakout_reference"), np.nan),
        fvg_bottom=_safe_float(unicorn.get("fvg_bottom"), np.nan),
        fvg_top=_safe_float(unicorn.get("fvg_top"), np.nan),
        breaker_bottom=_safe_float(unicorn.get("breaker_bottom"), np.nan),
        sweep_low=_safe_float(unicorn.get("sweep_low"), np.nan),
    )

    lifecycle = _build_setup_lifecycle_snapshot(
        stock_res,
        {"setup_kind": kind, "entry_valid": entry_valid, "plan_reason": plan_reason},
        plan_reason=plan_reason,
    )

    return {
        "entry_zone_low": entry_zone_low,
        "entry_zone_high": entry_zone_high,
        "entry_price_plan": entry_price,
        "entry_trigger": entry_trigger,
        "stop_loss_plan": stop_price,
        "target_1": target_1,
        "target_2": target_2,
        "risk_per_share": risk,
        "risk_reward_1": rr1,
        "risk_reward_2": rr2,
        "upside_to_t1_pct": (target_1 / max(entry_price, 1e-9) - 1.0) * 100.0,
        "upside_to_t2_pct": (target_2 / max(entry_price, 1e-9) - 1.0) * 100.0,
        "plan_reason": plan_reason,
        "setup_kind": kind,
        "setup_variant": variant,
        "breakout_confirmed": bool(breakout.get("valid", False)),
        "breakout_reference": breakout.get("breakout_reference", np.nan),
        "breakout_invalidation_level": breakout.get("invalidation_level", np.nan),
        "pullback_valid": bool(pullback.get("valid", False)),
        "pullback_reference": pullback.get("pullback_reference", np.nan),
        "pullback_invalidation_level": pullback.get("invalidation_level", np.nan),
        "entry_valid": entry_valid,
        "entry_mode": entry_mode,
        "tradeability_ok": bool(stock_res.get("tradeability_gate_ok", True)),
        "tradeability_score": _safe_float(stock_res.get("tradeability_score"), np.nan),
        "tradeability_reason": str(stock_res.get("tradeability_reason", "n/a")),
        "tradeability_tier": str(stock_res.get("tradeability_tier", "n/a")),
        "tradeability_gate_reason": str(stock_res.get("tradeability_gate_reason", "n/a")),
        "setup_lifecycle_stage": lifecycle["setup_lifecycle_stage"],
        "setup_validity_ok": lifecycle["setup_validity_ok"],
        "setup_validity_reason": lifecycle["setup_validity_reason"],
        "setup_next_action": lifecycle["setup_next_action"],
        "setup_age_bars": setup_age_bars,
        "setup_age_limit": setup_age_limit,
        "setup_distance_to_entry_pct": distance_to_entry_pct,
        "setup_distance_to_entry_atr": setup_distance_to_entry_atr,
        "setup_rr_1": setup_rr_1,
        "setup_rr_2": setup_rr_2,
        "spread_proxy_20d": _safe_float(stock_res.get("spread_proxy_20d"), np.nan),
        "gap_proxy_20d": _safe_float(stock_res.get("gap_proxy_20d"), np.nan),
        "avg_value_traded_20d": _safe_float(stock_res.get("avg_value_traded_20d"), np.nan),
        "tradeability_threshold": _safe_float(stock_res.get("tradeability_threshold"), np.nan),
        "tradeability_value_floor": _safe_float(stock_res.get("tradeability_value_floor"), np.nan),
        "tradeability_spread_cap": _safe_float(stock_res.get("tradeability_spread_cap"), np.nan),
        "tradeability_gap_cap": _safe_float(stock_res.get("tradeability_gap_cap"), np.nan),
        "execution_status": "READY" if entry_valid else "WAIT",
        "execution_status_reason": plan_reason,
        "entry_candidate_label": f"{kind}_{variant}".strip("_"),
        "candidate_entry_price": entry_price,
        "candidate_stop_price": stop_price,
        "candidate_target_1": target_1,
        "candidate_target_2": target_2,
        "candidate_entry_zone_low": entry_zone_low,
        "candidate_entry_zone_high": entry_zone_high,
        "candidate_risk_reward_1": rr1,
        "candidate_risk_reward_2": rr2,
        "projected_first_leg": proj["projected_first_leg"],
        "projected_rebound_leg": proj["projected_rebound_leg"],
        "entry_zone_role": proj["entry_zone_role"],
        "entry_zone_label": proj["entry_zone_label"],
        "entry_projection_summary": proj["entry_projection_summary"],
        "projected_entry_price": proj.get("projected_entry_price", entry_price),
        "projected_entry_zone": proj.get("projected_entry_zone", f"{entry_zone_low:.2f}-{entry_zone_high:.2f}"),
        "retest_anchor": proj["retest_anchor"],
        "setup_states": setup_states,
    }


def build_entry_plan(
    stock_res: dict,
    entry_buffer_atr: float = 0.25,
    stop_loss_atr: float = 1.8,
    target_1_atr: float = 2.2,
    target_2_atr: float = 3.8,
) -> dict:
    return _build_consistent_entry_plan(
        stock_res,
        entry_buffer_atr=entry_buffer_atr,
        stop_loss_atr=stop_loss_atr,
        target_1_atr=target_1_atr,
        target_2_atr=target_2_atr,
    )


def score_stock_smc(
    df: pd.DataFrame,
    flow_used: bool,
    flow_val: float,
    min_avg_volume: float,
    min_price: float,
    max_price: float,
    mode: str,
    min_history_bars: int,
    macro_context: dict | None = None,
    future_fundamental_context: dict | None = None,
) -> dict:
    d = _ensure_technical_columns(df.copy()) if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if d.empty or len(d) < int(min_history_bars):
        return {"valid": False, "reason": "Data historis tidak mencukupi"}

    d["Pivot_High_Confirmed"], d["Pivot_Low_Confirmed"] = _pivot_series(d, left=3, right=3)
    d["Bullish_OB"] = False
    d["Bullish_FVG"] = False

    # Simple bullish OB proxy: last bearish candle before strong bullish candle.
    close = d["Close"]
    open_ = d["Open"]
    body = close - open_
    strong_bull = (body > 0) & (body > body.rolling(3).mean().fillna(body))
    bearish_before = (body.shift(1) < 0) & strong_bull
    d.loc[bearish_before.fillna(False), "Bullish_OB"] = True

    # Simple bullish FVG proxy: low of current bar above high two bars ago.
    fvg_mask = d["Low"] > d["High"].shift(2)
    d.loc[fvg_mask.fillna(False), "Bullish_FVG"] = True

    last = d.iloc[-1]
    close_v = _safe_float(last.get("Close"), np.nan)
    atr_v = _safe_float(last.get("ATR14"), np.nan)
    ema20 = _safe_float(last.get("EMA20"), np.nan)
    ema50 = _safe_float(last.get("EMA50"), np.nan)
    ema200 = _safe_float(last.get("EMA200"), np.nan)
    rsi_v = _safe_float(last.get("RSI14"), np.nan)
    macd_hist = _safe_float(last.get("MACD_HIST"), np.nan)
    adx_v = _safe_float(last.get("ADX14"), np.nan)
    rel_vol = _safe_float(last.get("REL_VOL"), np.nan)
    cmf_v = _safe_float(last.get("CMF20"), np.nan)
    mfi_v = _safe_float(last.get("MFI14"), np.nan)
    obv_slope = _safe_float(last.get("OBV_SLOPE10"), np.nan)
    roc12 = _safe_float(last.get("ROC12"), np.nan)

    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close_v * 0.02, 1.0)

    trend_score = 50.0
    market_structure_score = 50.0
    momentum_score = 50.0
    smart_money_score = 50.0
    risk_score = 50.0

    if np.isfinite(close_v) and np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(ema200):
        if close_v > ema20 > ema50 > ema200:
            trend_score = 92.0
            market_structure_score = 88.0
        elif close_v > ema20 and ema20 >= ema50:
            trend_score = 78.0
            market_structure_score = 72.0
        elif close_v < ema20 < ema50 < ema200:
            trend_score = 22.0
            market_structure_score = 20.0
        else:
            trend_score = 50.0
            market_structure_score = 50.0

    if np.isfinite(rsi_v):
        momentum_score += (rsi_v - 50.0) * 0.75
    if np.isfinite(macd_hist):
        momentum_score += np.clip(macd_hist / max(atr_v, 1e-9) * 8.0, -10.0, 10.0)
    if np.isfinite(roc12):
        momentum_score += np.clip(roc12 * 0.15, -8.0, 12.0)

    if np.isfinite(cmf_v):
        smart_money_score += cmf_v * 100.0 * 0.35
    if np.isfinite(mfi_v):
        smart_money_score += (mfi_v - 50.0) * 0.18
    if np.isfinite(obv_slope):
        smart_money_score += np.clip(obv_slope / (abs(obv_slope) + 1e-9) * 4.0, -4.0, 4.0)
    if bool(last.get("Bullish_OB", False)):
        smart_money_score += 6.0
    if bool(last.get("Bullish_FVG", False)):
        smart_money_score += 4.0

    if np.isfinite(rel_vol):
        risk_score += np.clip((rel_vol - 1.0) * 8.0, -12.0, 10.0)
    if np.isfinite(adx_v):
        risk_score += np.clip((adx_v - 20.0) * 0.5, -10.0, 10.0)

    macro_context = macro_context or {}
    macro_score = _safe_float(macro_context.get("macro_score"), 50.0)
    macro_gate_ok = bool(macro_context.get("macro_gate_ok", True))
    macro_multiplier = _safe_float(macro_context.get("macro_multiplier"), 1.0)

    # Tradeability gate: price + liquidity.
    avg_value_traded_20d = float((d["Close"] * d["Volume"]).rolling(20).mean().iloc[-1]) if len(d) >= 20 else np.nan
    tradeable_price = bool(np.isfinite(close_v) and close_v >= float(min_price) and close_v <= float(max_price))
    tradeable_liquidity = bool(np.isfinite(avg_value_traded_20d) and avg_value_traded_20d >= float(min_avg_volume))
    tradeability_gate_ok = bool(tradeable_price and tradeable_liquidity)

    tradeability_score = 50.0
    if tradeable_price:
        tradeability_score += 15.0
    if tradeable_liquidity:
        tradeability_score += 20.0
    if np.isfinite(rel_vol):
        tradeability_score += np.clip((rel_vol - 1.0) * 5.0, -5.0, 8.0)
    tradeability_score = float(np.clip(tradeability_score, 0.0, 100.0))

    tradeability_reason = "OK" if tradeability_gate_ok else "Price/liquidity filter not met"
    tradeability_tier = "A" if tradeability_score >= 80 else ("B" if tradeability_score >= 65 else ("C" if tradeability_score >= 50 else "D"))

    cycle_tuple = compute_cycle_features(d["Close"])
    dominant_period, time_to_bottom, cycle_gate_ok, cycle_meta = cycle_tuple
    cycle_reliability = _safe_float(cycle_meta.get("cycle_reliability"), 0.0)
    phase, phase_conf = _phase_from_trend(d)

    # Setup states and unified entry plan.
    stock_res = {
        "df": d,
        "last": last,
        "tradeability_gate_ok": tradeability_gate_ok,
        "tradeability_score": tradeability_score,
        "tradeability_reason": tradeability_reason,
        "tradeability_tier": tradeability_tier,
        "tradeability_gate_reason": tradeability_reason,
        "tradeability_threshold": 65.0,
        "tradeability_value_floor": float(min_avg_volume),
        "tradeability_spread_cap": np.nan,
        "tradeability_gap_cap": np.nan,
        "spread_proxy_20d": np.nan,
        "gap_proxy_20d": np.nan,
        "avg_value_traded_20d": avg_value_traded_20d,
        "cycle_tuple": cycle_tuple,
    }
    entry_plan = build_entry_plan(stock_res)

    any_setup_valid = any(bool(entry_plan["setup_states"][k].get("valid", False)) for k in ("unicorn", "breakout", "pullback", "reversal"))
    primary_kind = str(entry_plan.get("setup_kind", "None"))

    # Composite score is ranking only, never hard gate.
    score_raw = (
        0.26 * trend_score
        + 0.18 * momentum_score
        + 0.18 * market_structure_score
        + 0.14 * smart_money_score
        + 0.10 * risk_score
        + 0.10 * macro_score
        + 0.04 * tradeability_score
    )
    if flow_used:
        score_raw += (float(flow_val) - 50.0) * 0.12
    score_raw *= float(np.clip(macro_multiplier, 0.85, 1.15))
    score = float(np.clip(score_raw, 0.0, 100.0))

    mode_u = str(mode or "Balanced").strip().upper()
    thresholds = {
        "CONSERVATIVE": 78.0,
        "BALANCED": 68.0,
        "AGGRESSIVE": 58.0,
    }
    buy_threshold = thresholds.get(mode_u, 68.0)
    strong_threshold = buy_threshold + 10.0

    decision = "AVOID"
    if any_setup_valid and score >= buy_threshold and tradeability_gate_ok and macro_gate_ok:
        decision = "BUY"
    if any_setup_valid and score >= strong_threshold and tradeability_gate_ok and macro_gate_ok:
        decision = "STRONG BUY"

    # Setup-derived validity / lifecycle.
    setup_validity_ok = bool(any_setup_valid)
    if setup_validity_ok:
        if primary_kind.upper() == "UNICORN":
            setup_lifecycle_stage = "ENTRY_ZONE" if entry_plan.get("entry_valid", False) else "WATCHLIST"
        else:
            setup_lifecycle_stage = "ENTRY_ZONE" if entry_plan.get("entry_valid", False) else "WATCHLIST"
        setup_next_action = "MONITOR_ENTRY" if entry_plan.get("entry_valid", False) else "WAIT"
        setup_validity_reason = f"Primary {primary_kind} setup valid" if primary_kind != "None" else "Setup valid"
    else:
        setup_lifecycle_stage = "NO_SETUP"
        setup_next_action = "WAIT"
        setup_validity_reason = "No valid setup"
    setup_age_bars = entry_plan.get("setup_age_bars", np.nan)

    # Fill result structure
    res = {
        "valid": True,
        "symbol": "",
        "df": d,
        "last": last,
        "score_raw": float(score_raw),
        "score": float(score),
        "decision": decision,
        "Decision": decision,
        "decision_reason": f"{primary_kind} | tradeability={tradeability_reason}",
        "DecisionRaw": decision,
        "buy_threshold": buy_threshold,
        "strong_threshold": strong_threshold,
        "tradeability_gate_ok": tradeability_gate_ok,
        "tradeability_score": tradeability_score,
        "tradeability_reason": tradeability_reason,
        "tradeability_tier": tradeability_tier,
        "tradeability_gate_reason": tradeability_reason,
        "tradeability_threshold": 65.0,
        "tradeability_value_floor": float(min_avg_volume),
        "tradeability_spread_cap": np.nan,
        "tradeability_gap_cap": np.nan,
        "avg_value_traded_20d": avg_value_traded_20d,
        "flow_used": bool(flow_used),
        "flow_val": float(flow_val),
        "market_structure_score": float(np.clip(market_structure_score, 0.0, 100.0)),
        "trend_score": float(np.clip(trend_score, 0.0, 100.0)),
        "momentum_score": float(np.clip(momentum_score, 0.0, 100.0)),
        "risk_score": float(np.clip(risk_score, 0.0, 100.0)),
        "smart_money_score": float(np.clip(smart_money_score, 0.0, 100.0)),
        "phase": phase,
        "phase_confidence": phase_conf,
        "dominant_period": dominant_period,
        "cycle_reliability": cycle_reliability,
        "cycle_gate_reason": cycle_meta.get("cycle_gate_reason", "n/a"),
        "time_to_next_bottom": time_to_bottom,
        "macro_phase": macro_context.get("macro_phase", "Unknown"),
        "macro_phase_confidence": _safe_float(macro_context.get("macro_phase_confidence"), 0.0),
        "macro_period": macro_context.get("macro_period", np.nan),
        "macro_time_to_bottom": macro_context.get("macro_time_to_bottom", np.nan),
        "macro_time_to_top": macro_context.get("macro_time_to_top", np.nan),
        "macro_phase_age_bars": macro_context.get("macro_phase_age_bars", np.nan),
        "macro_phase_age_pct": macro_context.get("macro_phase_age_pct", np.nan),
        "macro_cycle_reliability": _safe_float(macro_context.get("macro_cycle_reliability"), 0.0),
        "macro_cycle_gate_reason": macro_context.get("macro_cycle_gate_reason", "n/a"),
        "macro_score": float(np.clip(macro_score, 0.0, 100.0)),
        "macro_gate_ok": bool(macro_gate_ok),
        "macro_gate_reason": macro_context.get("macro_gate_reason", "OK"),
        "macro_multiplier": float(macro_multiplier),
        "market_regime": macro_context.get("market_regime", "SIDEWAYS"),
        "market_regime_confidence": _safe_float(macro_context.get("market_regime_confidence"), 0.5),
        "market_regime_reason": macro_context.get("market_regime_reason", "n/a"),
        "setup_validity_ok": setup_validity_ok,
        "setup_validity_reason": setup_validity_reason,
        "setup_lifecycle_stage": setup_lifecycle_stage,
        "setup_next_action": setup_next_action,
        "setup_age_bars": setup_age_bars,
        "setup_kind": entry_plan.get("setup_kind", "None"),
        "setup_variant": entry_plan.get("setup_variant", "None"),
        "entry_plan": entry_plan,
    }

    # Unpack the unified plan and expose raw setup states.
    res.update(entry_plan)
    states = entry_plan.get("setup_states", {})
    breakout = states.get("breakout", {})
    pullback = states.get("pullback", {})
    unicorn = states.get("unicorn", {})
    reversal = states.get("reversal", {})

    res.update({
        "breakout_setup_valid": bool(breakout.get("valid", False)),
        "breakout_confirmed": bool(breakout.get("confirmed", False)),
        "breakout_setup_status": breakout.get("status", "-"),
        "pullback_continuation_valid": bool(pullback.get("valid", False)),
        "pullback_continuation_confirmed": bool(pullback.get("confirmed", False)),
        "pullback_continuation_status": pullback.get("status", "-"),
        "reversal_accumulation_valid": bool(reversal.get("valid", False)),
        "reversal_accumulation_confirmed": bool(reversal.get("confirmed", False)),
        "reversal_accumulation_status": reversal.get("status", "-"),
        "unicorn_setup_valid": bool(unicorn.get("valid", False)),
        "unicorn_setup_confirmed": bool(unicorn.get("confirmed", False)),
        "unicorn_setup_status": unicorn.get("status", "-"),
        "unicorn_setup_state": unicorn.get("status", "-"),
        "unicorn_setup_fresh": bool(unicorn.get("status", "") == "fresh"),
        "unicorn_sniper_valid": bool(unicorn.get("sniper_valid", False)),
        "unicorn_sniper_confirmed": bool(unicorn.get("sniper_valid", False)),
        "unicorn_sniper_status": unicorn.get("sniper_status", "-"),
        "unicorn_sniper_state": unicorn.get("sniper_status", "-"),
        "fvg_present": bool(unicorn.get("valid", False)),
        "fvg_age_bars": unicorn.get("age_bars", np.nan),
        "fvg_status": unicorn.get("status", "-"),
        "fvg_top": unicorn.get("fvg_top", np.nan),
        "fvg_bottom": unicorn.get("fvg_bottom", np.nan),
        "ob_present": bool(bool(last.get("Bullish_OB", False))),
        "trend_ok": bool(trend_score >= 60.0),
        "reversal_hits": int(bool(reversal.get("valid", False))),
        "breakout_reference": breakout.get("breakout_reference", np.nan),
        "breakout_invalidation_level": breakout.get("invalidation_level", np.nan),
        "pullback_continuation_reference": pullback.get("pullback_reference", np.nan),
        "pullback_continuation_invalidation": pullback.get("invalidation_level", np.nan),
        "reversal_accumulation_reference": reversal.get("resistance_anchor", np.nan),
        "reversal_accumulation_invalidation": reversal.get("invalidation_level", np.nan),
        "unicorn_sweep_low": unicorn.get("sweep_low", np.nan),
        "unicorn_breaker_bottom": unicorn.get("breaker_bottom", np.nan),
        "unicorn_breaker_top": unicorn.get("breaker_top", np.nan),
        "unicorn_fvg_bottom": unicorn.get("fvg_bottom", np.nan),
        "unicorn_fvg_top": unicorn.get("fvg_top", np.nan),
        "setup_distance_to_entry_atr": _safe_float(entry_plan.get("setup_distance_to_entry_atr"), np.nan),
    })

    # Compatibility aliases used by the app.
    res["candidate_entry_price"] = entry_plan.get("candidate_entry_price", np.nan)
    res["candidate_stop_price"] = entry_plan.get("candidate_stop_price", np.nan)
    res["candidate_target_1"] = entry_plan.get("candidate_target_1", np.nan)
    res["candidate_target_2"] = entry_plan.get("candidate_target_2", np.nan)
    res["candidate_entry_zone_low"] = entry_plan.get("candidate_entry_zone_low", np.nan)
    res["candidate_entry_zone_high"] = entry_plan.get("candidate_entry_zone_high", np.nan)
    res["candidate_risk_reward_1"] = entry_plan.get("candidate_risk_reward_1", np.nan)
    res["candidate_risk_reward_2"] = entry_plan.get("candidate_risk_reward_2", np.nan)

    return res
