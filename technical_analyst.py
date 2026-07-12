import concurrent.futures as cf
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema, hilbert, periodogram

from data_engine import load_ticker_data, normalize_ticker


def _status_has_entry(status) -> bool:
    text = str(status or "").upper().strip()
    return "ENTRY" in text


def _min_finite(values, default=np.nan):
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(min(finite)) if finite else default


def _max_finite(values, default=np.nan):
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(max(finite)) if finite else default



def _setup_entry_profile(
    setup_kind: str,
    entry_buffer_atr: float,
    stop_loss_atr: float,
    target_1_atr: float,
    target_2_atr: float,
) -> dict:
    """Return setup-specific entry/exit parameters.

    The profiles keep the same API but widen the entry zone slightly and avoid
    over-tight stops that can make IDX setups disappear too often.
    """
    kind = str(setup_kind or "").strip().upper()
    profiles = {
        "BREAKOUT": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.72),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.50),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.00),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.10),
            "entry_bias": 0.24,
            "late_entry_bias": 0.14,
            "chase_limit_atr": 0.24,
            "rr_floor_1": 1.60,
            "rr_floor_2": 2.35,
            "max_risk_pct": 0.075,
            "pullback_floor_atr": 0.18,
        },
        "SNIPER": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.82),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.74),
            "target_1_atr": max(0.0, float(target_1_atr) * 0.98),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.00),
            "entry_bias": 0.28,
            "late_entry_bias": 0.18,
            "chase_limit_atr": 0.14,
            "rr_floor_1": 1.95,
            "rr_floor_2": 2.75,
            "max_risk_pct": 0.080,
            "pullback_floor_atr": 0.05,
        },
        "UNICORN": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.88),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.80),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.00),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.05),
            "entry_bias": 0.30,
            "late_entry_bias": 0.20,
            "chase_limit_atr": 0.18,
            "rr_floor_1": 1.70,
            "rr_floor_2": 2.60,
            "max_risk_pct": 0.082,
            "pullback_floor_atr": 0.06,
        },
        "PULLBACK": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.76),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.90),
            "target_1_atr": max(0.0, float(target_1_atr) * 0.96),
            "target_2_atr": max(0.0, float(target_2_atr) * 0.99),
            "entry_bias": 0.36,
            "late_entry_bias": 0.24,
            "chase_limit_atr": 0.16,
            "rr_floor_1": 1.50,
            "rr_floor_2": 2.25,
            "max_risk_pct": 0.078,
            "pullback_floor_atr": 0.04,
        },
        "REVERSAL": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.78),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.92),
            "target_1_atr": max(0.0, float(target_1_atr) * 0.96),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.00),
            "entry_bias": 0.40,
            "late_entry_bias": 0.26,
            "chase_limit_atr": 0.16,
            "rr_floor_1": 1.55,
            "rr_floor_2": 2.45,
            "max_risk_pct": 0.085,
            "pullback_floor_atr": 0.03,
        },
    }
    return profiles.get(kind, profiles["UNICORN"]).copy()

def _projected_entry_flow(
    setup_kind: str,
    *,
    entry_zone_low: float = np.nan,
    entry_zone_high: float = np.nan,
    breakout_reference: float = np.nan,
    support_anchor: float = np.nan,
    resistance_anchor: float = np.nan,
    fvg_bottom: float = np.nan,
    fvg_top: float = np.nan,
    sweep_low: float = np.nan,
    breaker_bottom: float = np.nan,
) -> dict:
    """Describe the expected first leg and rebound zone for a long setup."""
    kind = str(setup_kind or "").strip().upper()

    def _fmt(v: float) -> str:
        return f"Rp {float(v):,.0f}" if np.isfinite(v) else "n/a"

    zone_low = _fmt(entry_zone_low)
    zone_high = _fmt(entry_zone_high)

    if kind == "BREAKOUT":
        return {
            "projected_first_leg": "Pull back from breakout extension into former resistance",
            "projected_rebound_leg": "Rebound upward after the retest confirms support",
            "entry_zone_role": "Former resistance / breakout retest",
            "entry_zone_label": f"{zone_low} - {zone_high}",
            "entry_projection_summary": f"Breakout retest: wait for pullback to the breakout zone ({zone_low} - {zone_high}) and buy the rebound.",
            "retest_anchor": breakout_reference,
        }

    if kind in {"UNICORN", "SNIPER"}:
        return {
            "projected_first_leg": "Retrace back into the displaced FVG / breaker zone",
            "projected_rebound_leg": "Rebound upward after FVG mitigation and reclaim",
            "entry_zone_role": "FVG / breaker mitigation zone",
            "entry_zone_label": f"{zone_low} - {zone_high}",
            "entry_projection_summary": f"Unicorn retest: price typically pulls back into the FVG / breaker zone ({zone_low} - {zone_high}) before continuation.",
            "retest_anchor": _min_finite([fvg_bottom, fvg_top, breaker_bottom, sweep_low], default=np.nan),
        }

    if kind == "PULLBACK":
        return {
            "projected_first_leg": "Pull back into trend support / EMA confluence",
            "projected_rebound_leg": "Rebound upward with the trend after support holds",
            "entry_zone_role": "Trend support / retest zone",
            "entry_zone_label": f"{zone_low} - {zone_high}",
            "entry_projection_summary": f"Pullback continuation: buy the retest of support ({zone_low} - {zone_high}) rather than the breakout candle.",
            "retest_anchor": support_anchor,
        }

    if kind == "REVERSAL":
        return {
            "projected_first_leg": "Revisit the reclaimed base / accumulation support",
            "projected_rebound_leg": "Rebound upward from the reclaimed base",
            "entry_zone_role": "Accumulation base / reclaim zone",
            "entry_zone_label": f"{zone_low} - {zone_high}",
            "entry_projection_summary": f"Reversal accumulation: price usually revisits the reclaimed base ({zone_low} - {zone_high}) before the rebound leg.",
            "retest_anchor": support_anchor,
        }

    return {
        "projected_first_leg": "Retrace into the planned entry zone",
        "projected_rebound_leg": "Rebound/continuation upward after the zone holds",
        "entry_zone_role": "Structural entry zone",
        "entry_zone_label": f"{zone_low} - {zone_high}",
        "entry_projection_summary": f"Entry zone projected at {zone_low} - {zone_high}.",
        "retest_anchor": np.nan,
    }


def _setup_distance_to_entry_atr(close: float, entry_price: float, atr_v: float) -> float:
    """Return the absolute distance from close to the planned entry in ATR units."""
    if not np.isfinite(close) or not np.isfinite(entry_price) or not np.isfinite(atr_v) or atr_v <= 0:
        return np.nan
    return float(abs(close - entry_price) / max(atr_v, 1e-9))


def _setup_structure_confluence_score(
    entry_price: float,
    atr_v: float,
    *,
    entry_zone_low: float = np.nan,
    entry_zone_high: float = np.nan,
    recent_swing_low: float = np.nan,
    recent_swing_high: float = np.nan,
    sweep_low: float = np.nan,
    breaker_bottom: float = np.nan,
    fvg_bottom: float = np.nan,
    fvg_top: float = np.nan,
    support_anchor: float = np.nan,
    resistance_anchor: float = np.nan,
) -> float:
    """Estimate how much real structure is clustered around the entry."""
    if not np.isfinite(entry_price) or not np.isfinite(atr_v) or atr_v <= 0:
        return np.nan

    levels = [
        recent_swing_low,
        recent_swing_high,
        sweep_low,
        breaker_bottom,
        fvg_bottom,
        fvg_top,
        support_anchor,
        resistance_anchor,
    ]

    score = 0.0
    above = False
    below = False
    for raw in levels:
        fv = _safe_float(raw, np.nan)
        if not np.isfinite(fv):
            continue
        dist_atr = abs(float(fv) - float(entry_price)) / max(float(atr_v), 1e-9)
        if fv > entry_price:
            above = True
        if fv < entry_price:
            below = True
        if dist_atr <= 0.15:
            score += 8.0
        elif dist_atr <= 0.35:
            score += 6.0
        elif dist_atr <= 0.70:
            score += 4.0
        elif dist_atr <= 1.20:
            score += 2.0

    zone_low = _safe_float(entry_zone_low, np.nan)
    zone_high = _safe_float(entry_zone_high, np.nan)
    if np.isfinite(zone_low) and np.isfinite(zone_high):
        zone_width_atr = abs(zone_high - zone_low) / max(float(atr_v), 1e-9)
        if 0.35 <= zone_width_atr <= 1.50:
            score += 8.0
        elif 1.50 < zone_width_atr <= 2.20:
            score += 5.0
        elif zone_width_atr < 0.25:
            score -= 4.0
        else:
            score -= 2.0

    if above and below:
        score += 4.0

    return float(np.clip(score, 0.0, 30.0))


def _nearest_level_above(levels, ref: float) -> float:
    vals = []
    for v in levels:
        fv = _safe_float(v, np.nan)
        if np.isfinite(fv) and np.isfinite(ref) and fv > ref:
            vals.append(float(fv))
    return float(min(vals)) if vals else np.nan


def _liquidity_target_pair_long(
    pivot_highs,
    entry_price: float,
    fallback_anchor_high: float,
    atr_v: float,
    t1_atr_pad: float = 0.80,
    t2_atr_pad: float = 1.60,
) -> tuple[float, float]:
    """Pick structural upside targets from liquidity above the entry.

    The first target prefers the nearest confirmed swing high above entry.
    The second target prefers the next liquidity pool above target 1.
    ATR is used only as a fallback padding, not as the main definition.
    """
    highs = []
    for v in pivot_highs:
        fv = _safe_float(v, np.nan)
        if np.isfinite(fv):
            highs.append(float(fv))
    highs = sorted(set(highs))
    t1 = _nearest_level_above(highs, entry_price)
    if not np.isfinite(t1):
        t1 = max(_safe_float(fallback_anchor_high, np.nan), entry_price + atr_v * t1_atr_pad)
    t2_candidates = [v for v in highs if np.isfinite(v) and v > t1 + max(atr_v * 0.05, 1e-9)]
    t2 = float(min(t2_candidates)) if t2_candidates else np.nan
    if not np.isfinite(t2):
        t2 = max(t1 + atr_v * t2_atr_pad, entry_price + atr_v * max(t2_atr_pad, t1_atr_pad + 0.8))
    if t2 <= t1:
        t2 = max(t1 + atr_v * max(0.9, t2_atr_pad * 0.75), entry_price + atr_v * max(t2_atr_pad, 2.0))
    return float(t1), float(t2)



def _setup_take_profit_pair_long(
    setup_kind: str,
    pivot_highs,
    entry_price: float,
    atr_v: float,
    *,
    recent_swing_high: float = np.nan,
    recent_swing_low: float = np.nan,
    breakout_reference: float = np.nan,
    breakout_range_height: float = np.nan,
    support_anchor: float = np.nan,
    resistance_anchor: float = np.nan,
    sweep_low: float = np.nan,
    breaker_bottom: float = np.nan,
    fvg_bottom: float = np.nan,
    fvg_top: float = np.nan,
) -> tuple[float, float]:
    """Derive setup-specific long take-profit targets.

    Each setup keeps its own target logic:
    - Breakout: measured move / range expansion above the broken range.
    - Pullback: continuation into prior swing-high liquidity.
    - Unicorn/Sniper: nearest liquidity pool above the sweep, then the next pool.
    - Reversal: reclaim of overhead resistance first, then extension into the next pool.

    Pivot highs are preferred whenever available; ATR is only a fallback.
    """
    kind = str(setup_kind or "").strip().upper()

    highs = []
    for v in pivot_highs:
        fv = _safe_float(v, np.nan)
        if np.isfinite(fv):
            highs.append(float(fv))
    highs = sorted(set(highs))

    atr_v = float(atr_v) if np.isfinite(atr_v) and atr_v > 0 else np.nan
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(_safe_float(entry_price, 1.0) * 0.02, 1.0)

    entry_price = _safe_float(entry_price, np.nan)
    recent_swing_high = _safe_float(recent_swing_high, np.nan)
    recent_swing_low = _safe_float(recent_swing_low, np.nan)
    breakout_reference = _safe_float(breakout_reference, np.nan)
    breakout_range_height = _safe_float(breakout_range_height, np.nan)
    support_anchor = _safe_float(support_anchor, np.nan)
    resistance_anchor = _safe_float(resistance_anchor, np.nan)
    sweep_low = _safe_float(sweep_low, np.nan)
    fvg_top = _safe_float(fvg_top, np.nan)

    if not np.isfinite(entry_price):
        entry_price = 0.0

    # Helpful structural fallback ranges.
    swing_range = np.nan
    if np.isfinite(recent_swing_high) and np.isfinite(recent_swing_low):
        swing_range = max(recent_swing_high - recent_swing_low, atr_v * 0.8)

    measured_range = breakout_range_height
    if not np.isfinite(measured_range) or measured_range <= 0:
        measured_range = swing_range if np.isfinite(swing_range) else atr_v * 1.6

    pivot_t1 = _nearest_level_above(highs, entry_price)
    pivot_t2 = np.nan
    if np.isfinite(pivot_t1):
        pivot_t2_candidates = [v for v in highs if np.isfinite(v) and v > pivot_t1 + max(atr_v * 0.05, 1e-9)]
        pivot_t2 = float(min(pivot_t2_candidates)) if pivot_t2_candidates else np.nan

    if kind == "BREAKOUT":
        # Measured-move logic from the broken range is the primary TP concept.
        base = breakout_reference if np.isfinite(breakout_reference) else recent_swing_high
        if not np.isfinite(base):
            base = entry_price

        measured_t1 = max(base + measured_range * 1.0, entry_price + atr_v * 0.9)
        measured_t2 = max(base + measured_range * 1.618, entry_price + atr_v * 1.8)

        t1 = pivot_t1 if np.isfinite(pivot_t1) else measured_t1
        t2 = pivot_t2 if np.isfinite(pivot_t2) else np.nan

        if not np.isfinite(t2):
            t2 = measured_t2
        if t1 <= entry_price:
            t1 = measured_t1
        if t2 <= t1:
            t2 = max(t1 + max(measured_range * 0.55, atr_v * 1.0), measured_t2)

    elif kind == "PULLBACK":
        # Pullback continuation aims at the prior impulse high and then extension.
        base = resistance_anchor if np.isfinite(resistance_anchor) else recent_swing_high
        if not np.isfinite(base):
            base = entry_price + measured_range * 0.85

        extension = max(measured_range * 0.85, atr_v * 1.2)
        t1 = pivot_t1 if np.isfinite(pivot_t1) else base
        t2 = pivot_t2 if np.isfinite(pivot_t2) else np.nan

        if not np.isfinite(t1) or t1 <= entry_price:
            t1 = max(base, entry_price + extension * 0.65)
        if not np.isfinite(t2) or t2 <= t1:
            t2 = max(t1 + extension * 0.90, base + extension * 1.20)

    elif kind in {"UNICORN", "SNIPER"}:

        # Unicorn targets should follow the actual swept structure:
        # 1) reclaim above the breaker/FVG and nearest overhead liquidity,
        # 2) then the next liquidity pool above that.
        #
        # The prior version tended to collapse many names into similar RR bands.
        # This version keeps the targets anchored to structure, while letting
        # each ticker breathe based on its own swing span and reclaim gap.
        structure_components = []

        for v in [recent_swing_high, recent_swing_low, sweep_low, breaker_bottom, fvg_bottom, fvg_top]:
            if np.isfinite(v):
                structure_components.append(float(v))

        structure_span_candidates = []
        if np.isfinite(recent_swing_high) and np.isfinite(recent_swing_low):
            structure_span_candidates.append(float(max(recent_swing_high - recent_swing_low, atr_v * 0.80)))
        if np.isfinite(fvg_top) and np.isfinite(fvg_bottom):
            structure_span_candidates.append(float(max(fvg_top - fvg_bottom, atr_v * 0.60)))
        if np.isfinite(fvg_top) and np.isfinite(sweep_low):
            structure_span_candidates.append(float(max(fvg_top - sweep_low, atr_v * 0.90)))
        if np.isfinite(breaker_bottom) and np.isfinite(sweep_low):
            structure_span_candidates.append(float(max(breaker_bottom - sweep_low, atr_v * 0.75)))
        structure_span_candidates.append(float(max(measured_range, atr_v * (0.95 if kind == "SNIPER" else 1.05))))
        structure_span = max(structure_span_candidates)

        overhead_anchor_candidates = [v for v in [breaker_bottom, fvg_top, recent_swing_high] if np.isfinite(v)]
        overhead_anchor = max(overhead_anchor_candidates) if overhead_anchor_candidates else entry_price + structure_span * 0.35

        reclaim_gap = max(overhead_anchor - entry_price, atr_v * (0.55 if kind == "SNIPER" else 0.70))
        structure_ratio = float(np.clip(structure_span / max(reclaim_gap, 1e-9), 0.75, 5.00))

        # Structure-derived RR bands; no ticker should collapse to the same numbers
        # unless its actual geometry is nearly identical.
        rr1_target = (
            (1.68 if kind == "SNIPER" else 1.82)
            + 0.18 * max(structure_ratio - 1.00, 0.0)
            + (0.08 if np.isfinite(fvg_top) and np.isfinite(fvg_bottom) else 0.00)
            + (0.05 if np.isfinite(breaker_bottom) and np.isfinite(sweep_low) else 0.00)
        )
        rr2_target = rr1_target + (
            (1.02 if kind == "SNIPER" else 1.18)
            + 0.28 * max(structure_ratio - 1.00, 0.0)
        )

        t1_struct = entry_price + rr1_target * reclaim_gap
        t2_struct = entry_price + rr2_target * reclaim_gap

        t1 = pivot_t1 if np.isfinite(pivot_t1) and pivot_t1 > max(entry_price, overhead_anchor - atr_v * 0.05) else np.nan
        if not np.isfinite(t1) or t1 <= entry_price:
            t1 = t1_struct

        t2 = pivot_t2 if np.isfinite(pivot_t2) and pivot_t2 > t1 else np.nan
        if not np.isfinite(t2) or t2 <= t1:
            t2 = t2_struct

        # Keep targets above the reclaimed structure, but do not flatten different stocks into the same RR.
        t1 = max(t1, overhead_anchor + atr_v * (0.12 if kind == "SNIPER" else 0.18))
        t2 = max(t2, t1 + atr_v * (0.90 if kind == "SNIPER" else 1.05))
        if t2 <= t1:
            t2 = t1 + atr_v * (1.10 if kind == "SNIPER" else 1.25)

    elif kind == "REVERSAL":
        # Reversal accumulation first targets the reclaimed resistance, then the next overhead pool.
        base = resistance_anchor if np.isfinite(resistance_anchor) else recent_swing_high
        if not np.isfinite(base):
            base = entry_price + measured_range * 0.75

        structural_height = np.nan
        if np.isfinite(base) and np.isfinite(support_anchor):
            structural_height = max(base - support_anchor, atr_v * 1.0)
        elif np.isfinite(swing_range):
            structural_height = swing_range
        else:
            structural_height = measured_range

        t1 = pivot_t1 if np.isfinite(pivot_t1) else base
        t2 = pivot_t2 if np.isfinite(pivot_t2) else np.nan

        if not np.isfinite(t1) or t1 <= entry_price:
            t1 = max(base, entry_price + structural_height * 0.60)
        if not np.isfinite(t2) or t2 <= t1:
            t2 = max(t1 + structural_height * 0.90, base + structural_height * 0.95)

    else:
        # Generic fallback: keep targets liquidity-based.
        t1 = pivot_t1 if np.isfinite(pivot_t1) else entry_price + measured_range * 0.85
        t2 = pivot_t2 if np.isfinite(pivot_t2) else max(t1 + measured_range * 0.85, entry_price + measured_range * 1.60)

    # Absolute safeguards.
    if not np.isfinite(t1) or t1 <= entry_price:
        t1 = entry_price + atr_v * 1.0
    if not np.isfinite(t2) or t2 <= t1:
        t2 = t1 + atr_v * 1.25

    return float(t1), float(t2)



def _estimate_setup_fill_probability(
    setup_kind: str,
    distance_to_entry_atr: float,
    age_bars: float | int | float = np.nan,
    setup_valid: bool = True,
    has_liquidity_sweep: bool = False,
    has_mss: bool = False,
    rr1: float = np.nan,
    rr2: float = np.nan,
    setup_fresh: bool = False,
    entry_zone_width_atr: float = np.nan,
    structure_confluence: float = np.nan,
) -> float:
    """Estimate how likely the setup is to fill.

    The goal is not to punish every setup that is slightly away from price.
    Instead, we reward closeness to structure, freshness, and confluence while
    still capping obviously late or weak setups.
    """
    kind = str(setup_kind or "").strip().upper()
    base_map = {
        "BREAKOUT": 58.0,
        "SNIPER": 68.0,
        "UNICORN": 66.0,
        "PULLBACK": 70.0,
        "REVERSAL": 62.0,
    }
    score = base_map.get(kind, 60.0)

    dist = float(distance_to_entry_atr) if np.isfinite(distance_to_entry_atr) else 4.0
    if dist >= 6.0:
        return 0.0

    if dist <= 0.35:
        score += 12.0
    elif dist <= 0.80:
        score += 8.0
    elif dist <= 1.50:
        score += 4.0
    else:
        score -= (dist - 1.50) * (5.0 if kind in {"BREAKOUT"} else 4.0)

    if kind in {"UNICORN", "SNIPER", "PULLBACK", "REVERSAL"}:
        score -= max(0.0, dist - 2.00) * 3.5
        score -= max(0.0, dist - 3.50) * 4.0
    else:
        score -= max(0.0, dist - 2.50) * 5.0

    if np.isfinite(age_bars):
        age_bars = float(age_bars)
        score -= max(0.0, age_bars - 3.0) * 1.4
        score -= max(0.0, age_bars - 8.0) * 0.9
        score -= max(0.0, age_bars - 14.0) * 0.5

    zone_w = float(entry_zone_width_atr) if np.isfinite(entry_zone_width_atr) else np.nan
    if np.isfinite(zone_w):
        if 0.30 <= zone_w <= 1.50:
            score += 8.0
        elif 1.50 < zone_w <= 2.40:
            score += 4.5
        elif zone_w < 0.25:
            score -= 3.0
        elif zone_w > 3.00:
            score -= 4.0

    if np.isfinite(structure_confluence):
        score += min(max(float(structure_confluence), 0.0), 30.0) * 0.60

    if setup_fresh:
        score += 8.0
    if has_liquidity_sweep:
        score += 7.0 if kind in {"UNICORN", "SNIPER", "PULLBACK", "REVERSAL"} else 4.0
    if has_mss:
        score += 7.0 if kind in {"UNICORN", "SNIPER", "PULLBACK", "REVERSAL"} else 4.0

    if kind in {"UNICORN", "SNIPER", "PULLBACK"} and dist <= 1.25 and (has_liquidity_sweep or has_mss):
        score += 4.0
    if kind in {"UNICORN", "SNIPER"} and dist <= 0.90 and setup_fresh:
        score += 4.0

    if np.isfinite(rr1):
        rr1 = float(rr1)
        score += np.clip((rr1 - 1.4) * 3.5, -7.0, 7.0)
    if np.isfinite(rr2):
        rr2 = float(rr2)
        score += np.clip((rr2 - 2.1) * 2.5, -6.0, 6.0)

    if not setup_valid:
        score -= 10.0

    return float(np.clip(score, 0.0, 100.0))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def zlema(series: pd.Series, span: int) -> pd.Series:
    """Zero-lag EMA approximation using a de-lagged input series."""
    s = series.astype(float).copy()
    if s.empty:
        return s
    lag = max(1, (int(span) - 1) // 2)
    lagged = s.shift(lag)
    de_lagged = s + (s - lagged)
    return de_lagged.ewm(span=span, adjust=False, min_periods=1).mean()

def highpass_filter(series: pd.Series, period: int = 48) -> pd.Series:
    """Causal 2-pole high-pass filter for low-lag detrending."""
    s = series.astype(float).ffill().bfill()
    if s.empty:
        return s
    period = int(max(10, period))
    # Ehlers-style coefficient; stable for trend extraction on daily data.
    denom = np.cos(0.707 * 2 * np.pi / period)
    if abs(denom) < 1e-9:
        denom = 1e-9
    alpha = (
        np.cos(0.707 * 2 * np.pi / period)
        + np.sin(0.707 * 2 * np.pi / period)
        - 1
    ) / denom

    vals = s.to_numpy(dtype=float)
    hp = np.zeros(len(vals), dtype=float)

    if len(vals) < 3:
        return pd.Series(hp, index=s.index)

    a1 = (1 - alpha / 2.0) ** 2
    b1 = 2 * (1 - alpha)
    b2 = (1 - alpha) ** 2

    for i in range(2, len(vals)):
        hp[i] = (
            a1 * (vals[i] - 2 * vals[i - 1] + vals[i - 2])
            + b1 * hp[i - 1]
            - b2 * hp[i - 2]
        )

    return pd.Series(hp, index=s.index)

def linear_forecast_pad(arr: np.ndarray, n_future: int = 12, fit_points: int = 20) -> np.ndarray:
    """Append a small linear forecast to reduce Hilbert edge distortion on the last bar."""
    x = np.asarray(arr, dtype=float)
    if x.size < 3 or n_future <= 0:
        return x.copy()

    fit_points = int(max(3, min(fit_points, x.size)))
    y = x[-fit_points:]
    idx = np.arange(fit_points, dtype=float)

    try:
        slope, intercept = np.polyfit(idx, y, 1)
        future_idx = np.arange(fit_points, fit_points + int(n_future), dtype=float)
        future = slope * future_idx + intercept
        return np.concatenate([x, future])
    except Exception:
        return x.copy()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series):
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr_w = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr_w
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr_w
    )

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid, mid + std_mult * std, mid - std_mult * std

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    return (direction * df["Volume"].fillna(0.0)).cumsum()

def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"].fillna(0.0)

    price_range = (high - low).replace(0, np.nan)
    mfm = (((close - low) - (high - close)) / price_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mfv = mfm * volume
    cmf = mfv.rolling(period, min_periods=period).sum() / volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    return cmf

def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    raw_money_flow = typical_price * df["Volume"].fillna(0.0)
    delta = typical_price.diff()
    positive_mf = raw_money_flow.where(delta > 0, 0.0)
    negative_mf = raw_money_flow.where(delta < 0, 0.0).abs()
    pos_sum = positive_mf.rolling(period, min_periods=period).sum()
    neg_sum = negative_mf.rolling(period, min_periods=period).sum().replace(0, np.nan)
    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))

def stochastic_oscillator(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple[pd.Series, pd.Series]:
    low_min = df["Low"].rolling(period, min_periods=period).min()
    high_max = df["High"].rolling(period, min_periods=period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["Close"] - low_min) / denom
    k = k.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(smooth_d, min_periods=1).mean()
    return k, d

def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=period).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def rate_of_change(close: pd.Series, period: int = 12) -> pd.Series:
    return close.pct_change(periods=period) * 100

def _ensure_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        return d
    if "EMA20" not in d.columns:
        d["EMA20"] = ema(d["Close"], 20)
    if "EMA50" not in d.columns:
        d["EMA50"] = ema(d["Close"], 50)
    if "EMA200" not in d.columns:
        d["EMA200"] = ema(d["Close"], 200)
    if "RSI14" not in d.columns:
        d["RSI14"] = rsi(d["Close"], 14)
    if "MACD_HIST" not in d.columns or "MACD" not in d.columns or "MACD_SIGNAL" not in d.columns:
        d["MACD"], d["MACD_SIGNAL"], d["MACD_HIST"] = macd(d["Close"])
    if "ATR14" not in d.columns:
        d["ATR14"] = atr(d, 14)
    if "ADX14" not in d.columns:
        d["ADX14"] = adx(d, 14)
    if "VOL_SMA20" not in d.columns:
        d["VOL_SMA20"] = d["Volume"].rolling(20).mean()
    if "REL_VOL" not in d.columns:
        d["REL_VOL"] = d["Volume"] / d["VOL_SMA20"]
    if "OBV" not in d.columns:
        d["OBV"] = obv(d)
    if "OBV_SLOPE10" not in d.columns:
        d["OBV_SLOPE10"] = d["OBV"] - d["OBV"].shift(10)
    if "CMF20" not in d.columns:
        d["CMF20"] = chaikin_money_flow(d, 20)
    if "MFI14" not in d.columns:
        d["MFI14"] = money_flow_index(d, 14)
    if "STOCH_K" not in d.columns or "STOCH_D" not in d.columns:
        d["STOCH_K"], d["STOCH_D"] = stochastic_oscillator(d, 14, 3, 3)
    if "CCI20" not in d.columns:
        d["CCI20"] = cci(d, 20)
    if "ROC12" not in d.columns:
        d["ROC12"] = rate_of_change(d["Close"], 12)
    return d

def _score_bucket(value: float, lo: float, hi: float, invert: bool = False) -> float:
    if value is None or pd.isna(value):
        return 50.0
    if hi == lo:
        return 50.0
    x = (float(value) - lo) / (hi - lo)
    x = float(np.clip(x, 0.0, 1.0))
    if invert:
        x = 1.0 - x
    return float(np.clip(x * 100.0, 0.0, 100.0))


def _neutral_mid_score(value: float, center: float = 50.0, width: float = 20.0) -> float:
    """Score values near a preferred midpoint higher, while still tolerating outliers."""
    if value is None or pd.isna(value):
        return 50.0
    width = max(float(width), 1e-6)
    distance = abs(float(value) - float(center))
    score = 100.0 - (distance / width) * 100.0
    return float(np.clip(score, 0.0, 100.0))

def _trend_score(series: pd.Series) -> float:
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3:
        return 50.0
    y = s.tail(min(8, len(s))).to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    try:
        slope = np.polyfit(x, y, 1)[0]
        scale = np.nanmean(np.abs(y)) + 1e-9
        return float(np.clip(50.0 + (slope / scale) * 250.0, 0.0, 100.0))
    except Exception:
        return 50.0

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())

def compute_cycle_features(close: pd.Series) -> tuple[int, int, bool, dict]:
    close = close.dropna()
    n = len(close)
    if n < 30:
        return 20, 999, False, {
            "fft_period": np.nan,
            "hilbert_period": np.nan,
            "autocorr_period": np.nan,
            "weighted_period": 20,
            "fft_confidence": 0.0,
            "hilbert_confidence": 0.0,
            "autocorr_confidence": 0.0,
            "composite_confidence": 0.0,
            "cycle_reliability": 0.0,
            "anchor_idx": np.nan,
            "bars_since_anchor": np.nan,
            "time_to_next_top": np.nan,
            "phase_age_bars": np.nan,
            "phase_age_pct": np.nan,
            "time_to_next_bottom": 999,
            "threshold": np.nan,
            "cycle_position_pct": np.nan,
            "detrend_method": "HighPass+TailHilbert",
            "trend_lag_source": "adaptive_cycle_lag",
            "trend_lag_bars": np.nan,
            "cycle_gate_reason": "",
        }

    series = close.astype(float).copy().ffill().bfill()
    log_close = np.log(series.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    basis = log_close if len(log_close) >= 30 else series
    basis = basis.astype(float).copy().ffill().bfill()
    n_basis = len(basis)

    min_period = 5
    max_period = int(min(120, max(20, n_basis // 2)))
    max_period = max(min_period + 1, max_period)

    # Use a recent cycle window to keep the estimate responsive while still robust.
    cycle_window = int(np.clip(n_basis, 64, 160))
    cycle_arr = basis.to_numpy(dtype=float)[-cycle_window:]
    cycle_arr = cycle_arr - np.nanmean(cycle_arr)
    cycle_arr = np.nan_to_num(cycle_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Low-lag detrending: high-pass filter removes macro trend without center=True leakage.
    hp_period = int(np.clip(n_basis // 3, 20, 60))
    hp_series = highpass_filter(basis, hp_period)
    detrended = hp_series.dropna().astype(float).to_numpy(dtype=float)
    if detrended.size < 20:
        detrended = cycle_arr.copy()
    detrended = detrended - np.nanmean(detrended)
    detrended = np.nan_to_num(detrended, nan=0.0, posinf=0.0, neginf=0.0)

    def confidence_from_peak(peak: float, baseline: float) -> float:
        if not np.isfinite(peak):
            return 0.0
        base = abs(baseline) + 1e-9
        return float(np.clip(peak / base, 0.0, 10.0))

    fft_period = np.nan
    fft_conf = 0.0
    frequencies, power = periodogram(detrended)
    valid = frequencies > 0
    if np.any(valid):
        periods = np.full_like(frequencies, np.inf, dtype=float)
        periods[valid] = 1.0 / frequencies[valid]
        valid = valid & (periods >= min_period) & (periods <= max_period)
    if np.any(valid):
        vf = frequencies[valid]
        vp = power[valid]
        if len(vp) > 0 and np.any(vp > 0):
            best_idx = int(np.argmax(vp))
            fft_freq = float(vf[best_idx])
            if fft_freq > 0:
                fft_period = float(np.clip(round(1 / fft_freq), min_period, max_period))
                fft_conf = confidence_from_peak(float(vp[best_idx]), float(np.median(vp) + 1e-9))

    hilbert_period = np.nan
    hilbert_conf = 0.0
    try:
        hilbert_window = int(np.clip(len(detrended), 64, 160))
        segment = detrended[-hilbert_window:] if len(detrended) > hilbert_window else detrended
        # Forecast-padding on the right tail reduces Hilbert edge distortion on the last bar.
        fit_points = min(20, len(segment))
        pad_future = max(8, min(16, max(8, len(segment) // 8)))
        segment_ext = linear_forecast_pad(segment, n_future=pad_future, fit_points=fit_points)
        analytic = hilbert(segment_ext[: len(segment) + pad_future])
        phase = np.unwrap(np.angle(analytic[: len(segment)]))
        dphase = np.diff(phase)
        if len(dphase) > 0:
            freq_series = np.abs(dphase) / (2 * np.pi)
            freq_series = freq_series[np.isfinite(freq_series) & (freq_series > 0)]
            if len(freq_series) > 0:
                median_freq = float(np.median(freq_series))
                if median_freq > 0:
                    hilbert_period = float(np.clip(round(1 / median_freq), min_period, max_period))
                    hilbert_conf = float(np.clip(1 - np.std(freq_series) / (np.mean(freq_series) + 1e-9), 0.0, 1.0))
    except Exception:
        hilbert_period = np.nan
        hilbert_conf = 0.0

    autocorr_period = np.nan
    autocorr_conf = 0.0
    x = detrended - np.mean(detrended)
    x_std = np.std(x)
    if x_std > 0:
        x = x / x_std
        corr_vals = []
        for lag in range(min_period, max_period + 1):
            if len(x) <= lag + 2:
                break
            c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            if np.isfinite(c):
                corr_vals.append((lag, c))
        if corr_vals:
            lag_arr = np.array([v[0] for v in corr_vals], dtype=float)
            c_arr = np.array([v[1] for v in corr_vals], dtype=float)
            valid_corr = c_arr > 0
            if np.any(valid_corr):
                best_idx = int(np.argmax(c_arr * valid_corr))
                autocorr_period = float(np.clip(lag_arr[best_idx], min_period, max_period))
                autocorr_conf = float(np.clip(c_arr[best_idx], 0.0, 1.0))

    candidates = []
    weights = []
    if np.isfinite(fft_period):
        candidates.append(float(fft_period))
        weights.append(float(np.clip(fft_conf, 0.1, 5.0)))
    if np.isfinite(hilbert_period):
        candidates.append(float(hilbert_period))
        weights.append(float(np.clip(hilbert_conf * 3.0, 0.1, 3.5)))
    if np.isfinite(autocorr_period):
        candidates.append(float(autocorr_period))
        weights.append(float(np.clip(autocorr_conf * 4.0, 0.1, 4.0)))

    if not candidates:
        dominant_period = int(np.clip(20, min_period, max_period))
    else:
        weights_arr = np.array(weights, dtype=float)
        candidate_arr = np.array(candidates, dtype=float)
        weighted_period = float(np.average(candidate_arr, weights=weights_arr))
        dominant_period = int(np.clip(round(weighted_period), min_period, max_period))

    order = int(np.clip(n_basis // 30, 2, 10))
    minima = argrelextrema(series.values, np.less_equal, order=order)[0]
    if len(minima) > 0:
        recent_cutoff = max(0, n_basis - max_period * 2)
        recent_minima = minima[minima >= recent_cutoff]
        anchor_idx = int(recent_minima[-1] if len(recent_minima) else minima[-1])
    else:
        window = min(max_period, n_basis)
        anchor_idx = int(max(0, n_basis - window))

    bars_since_anchor = max(0, (n_basis - 1) - anchor_idx)
    rem = bars_since_anchor % dominant_period
    time_to_next_bottom = 0 if rem == 0 else dominant_period - rem
    half_cycle = max(1, int(round(dominant_period / 2.0)))
    time_to_next_top = (half_cycle - rem) % dominant_period
    phase_age_bars = bars_since_anchor
    phase_age_pct = float(np.clip((phase_age_bars / max(dominant_period, 1)) * 100.0, 0.0, 100.0))
    threshold = max(4, int(round(dominant_period * 0.15)))

    composite_conf = float(np.clip(np.nanmean([fft_conf / 3.0, hilbert_conf, autocorr_conf]), 0.0, 1.0) * 100)
    if len(candidates) >= 2:
        candidate_arr = np.array(candidates, dtype=float)
        spread = float(np.std(candidate_arr) / (np.mean(candidate_arr) + 1e-9))
        agreement_score = float(np.clip(100.0 * (1.0 - spread), 0.0, 100.0))
    elif len(candidates) == 1:
        agreement_score = 60.0
    else:
        agreement_score = 0.0

    reliability = float(np.clip((composite_conf * 0.55) + (agreement_score * 0.45), 0.0, 100.0))
    cycle_ok = ((time_to_next_bottom <= threshold) or (bars_since_anchor <= threshold)) and (reliability >= 45.0)

    # Make lag adaptive to the detected dominant cycle instead of the sample length.
    # This avoids the common "everything becomes 30 bars" behavior from n_basis//3.
    if np.isfinite(dominant_period) and dominant_period > 0:
        if dominant_period <= 18:
            adaptive_lag_factor = 0.22
        elif dominant_period <= 34:
            adaptive_lag_factor = 0.20
        elif dominant_period <= 54:
            adaptive_lag_factor = 0.18
        else:
            adaptive_lag_factor = 0.16
        adaptive_lag = dominant_period * adaptive_lag_factor
    else:
        adaptive_lag_factor = np.nan
        adaptive_lag = max(8.0, n_basis / 8.0)
    trend_lag_bars = int(np.clip(round(adaptive_lag), 4, 36))

    details = {
        "fft_period": int(fft_period) if np.isfinite(fft_period) else np.nan,
        "hilbert_period": int(hilbert_period) if np.isfinite(hilbert_period) else np.nan,
        "autocorr_period": int(autocorr_period) if np.isfinite(autocorr_period) else np.nan,
        "weighted_period": int(dominant_period),
        "fft_confidence": float(np.clip(fft_conf * 10, 0.0, 100.0)),
        "hilbert_confidence": float(np.clip(hilbert_conf * 100, 0.0, 100.0)),
        "autocorr_confidence": float(np.clip(autocorr_conf * 100, 0.0, 100.0)),
        "composite_confidence": composite_conf,
        "cycle_reliability": reliability,
        "anchor_idx": int(anchor_idx),
        "bars_since_anchor": int(bars_since_anchor),
        "threshold": int(threshold),
        "time_to_next_bottom": int(time_to_next_bottom),
        "time_to_next_top": int(time_to_next_top),
        "phase_age_bars": int(phase_age_bars),
        "phase_age_pct": float(phase_age_pct),
        "cycle_position_pct": float(np.clip((rem / max(dominant_period, 1)) * 100.0, 0.0, 100.0)),
        "detrend_method": "HighPass+TailHilbert",
        "trend_lag_bars": trend_lag_bars,
        "trend_lag_source": "adaptive_cycle_lag",
        "cycle_gate_reason": "",
        "cycle_window": int(cycle_window),
        "hilbert_window": int(min(len(detrended), 160)),
        "pad_future": int(max(8, min(16, max(8, len(segment) // 8)))) if 'segment' in locals() else 0,
    }

    return dominant_period, int(time_to_next_bottom), cycle_ok, details

def score_to_grade(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        s = np.nan
    if not np.isfinite(s):
        return "n/a"
    if s >= 90:
        return "A+"
    if s >= 80:
        return "A"
    if s >= 70:
        return "B"
    if s >= 60:
        return "C"
    if s >= 50:
        return "D"
    return "E"

def format_score_delta(delta: float) -> str:
    try:
        if delta is None or pd.isna(delta):
            return "n/a"
        return f"{float(delta):+.2f}"
    except Exception:
        return "n/a"

def compute_institutional_forward_score(
    symbol: str,
    price_df: pd.DataFrame | None = None,
    bench_df: pd.DataFrame | None = None,
    current_fundamental: dict | None = None,
    future_context: dict | None = None,
    technical_context: dict | None = None,
) -> dict:
    """Combine future fundamentals, accumulation, relative strength, quality, and catalyst into one score."""
    symbol = str(symbol).strip()
    current_fundamental = current_fundamental or {}
    future_context = future_context or {}
    technical_context = technical_context or {}

    price = price_df.copy() if price_df is not None else pd.DataFrame()
    if not price.empty:
        price = _ensure_technical_columns(price).dropna().copy()

    bench = bench_df.copy() if bench_df is not None else pd.DataFrame()
    if not bench.empty:
        bench = _ensure_technical_columns(bench).dropna().copy()

    last = price.iloc[-1] if not price.empty else None

    current_fundamental_score = _safe_float(current_fundamental.get("fundamental_score"), np.nan)
    if not np.isfinite(current_fundamental_score):
        current_fundamental_score = 50.0

    future_fundamental_score = _safe_float(future_context.get("future_fundamental_score"), np.nan)
    if not np.isfinite(future_fundamental_score):
        future_fundamental_score = current_fundamental_score

    future_confidence = _safe_float(future_context.get("future_fundamental_confidence"), 50.0)
    future_direction = str(future_context.get("future_fundamental_direction", "Flat"))
    future_phase = str(future_context.get("future_phase", "Unknown"))
    expected_rev_growth = _safe_float(future_context.get("expected_revenue_growth_next_q"), np.nan)
    expected_eps_growth = _safe_float(future_context.get("expected_eps_growth_next_q"), np.nan)
    expected_margin_next_q = _safe_float(future_context.get("expected_margin_next_q"), np.nan)

    quality_score = float(np.clip(current_fundamental_score, 0.0, 100.0))

    smart_money_score = _safe_float(technical_context.get("smart_money_score"), np.nan)
    if not np.isfinite(smart_money_score):
        smart_money_score = 50.0

    cmf_score = 50.0
    obv_score = 50.0
    breakout_score = 50.0
    accel_score = 50.0
    phase_support = 50.0
    if last is not None:
        cmf_score = _score_bucket(_safe_float(last.get("CMF20"), 0.0), -0.15, 0.20)
        obv_slope = _safe_float(last.get("OBV_SLOPE10"), 0.0)
        obv_score = 72.0 if obv_slope > 0 else 28.0 if obv_slope < 0 else 50.0
        close = _safe_float(last.get("Close"), np.nan)
        ema20 = _safe_float(last.get("EMA20"), np.nan)
        if np.isfinite(close) and np.isfinite(ema20) and ema20 != 0:
            breakout_score = _score_bucket((close / ema20) - 1.0, -0.06, 0.18)
        if len(price) >= 30:
            mom20 = price["Close"].pct_change(20).iloc[-1]
            mom60 = price["Close"].pct_change(60).iloc[-1]
            accel_score = _score_bucket(mom20 - mom60, -0.18, 0.22)
        phase_info = classify_8_phase(price) if len(price) >= 60 else {"phase": "Unknown", "phase_confidence": 0.0}
        phase = str(phase_info.get("phase", "Unknown"))
        if phase in {"Early Accumulation", "Accumulation", "Late Accumulation"}:
            phase_support = 82.0
        elif phase in {"Early Markup", "Markup"}:
            phase_support = 76.0
        elif phase in {"Late Markup"}:
            phase_support = 58.0
        elif phase in {"Distribution", "Markdown"}:
            phase_support = 26.0

    if not bench.empty and not price.empty:
        rs_line = compute_relative_strength(price["Close"], bench["Close"])
        if len(rs_line.dropna()) >= 3:
            rs63 = rs_line.pct_change(63).iloc[-1] if len(rs_line) > 63 else np.nan
            rs126 = rs_line.pct_change(126).iloc[-1] if len(rs_line) > 126 else np.nan
            rs252 = rs_line.pct_change(252).iloc[-1] if len(rs_line) > 252 else np.nan
            rs_components = [v for v in [rs63, rs126, rs252] if pd.notna(v)]
            if rs_components:
                rs_score = float(np.clip(
                    (
                        _score_bucket(rs63 if pd.notna(rs63) else np.nan, -0.20, 0.35) * 0.40
                        + _score_bucket(rs126 if pd.notna(rs126) else np.nan, -0.25, 0.50) * 0.35
                        + _score_bucket(rs252 if pd.notna(rs252) else np.nan, -0.30, 0.70) * 0.25
                    ),
                    0.0,
                    100.0,
                ))
            else:
                rs_score = 50.0
        else:
            rs_score = 50.0
    else:
        rs_score = 50.0

    accumulation_score = float(np.clip(
        (smart_money_score * 0.45)
        + (cmf_score * 0.20)
        + (obv_score * 0.20)
        + (phase_support * 0.15),
        0.0,
        100.0,
    ))

    expected_metric_score = 50.0
    if np.isfinite(expected_rev_growth):
        expected_metric_score += _score_bucket(expected_rev_growth, -0.10, 0.35) * 0.35
    if np.isfinite(expected_eps_growth):
        expected_metric_score += _score_bucket(expected_eps_growth, -0.15, 0.55) * 0.35
    if np.isfinite(expected_margin_next_q):
        expected_metric_score += _score_bucket(expected_margin_next_q, 0.03, 0.30) * 0.30

    catalyst_score = float(np.clip(
        (future_confidence * 0.22)
        + (accel_score * 0.20)
        + (breakout_score * 0.18)
        + (phase_support * 0.16)
        + (60.0 if future_direction == "Improving" else 40.0 if future_direction == "Flat" else 25.0) * 0.08
        + (expected_metric_score * 0.16),
        0.0,
        100.0,
    ))

    ifs_score = float(np.clip(
        (future_fundamental_score * 0.30)
        + (accumulation_score * 0.25)
        + (rs_score * 0.20)
        + (quality_score * 0.15)
        + (catalyst_score * 0.10),
        0.0,
        100.0,
    ))

    return {
        "ifs_score": ifs_score,
        "ifs_grade": score_to_grade(ifs_score),
        "ifs_breakdown": {
            "Forward Fundamental": float(future_fundamental_score),
            "Accumulation": float(accumulation_score),
            "Relative Strength": float(rs_score),
            "Quality": float(quality_score),
            "Catalyst": float(catalyst_score),
        },
        "ifs_detail": {
            "future_direction": future_direction,
            "future_phase": future_phase,
            "future_confidence": float(future_confidence),
            "expected_revenue_growth_next_q": float(expected_rev_growth) if np.isfinite(expected_rev_growth) else np.nan,
            "expected_eps_growth_next_q": float(expected_eps_growth) if np.isfinite(expected_eps_growth) else np.nan,
            "expected_margin_next_q": float(expected_margin_next_q) if np.isfinite(expected_margin_next_q) else np.nan,
            "smart_money_score": float(smart_money_score),
            "quality_score": float(quality_score),
            "accumulation_score": float(accumulation_score),
            "relative_strength_score": float(rs_score),
            "catalyst_score": float(catalyst_score),
        },
    }


def _normalize_market_regime(
    macro_phase: str,
    macro_score: float,
    macro_gate_ok: bool,
    macro_phase_confidence: float = 0.0,
    macro_cycle_reliability: float = np.nan,
) -> tuple[str, float, str]:
    """Map the benchmark state into one of the 3 scanner regimes."""
    phase = str(macro_phase or "Unknown").strip().lower()
    score = float(macro_score) if pd.notna(macro_score) else 50.0
    phase_conf = float(np.clip(float(macro_phase_confidence or 0.0) / 100.0, 0.0, 1.0))
    cycle_rel = float(macro_cycle_reliability) if pd.notna(macro_cycle_reliability) else np.nan

    if phase == "markdown" or (score < 45.0 and not macro_gate_ok):
        regime = "BEAR"
        reason = "Benchmark phase weak / markdown"
    elif phase == "distribution":
        regime = "BEAR" if score < 58.0 else "SIDEWAYS"
        reason = "Distribution state"
    elif phase in {"early accumulation", "accumulation", "late accumulation"}:
        regime = "SIDEWAYS" if score < 68.0 else "BULL"
        reason = f"{macro_phase} state"
    elif phase in {"early markup", "markup"}:
        regime = "BULL"
        reason = f"{macro_phase} state"
    elif phase == "late markup":
        regime = "BULL" if score >= 70.0 and macro_gate_ok else "SIDEWAYS"
        reason = "Late markup / mature trend"
    else:
        if score >= 70.0 and macro_gate_ok:
            regime = "BULL"
            reason = "Fallback bull by score"
        elif score <= 48.0 and not macro_gate_ok:
            regime = "BEAR"
            reason = "Fallback bear by score"
        else:
            regime = "SIDEWAYS"
            reason = "Fallback sideways"

    confidence = 0.40 + (phase_conf * 0.35)
    confidence += 0.10 if macro_gate_ok else 0.0
    confidence += 0.08 if np.isfinite(cycle_rel) and cycle_rel >= 55.0 else 0.0
    if regime == "BEAR" and phase in {"markdown", "distribution"}:
        confidence += 0.07
    elif regime == "BULL" and phase in {"early markup", "markup"}:
        confidence += 0.07
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return regime, confidence, reason


def _market_regime_profile(regime: str) -> dict:
    regime = str(regime or "SIDEWAYS").strip().upper()
    profiles = {
        "BEAR": {
            "buy_threshold": 70.0,
            "strong_threshold": 80.0,
            "macro_score_floor": 50.0,
            "trend_rsi_floor": 47.0,
            "trend_soft_floor": 46.0,
            "score_buffer": 3.0,
            "quality_floor": 0.0,
            "strong_quality_floor": 58.0,
            "regime_multiplier": 0.98,
        },
        "SIDEWAYS": {
            "buy_threshold": 70.0,
            "strong_threshold": 80.0,
            "macro_score_floor": 52.0,
            "trend_rsi_floor": 49.0,
            "trend_soft_floor": 48.0,
            "score_buffer": 2.0,
            "quality_floor": 0.0,
            "strong_quality_floor": 56.0,
            "regime_multiplier": 1.00,
        },
        "BULL": {
            "buy_threshold": 70.0,
            "strong_threshold": 80.0,
            "macro_score_floor": 54.0,
            "trend_rsi_floor": 50.0,
            "trend_soft_floor": 49.0,
            "score_buffer": 2.0,
            "quality_floor": 0.0,
            "strong_quality_floor": 54.0,
            "regime_multiplier": 1.04,
        },
    }
    return profiles.get(regime, profiles["SIDEWAYS"]).copy()

def build_macro_liquidity_gate(bench_df: pd.DataFrame, benchmark_symbol: str = "^JKSE") -> dict:
    neutral = {
        "benchmark_symbol": benchmark_symbol,
        "macro_phase": "Unknown",
        "macro_phase_confidence": 0.0,
        "macro_period": np.nan,
        "macro_time_to_bottom": np.nan,
        "macro_time_to_top": np.nan,
        "macro_phase_age_bars": np.nan,
        "macro_phase_age_pct": np.nan,
        "macro_cycle_reliability": 0.0,
        "macro_cycle_gate_reason": "No benchmark data",
        "macro_score": 50.0,
        "macro_gate_ok": True,
        "macro_gate_reason": "OK",
        "macro_multiplier": 1.0,
        "market_regime": "SIDEWAYS",
        "market_regime_confidence": 0.5,
        "market_regime_reason": "No benchmark data",
        "cycle_tuple": (20, 999, False, {}),
        "benchmark_df": bench_df.copy() if bench_df is not None else pd.DataFrame(),
    }

    if bench_df is None or bench_df.empty:
        return neutral

    d = bench_df.copy()
    if d.empty or len(d) < 60:
        neutral["macro_cycle_gate_reason"] = "Benchmark data insufficient"
        neutral["benchmark_df"] = d
        return neutral

    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    d["EMA200"] = ema(d["Close"], 200)
    d["RSI14"] = rsi(d["Close"], 14)
    d["ATR14"] = atr(d, 14)
    d["ADX14"] = adx(d, 14)
    d["VOL_SMA20"] = d["Volume"].rolling(20).mean()
    d["REL_VOL"] = d["Volume"] / d["VOL_SMA20"]
    d["OBV"] = obv(d)
    d["OBV_SLOPE10"] = d["OBV"] - d["OBV"].shift(10)
    d["CMF20"] = chaikin_money_flow(d, 20)
    d["MFI14"] = money_flow_index(d, 14)
    d["STOCH_K"], d["STOCH_D"] = stochastic_oscillator(d, 14, 3, 3)
    d["CCI20"] = cci(d, 20)
    d["ROC12"] = rate_of_change(d["Close"], 12)
    d = d.dropna().copy()

    if d.empty or len(d) < 60:
        neutral["macro_cycle_gate_reason"] = "Benchmark data insufficient after indicators"
        neutral["benchmark_df"] = d
        return neutral

    last = d.iloc[-1]
    cycle_tuple = compute_cycle_features(d["Close"])
    phase_info = classify_8_phase(d)

    dominant_period, time_to_bottom, _, cycle_info = cycle_tuple
    adx_last = float(last["ADX14"]) if pd.notna(last["ADX14"]) else np.nan
    cycle_reliability = float(cycle_info.get("cycle_reliability", np.nan)) if pd.notna(cycle_info.get("cycle_reliability", np.nan)) else np.nan

    macro_phase = str(phase_info.get("phase", "Unknown"))
    macro_phase_confidence = float(phase_info.get("phase_confidence", 0.0))
    macro_time_to_top = int(cycle_info.get("time_to_next_top", np.nan)) if pd.notna(cycle_info.get("time_to_next_top", np.nan)) else np.nan
    macro_phase_age_bars = int(cycle_info.get("phase_age_bars", np.nan)) if pd.notna(cycle_info.get("phase_age_bars", np.nan)) else np.nan
    macro_phase_age_pct = float(cycle_info.get("phase_age_pct", np.nan)) if pd.notna(cycle_info.get("phase_age_pct", np.nan)) else np.nan

    macro_score = 100.0
    reasons = []

    if macro_phase == "Markdown":
        macro_score -= 40.0
        reasons.append("IHSG phase Markdown")
    elif macro_phase == "Distribution":
        macro_score -= 22.0
        reasons.append("IHSG phase Distribution")
    elif macro_phase == "Late Markup":
        macro_score -= 10.0

    if np.isfinite(adx_last) and adx_last > 35:
        macro_score -= 25.0
        reasons.append(f"IHSG ADX {adx_last:.0f} > 35")
    elif np.isfinite(adx_last) and adx_last > 28:
        macro_score -= 12.0
        reasons.append(f"IHSG ADX {adx_last:.0f} elevated")

    if np.isfinite(cycle_reliability) and cycle_reliability < 45:
        macro_score -= 15.0
        reasons.append(f"IHSG CycleRel {cycle_reliability:.0f} < 45")
    elif np.isfinite(cycle_reliability) and cycle_reliability < 60:
        macro_score -= 8.0
        reasons.append(f"IHSG CycleRel {cycle_reliability:.0f} moderate")

    macro_score = float(np.clip(macro_score, 0.0, 100.0))
    macro_gate_ok = (macro_phase != "Markdown") and (not (np.isfinite(adx_last) and adx_last > 35)) and (not (np.isfinite(cycle_reliability) and cycle_reliability < 45)) and (macro_score >= 55.0)
    macro_gate_reason = "OK" if macro_gate_ok else ", ".join(reasons) if reasons else "Macro gate off"
    market_regime, market_regime_confidence, market_regime_reason = _normalize_market_regime(
        macro_phase=macro_phase,
        macro_score=macro_score,
        macro_gate_ok=macro_gate_ok,
        macro_phase_confidence=macro_phase_confidence,
        macro_cycle_reliability=cycle_reliability,
    )
    macro_multiplier = (1.0 if macro_gate_ok else (0.72 if macro_score >= 40 else 0.55))
    macro_multiplier *= _market_regime_profile(market_regime)["regime_multiplier"]

    return {
        "benchmark_symbol": benchmark_symbol,
        "macro_phase": macro_phase,
        "macro_phase_confidence": macro_phase_confidence,
        "macro_period": int(dominant_period) if np.isfinite(dominant_period) else np.nan,
        "macro_time_to_bottom": int(time_to_bottom) if np.isfinite(time_to_bottom) else np.nan,
        "macro_time_to_top": macro_time_to_top,
        "macro_phase_age_bars": macro_phase_age_bars,
        "macro_phase_age_pct": macro_phase_age_pct,
        "macro_cycle_reliability": cycle_reliability if np.isfinite(cycle_reliability) else np.nan,
        "macro_cycle_gate_reason": cycle_info.get("cycle_gate_reason", "OK"),
        "macro_score": macro_score,
        "macro_gate_ok": macro_gate_ok,
        "macro_gate_reason": macro_gate_reason,
        "macro_multiplier": macro_multiplier,
        "market_regime": market_regime,
        "market_regime_confidence": market_regime_confidence,
        "market_regime_reason": market_regime_reason,
        "cycle_tuple": cycle_tuple,
        "benchmark_df": d,
        "benchmark_last": last,
        "benchmark_adx": adx_last,
        "benchmark_cycle_info": cycle_info,
    }

def compute_relative_strength(stock_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    aligned = pd.concat([stock_close.rename("stock"), bench_close.rename("bench")], axis=1, sort=False).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned["stock"] / aligned["bench"]

def classify_8_phase(d: pd.DataFrame) -> dict:
    # Only require core OHLC data; dropping on every NaN makes phase detection
    # fail too often because indicator columns naturally contain NaN on the left edge.
    x = d.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if x.empty or len(x) < 30:
        return {
            "phase": "Unknown",
            "phase_confidence": 0.0,
            "phase_rank": 0.0,
            "phase_reason": "Data historis belum cukup untuk klasifikasi phase.",
            "phase_scores": {},
        }

    last = x.iloc[-1]
    recent = x.tail(min(120, len(x))).copy()

    high20 = float(recent["High"].tail(20).max())
    low20 = float(recent["Low"].tail(20).min())
    high60 = float(recent["High"].max())
    low60 = float(recent["Low"].min())

    def safe_div(a, b):
        return float(a / b) if np.isfinite(b) and b != 0 else np.nan

    pos20 = safe_div(float(last["Close"]) - low20, high20 - low20)
    pos60 = safe_div(float(last["Close"]) - low60, high60 - low60)
    pos20 = float(np.clip(pos20 if np.isfinite(pos20) else 0.5, 0.0, 1.0))
    pos60 = float(np.clip(pos60 if np.isfinite(pos60) else 0.5, 0.0, 1.0))

    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])
    close = float(last["Close"])
    rsi_v = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0
    adx_v = float(last["ADX14"]) if pd.notna(last["ADX14"]) else 0.0
    cmf_v = float(last["CMF20"]) if "CMF20" in x.columns and pd.notna(last["CMF20"]) else 0.0
    mfi_v = float(last["MFI14"]) if "MFI14" in x.columns and pd.notna(last["MFI14"]) else 50.0
    stoch_k_v = float(last["STOCH_K"]) if "STOCH_K" in x.columns and pd.notna(last["STOCH_K"]) else 50.0
    stoch_d_v = float(last["STOCH_D"]) if "STOCH_D" in x.columns and pd.notna(last["STOCH_D"]) else 50.0
    cci_v = float(last["CCI20"]) if "CCI20" in x.columns and pd.notna(last["CCI20"]) else 0.0
    roc_v = float(last["ROC12"]) if "ROC12" in x.columns and pd.notna(last["ROC12"]) else 0.0
    obv_slope = float(last["OBV_SLOPE10"]) if pd.notna(last["OBV_SLOPE10"]) else 0.0

    ema20_slope = float(last["EMA20"] - x["EMA20"].iloc[max(0, len(x) - 6)]) if len(x) >= 6 else 0.0
    atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else max(close * 0.02, 1.0)

    bull_stack = (ema20 > ema50) and (ema50 > ema200)
    bear_stack = (ema20 < ema50) and (ema50 < ema200)
    above_ema20 = close > ema20
    above_ema50 = close > ema50
    above_ema200 = close > ema200
    breakout20 = close > high20 * 1.001
    breakdown20 = close < low20 * 0.999
    extended = (close - ema20) / atr14 if atr14 > 0 else 0.0

    low_regime = float(np.clip(1 - pos60, 0, 1))
    high_regime = float(np.clip(pos60, 0, 1))

    rsi_low = float(np.clip((55 - rsi_v) / 25, 0, 1))
    rsi_mid = float(np.clip(1 - abs(rsi_v - 60) / 18, 0, 1))
    rsi_very_low = float(np.clip((45 - rsi_v) / 20, 0, 1))

    adx_low = float(np.clip((20 - adx_v) / 20, 0, 1))
    adx_mid = float(np.clip(1 - abs(adx_v - 24) / 12, 0, 1))
    adx_high = float(np.clip((adx_v - 18) / 20, 0, 1))

    obv_up_score = float(np.clip((obv_slope > 0) * 1.0, 0, 1))
    obv_down_score = float(np.clip((obv_slope < 0) * 1.0, 0, 1))
    ema_bull = float(np.clip((bull_stack) * 1.0, 0, 1))
    ema_bear = float(np.clip((bear_stack) * 1.0, 0, 1))

    range_width = float((high20 - low20) / close) if close > 0 else 0.0
    compression = float(np.clip(1 - range_width / 0.18, 0, 1))

    scores = {
        "Early Accumulation": (
            low_regime * 35
            + adx_low * 20
            + obv_up_score * 18
            + rsi_low * 12
            + float(ema20_slope >= 0) * 5
            + float(cmf_v > 0) * 8
            + float(mfi_v <= 55) * 6
            + float(not bear_stack) * 10
            + float(cmf_v > 0) * 6
            + float(stoch_k_v >= stoch_d_v) * 6
        ),
        "Accumulation": (
            compression * 25
            + float(np.clip(1 - abs(pos60 - 0.35) / 0.25, 0, 1)) * 20
            + adx_low * 15
            + obv_up_score * 20
            + rsi_mid * 10
            + float(not bear_stack) * 10
        ),
        "Late Accumulation": (
            float(np.clip(1 - abs(pos60 - 0.55) / 0.25, 0, 1)) * 18
            + float(breakout20 or above_ema50) * 25
            + obv_up_score * 18
            + float(ema20 > ema50 or ema20_slope > 0) * 15
            + float(50 <= rsi_v <= 65) * 10
            + float(stoch_k_v >= stoch_d_v) * 8
            + adx_mid * 12
        ),
        "Early Markup": (
            float(breakout20) * 22
            + float(above_ema20 and above_ema50) * 20
            + float(ema20 > ema50) * 18
            + float(ema50 >= ema200) * 8
            + obv_up_score * 15
            + float(52 <= rsi_v <= 68) * 8
            + float(cmf_v > 0) * 6
            + adx_high * 7
        ),
        "Markup": (
            ema_bull * 28
            + float(above_ema20 and above_ema50 and above_ema200) * 15
            + obv_up_score * 18
            + float(55 <= rsi_v <= 75) * 15
            + float(stoch_k_v >= stoch_d_v) * 6
            + adx_high * 16
            + high_regime * 8
        ),
        "Late Markup": (
            ema_bull * 22
            + high_regime * 18
            + float(rsi_v >= 70) * 18
            + float(extended > 1.0) * 14
            + float(mfi_v >= 70) * 8
            + float(adx_v >= 20) * 10
            + obv_down_score * 8
            + float(obv_slope <= 0) * 10
        ),
        "Distribution": (
            high_regime * 24
            + float(rsi_v >= 60) * 10
            + obv_down_score * 20
            + float((close < ema20) or (close < ema50)) * 18
            + float((not breakout20) and (close < high20 * 0.995)) * 14
            + float(ema20_slope <= 0) * 8
            + float((adx_v >= 18) and (adx_v <= 30)) * 6
            + float(cmf_v < 0) * 8
        ),
        "Markdown": (
            ema_bear * 28
            + float(breakdown20) * 20
            + rsi_very_low * 16
            + obv_down_score * 18
            + float(close < ema50) * 10
            + float(pos60 < 0.45) * 8
            + float(adx_v >= 18) * 6
        ),
    }

    phase = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    best = float(sorted_scores[0])
    second = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    confidence = float(np.clip((best - second) + 50, 0, 100))

    reasons = {
        "Early Accumulation": "Harga masih dekat area bawah, OBV mulai membaik, momentum lemah namun stabil.",
        "Accumulation": "Base sedang terbentuk, volatilitas terkompresi, akumulasi relatif dominan.",
        "Late Accumulation": "Harga mulai keluar dari base dan bersiap transisi ke markup.",
        "Early Markup": "Breakout awal dan struktur mulai bullish, namun belum sepenuhnya matang.",
        "Markup": "Struktur bullish sudah jelas, momentum dan trend stack mendukung kelanjutan tren.",
        "Late Markup": "Tren masih naik tetapi sudah extended dan mulai rawan distribusi.",
        "Distribution": "Harga tinggi tetapi momentum melemah, tanda selling into strength mulai muncul.",
        "Markdown": "Struktur bearish dominan, tekanan jual menguasai.",
    }

    return {
        "phase": phase,
        "phase_confidence": confidence,
        "phase_rank": best,
        "phase_reason": reasons.get(phase, "-"),
        "phase_scores": scores,
    }


def detect_reversal_signals(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    if x.empty:
        return x

    x["Bullish_Engulfing"] = (
        (x["Close"] > x["Open"])
        & (x["Close"].shift(1) < x["Open"].shift(1))
        & (x["Close"] >= x["Open"].shift(1))
        & (x["Open"] <= x["Close"].shift(1))
    )

    body = (x["Close"] - x["Open"]).abs()
    candle_range = (x["High"] - x["Low"]).replace(0, np.nan)
    lower_wick = np.minimum(x["Open"], x["Close"]) - x["Low"]
    upper_wick = x["High"] - np.maximum(x["Open"], x["Close"])

    x["Hammer"] = (body / candle_range <= 0.35) & (lower_wick >= body * 2) & (upper_wick <= body)
    x["Inverted_Hammer"] = (body / candle_range <= 0.35) & (upper_wick >= body * 2) & (lower_wick <= body)

    prev2_bear = x["Close"].shift(2) < x["Open"].shift(2)
    prev1_small = (x["Close"].shift(1) - x["Open"].shift(1)).abs() <= (x["High"].shift(1) - x["Low"].shift(1)) * 0.35
    curr_bull = x["Close"] > x["Open"]
    x["Morning_Star"] = prev2_bear & prev1_small & curr_bull & (x["Close"] > (x["Open"].shift(2) + x["Close"].shift(2)) / 2)

    x["EMA20_Reclaim"] = (x["Close"] > x["EMA20"]) & (x["Close"].shift(1) <= x["EMA20"].shift(1))
    x["MACD_Bull_Cross"] = (x["MACD"] > x["MACD_SIGNAL"]) & (x["MACD"].shift(1) <= x["MACD_SIGNAL"].shift(1))
    x["RSI_Bounce"] = (x["RSI14"] > 50) & (x["RSI14"].shift(1) <= 50)
    x["Breakout_5D"] = x["Close"] > x["High"].rolling(5).max().shift(1)

    # --- CONFIRMED PIVOTS (NO REPAINT): the prior bar is only marked once the current bar closes.
    x["Pivot_Low_Confirmed"] = (
        (x["Low"].shift(1) < x["Low"].shift(2))
        & (x["Low"].shift(1) < x["Low"])
    ).fillna(False)
    x["Pivot_High_Confirmed"] = (
        (x["High"].shift(1) > x["High"].shift(2))
        & (x["High"].shift(1) > x["High"])
    ).fillna(False)

    # Keep the old names for compatibility, but now they are confirmed pivots only.
    x["Swing_Low"] = x["Pivot_Low_Confirmed"]
    x["Swing_High"] = x["Pivot_High_Confirmed"]

    # --- ICT / PRICE ACTION UNICORN MODEL ---
    x["Bullish_FVG"] = ((x["Low"] > x["High"].shift(2)) & (x["Close"].shift(1) > x["Open"].shift(1))).fillna(False)
    x["FVG_Top"] = np.where(x["Bullish_FVG"], x["Low"], np.nan)
    x["FVG_Bottom"] = np.where(x["Bullish_FVG"], x["High"].shift(2), np.nan)

    x["Breaker_Top"] = np.nan
    x["Breaker_Bottom"] = np.nan
    x["Liquidity_Sweep_Low"] = np.nan
    x["Unicorn_Setup"] = False

    last_confirmed_pivot_low = np.nan
    breaker_top = np.nan
    breaker_bottom = np.nan
    sweep_low = np.nan
    breaker_source_bar = None

    for i in range(3, len(x)):
        # A confirmed pivot low is available on the current bar when bar i-1 is the actual pivot.
        if bool(x["Pivot_Low_Confirmed"].iloc[i]):
            pivot_idx = i - 1
            pivot_low = float(x["Low"].iloc[pivot_idx])
            if np.isfinite(last_confirmed_pivot_low):
                # Pure sweep: a lower low than the prior confirmed pivot low.
                if pivot_low < last_confirmed_pivot_low:
                    # Displacement candle: bullish close above the candle before the sweep.
                    if x["Close"].iloc[i] > x["Open"].iloc[i] and x["Close"].iloc[i] > x["High"].iloc[i - 2]:
                        sweep_low = pivot_low
                        breaker_top = float(x["High"].iloc[i - 2])
                        breaker_bottom = float(x["Low"].iloc[i - 2])
                        breaker_source_bar = i - 2
            last_confirmed_pivot_low = pivot_low

        # Breaker / MSS confirmation: once a valid sweep exists, a clean bullish FVG that overlaps
        # the breaker band becomes the actionable Unicorn structure.
        if (
            bool(x["Bullish_FVG"].iloc[i])
            and np.isfinite(breaker_top)
            and np.isfinite(breaker_bottom)
            and np.isfinite(sweep_low)
        ):
            overlap = (x["FVG_Bottom"].iloc[i] <= breaker_top) and (x["FVG_Top"].iloc[i] >= breaker_bottom)
            if overlap:
                x.loc[x.index[i], "Breaker_Top"] = breaker_top
                x.loc[x.index[i], "Breaker_Bottom"] = breaker_bottom
                x.loc[x.index[i], "Liquidity_Sweep_Low"] = sweep_low
                x.loc[x.index[i], "Unicorn_Setup"] = True

    x["Breaker_Top"] = x["Breaker_Top"].ffill()
    x["Breaker_Bottom"] = x["Breaker_Bottom"].ffill()
    x["Liquidity_Sweep_Low"] = x["Liquidity_Sweep_Low"].ffill()
    x["Unicorn_Setup"] = x["Unicorn_Setup"].fillna(False)

    # Sniper is a pure structural retest: price revisits the active FVG zone after a liquidity sweep.
    x["Active_FVG_Top"] = x["FVG_Top"].ffill()
    x["Active_FVG_Bottom"] = x["FVG_Bottom"].ffill()
    x["Unicorn_Sniper"] = (
        x["Unicorn_Setup"]
        & x["Low"].le(x["Active_FVG_Top"])
        & x["Close"].ge(x["Active_FVG_Bottom"])
    ).fillna(False)

    # Override Bullish_OB so dashboard logic reads this pure price-action structure.
    x["Bullish_OB"] = x["Unicorn_Setup"]

    return x

def _safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        out = float(v)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _bar_age(index: pd.Index, current_pos: int, event_pos: int) -> tuple[float, float]:
    """Return age in bars and approximate calendar days from event to the latest bar."""
    age_bars = float(current_pos - event_pos)
    age_days = np.nan
    try:
        if len(index) > current_pos >= 0 and len(index) > event_pos >= 0:
            end_ts = pd.Timestamp(index[current_pos])
            start_ts = pd.Timestamp(index[event_pos])
            if pd.notna(end_ts) and pd.notna(start_ts):
                age_days = float((end_ts.normalize() - start_ts.normalize()).days)
    except Exception:
        age_days = np.nan
    return age_bars, age_days


def _evaluate_latest_bullish_fvg(d: pd.DataFrame, fresh_bars: int = 3, max_age_bars: int = 20) -> dict:
    """Inspect the latest bullish FVG and classify age / validity.

    A bullish FVG is considered:
    - fresh: newly printed within `fresh_bars`
    - valid: not fully mitigated and not older than `max_age_bars`
    - mitigated: price has traded down to the FVG bottom
    """
    out = {
        "index": None,
        "position": np.nan,
        "age_bars": np.nan,
        "age_days": np.nan,
        "top": np.nan,
        "bottom": np.nan,
        "mitigated": False,
        "fresh": False,
        "valid": False,
        "status": "none",
        "fill_position": None,
    }

    if d is None or getattr(d, "empty", True) or "Bullish_FVG" not in d.columns:
        return out

    fvg_positions = np.flatnonzero(d["Bullish_FVG"].fillna(False).to_numpy(dtype=bool))
    if fvg_positions.size == 0:
        return out

    current_pos = len(d) - 1
    for pos in reversed(fvg_positions.tolist()):
        top = _safe_float(d["FVG_Top"].iloc[pos], np.nan)
        bottom = _safe_float(d["FVG_Bottom"].iloc[pos], np.nan)
        if not np.isfinite(top) or not np.isfinite(bottom):
            continue
        if bottom > top:
            bottom, top = top, bottom

        future_lows = d["Low"].iloc[pos + 1 :].astype(float)
        mitigated = False
        fill_pos = None
        if not future_lows.empty:
            fill_mask = future_lows <= bottom
            if bool(fill_mask.any()):
                mitigated = True
                fill_pos = int(pos + 1 + np.flatnonzero(fill_mask.to_numpy(dtype=bool))[0])

        age_bars, age_days = _bar_age(d.index, current_pos, pos)
        fresh = bool((age_bars <= fresh_bars) and not mitigated)
        valid = bool((age_bars <= max_age_bars) and not mitigated)

        if valid or not mitigated:
            out.update(
                {
                    "index": d.index[pos],
                    "position": int(pos),
                    "age_bars": age_bars,
                    "age_days": age_days,
                    "top": float(top),
                    "bottom": float(bottom),
                    "mitigated": bool(mitigated),
                    "fresh": fresh,
                    "valid": valid,
                    "status": "fresh" if fresh else ("active" if valid else "stale"),
                    "fill_position": fill_pos,
                }
            )
            return out

    # If every FVG is mitigated, still return the most recent one for diagnostics.
    pos = int(fvg_positions[-1])
    top = _safe_float(d["FVG_Top"].iloc[pos], np.nan)
    bottom = _safe_float(d["FVG_Bottom"].iloc[pos], np.nan)
    if np.isfinite(top) and np.isfinite(bottom) and bottom > top:
        bottom, top = top, bottom
    age_bars, age_days = _bar_age(d.index, current_pos, pos)
    future_lows = d["Low"].iloc[pos + 1 :].astype(float)
    fill_pos = None
    if not future_lows.empty:
        fill_mask = future_lows <= bottom
        if bool(fill_mask.any()):
            fill_pos = int(pos + 1 + np.flatnonzero(fill_mask.to_numpy(dtype=bool))[0])

    out.update(
        {
            "index": d.index[pos],
            "position": int(pos),
            "age_bars": age_bars,
            "age_days": age_days,
            "top": float(top) if np.isfinite(top) else np.nan,
            "bottom": float(bottom) if np.isfinite(bottom) else np.nan,
            "mitigated": True,
            "fresh": False,
            "valid": False,
            "status": "filled",
            "fill_position": fill_pos,
        }
    )
    return out



def _evaluate_latest_unicorn_setup(d: pd.DataFrame, fresh_bars: int = 3, max_age_bars: int = 20) -> dict:
    """Inspect the latest Unicorn setup and determine whether it is still valid.

    The Unicorn remains a pure price-action structure:
    liquidity sweep -> displacement / breaker -> fresh or active FVG.
    Sniper is only an upgrade layer on top of the Unicorn, not a separate setup.
    """
    out = {
        "index": None,
        "position": np.nan,
        "age_bars": np.nan,
        "age_days": np.nan,
        "setup_valid": False,
        "setup_fresh": False,
        "setup_status": "none",
        "sniper_valid": False,
        "sniper_status": "none",
        "reason": "no_setup",
        "fvg_top": np.nan,
        "fvg_bottom": np.nan,
        "breaker_top": np.nan,
        "breaker_bottom": np.nan,
        "sweep_low": np.nan,
    }

    if d is None or getattr(d, "empty", True) or "Unicorn_Setup" not in d.columns:
        return out

    setup_positions = np.flatnonzero(d["Unicorn_Setup"].fillna(False).to_numpy(dtype=bool))
    if setup_positions.size == 0:
        return out

    current_pos = len(d) - 1
    pos = int(setup_positions[-1])
    row = d.iloc[pos]
    close = _safe_float(d["Close"].iloc[-1], np.nan)
    low = _safe_float(d["Low"].iloc[-1], np.nan)
    fvg_top = _safe_float(row.get("FVG_Top"), np.nan)
    fvg_bottom = _safe_float(row.get("FVG_Bottom"), np.nan)
    breaker_top = _safe_float(row.get("Breaker_Top"), np.nan)
    breaker_bottom = _safe_float(row.get("Breaker_Bottom"), np.nan)
    sweep_low = _safe_float(row.get("Liquidity_Sweep_Low"), np.nan)

    if np.isfinite(fvg_top) and np.isfinite(fvg_bottom) and fvg_bottom > fvg_top:
        fvg_bottom, fvg_top = fvg_top, fvg_bottom

    age_bars, age_days = _bar_age(d.index, current_pos, pos)

    setup_valid = True
    reasons = []

    mss_confirmed = bool(
        np.isfinite(fvg_top)
        and np.isfinite(fvg_bottom)
        and np.isfinite(breaker_top)
        and np.isfinite(breaker_bottom)
        and breaker_top > breaker_bottom
        and fvg_top >= breaker_bottom
        and fvg_bottom <= breaker_top
    )
    sweep_confirmed = bool(np.isfinite(sweep_low) and np.isfinite(breaker_bottom) and sweep_low < breaker_bottom)

    if not np.isfinite(fvg_top) or not np.isfinite(fvg_bottom) or not np.isfinite(breaker_top) or not np.isfinite(breaker_bottom):
        setup_valid = False
        reasons.append("missing_structure_levels")
    if age_bars > max_age_bars:
        setup_valid = False
        reasons.append("setup_too_old")
    if np.isfinite(breaker_bottom) and np.isfinite(close) and close < breaker_bottom:
        setup_valid = False
        reasons.append("close_below_breaker")
    if np.isfinite(fvg_bottom) and np.isfinite(low) and low <= fvg_bottom:
        setup_valid = False
        reasons.append("fvg_fully_mitigated")
    if not sweep_confirmed:
        setup_valid = False
        reasons.append("no_liquidity_sweep")
    if not mss_confirmed:
        setup_valid = False
        reasons.append("mss_not_confirmed")

    # Sniper is a retest after liquidity sweep + MSS.
    sniper_stack_ok = bool(
        setup_valid
        and np.isfinite(fvg_top)
        and np.isfinite(fvg_bottom)
        and np.isfinite(low)
        and np.isfinite(close)
        and (low <= fvg_top)
        and (close >= fvg_bottom)
    )

    fresh = bool(setup_valid and (age_bars <= fresh_bars))
    sniper_valid = bool(setup_valid and sniper_stack_ok and (age_bars <= max_age_bars))

    if setup_valid:
        setup_status = "fresh" if fresh else "valid"
    elif "setup_too_old" in reasons:
        setup_status = "stale"
    else:
        setup_status = "invalid"

    if sniper_valid:
        sniper_status = "valid"
    elif setup_valid:
        sniper_status = "not_confirmed"
    else:
        sniper_status = "invalid"

    out.update(
        {
            "index": d.index[pos],
            "position": int(pos),
            "age_bars": age_bars,
            "age_days": age_days,
            "setup_valid": bool(setup_valid),
            "setup_fresh": bool(fresh),
            "setup_status": setup_status,
            "sniper_valid": bool(sniper_valid),
            "sniper_status": sniper_status,
            "reason": ", ".join(reasons) if reasons else ("OK" if setup_valid else "invalid"),
            "fvg_top": float(fvg_top) if np.isfinite(fvg_top) else np.nan,
            "fvg_bottom": float(fvg_bottom) if np.isfinite(fvg_bottom) else np.nan,
            "breaker_top": float(breaker_top) if np.isfinite(breaker_top) else np.nan,
            "breaker_bottom": float(breaker_bottom) if np.isfinite(breaker_bottom) else np.nan,
            "sweep_low": float(sweep_low) if np.isfinite(sweep_low) else np.nan,
        }
    )
    return out



def _evaluate_latest_pullback_continuation(d: pd.DataFrame, fresh_bars: int = 3, max_age_bars: int = 30) -> dict:
    """Detect a clean pullback-continuation setup based on confirmed pivots and trend.

    The function separates:
    - WATCHLIST: trend + structural pullback exists, but reclaim trigger is not yet ready.
    - ENTRY: same pullback context plus bullish reclaim.
    """
    out = {
        "index": None,
        "position": np.nan,
        "age_bars": np.nan,
        "age_days": np.nan,
        "valid": False,
        "status": "none",
        "reason": "no_setup",
        "entry_zone_low": np.nan,
        "entry_zone_high": np.nan,
        "entry_price": np.nan,
        "stop_price": np.nan,
        "target_1": np.nan,
        "target_2": np.nan,
        "invalidation_level": np.nan,
        "support_anchor": np.nan,
        "resistance_anchor": np.nan,
    }

    if d is None or getattr(d, "empty", True):
        return out

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(d.columns)):
        return out

    last = d.iloc[-1]
    close = _safe_float(last.get("Close"), np.nan)
    open_ = _safe_float(last.get("Open"), np.nan)
    atr_v = _safe_float(last.get("ATR14"), np.nan)
    ema20 = _safe_float(last.get("EMA20"), np.nan)
    ema50 = _safe_float(last.get("EMA50"), np.nan)
    ema200 = _safe_float(last.get("EMA200"), np.nan)
    rsi14 = _safe_float(last.get("RSI14"), np.nan)
    adx14 = _safe_float(last.get("ADX14"), np.nan)
    rel_vol = _safe_float(last.get("REL_VOL"), np.nan)
    low = _safe_float(last.get("Low"), np.nan)
    high = _safe_float(last.get("High"), np.nan)

    if not np.isfinite(close) or close <= 0:
        return out
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close * 0.02, 1.0)

    trend_up = bool(
        (np.isfinite(ema20) and np.isfinite(ema50) and close > ema20 and ema20 >= ema50)
        or (np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(ema200) and close > ema20 and ema20 > ema50 > ema200)
    )

    pivot_high_mask = d.get("Pivot_High_Confirmed", pd.Series(False, index=d.index)).fillna(False)
    pivot_low_mask = d.get("Pivot_Low_Confirmed", pd.Series(False, index=d.index)).fillna(False)

    pivot_highs = d.loc[pivot_high_mask, "High"].dropna()
    pivot_lows = d.loc[pivot_low_mask, "Low"].dropna()

    # Fallbacks so the setup can still exist even when the pivot detector is sparse.
    recent_swing_high = float(d["High"].tail(20).max())
    recent_swing_low = float(d["Low"].tail(20).min())

    last_pivot_high = float(pivot_highs.iloc[-1]) if len(pivot_highs) >= 1 else recent_swing_high
    prev_pivot_high = float(pivot_highs.iloc[-2]) if len(pivot_highs) >= 2 else np.nan
    last_pivot_low = float(pivot_lows.iloc[-1]) if len(pivot_lows) >= 1 else recent_swing_low

    bullish_ob_rows = d.loc[d.get("Bullish_OB", pd.Series(False, index=d.index)).fillna(False)]
    if not bullish_ob_rows.empty:
        ob_idx = bullish_ob_rows.index[-1]
        ob_loc = d.index.get_loc(ob_idx)
        ob_low = float(d["Low"].iloc[ob_loc])
        ob_high = float(d["High"].iloc[ob_loc])
    else:
        ob_low = np.nan
        ob_high = np.nan

    block_anchor_candidates = [last_pivot_low, ob_low, ob_high, ema20, ema50, recent_swing_low]
    block_anchor_candidates = [v for v in block_anchor_candidates if np.isfinite(v)]
    if not block_anchor_candidates:
        out["reason"] = "no_structural_support_block"
        return out

    support_anchor = float(min(block_anchor_candidates))
    resistance_anchor = float(last_pivot_high if np.isfinite(last_pivot_high) else recent_swing_high)

    mitigation_ok = bool(low <= support_anchor + atr_v * 0.35)

    # Support reclaim is a softer trigger than full breakout; this allows a real pullback setup to appear.
    support_reclaim = bool(close > support_anchor and close > open_)

    if np.isfinite(prev_pivot_high):
        structure_ok = bool(last_pivot_high > prev_pivot_high)
    else:
        # When pivots are sparse, accept the broader trend structure instead of rejecting the setup.
        structure_ok = bool(trend_up and close >= ema20)

    # Former requirement was effectively breakout-like. Relax it to a reclaim trigger around EMA / structure.
    reclaim_trigger = np.nan
    if np.isfinite(ema20) and np.isfinite(ema50):
        reclaim_trigger = max(ema20, ema50)
    elif np.isfinite(ema20):
        reclaim_trigger = ema20
    else:
        reclaim_trigger = support_anchor + atr_v * 0.25

    reclaim_ok = bool(np.isfinite(reclaim_trigger) and close >= reclaim_trigger - atr_v * 0.08)

    setup_detected = bool(trend_up and structure_ok and mitigation_ok)
    entry_ready = bool(setup_detected and support_reclaim and reclaim_ok)

    # age from the last confirmed pivot low if available, otherwise from the latest bar
    pivot_low_positions = np.flatnonzero(pivot_low_mask.to_numpy(dtype=bool))
    if pivot_low_positions.size > 0:
        pos = int(pivot_low_positions[-1])
        age_bars, age_days = _bar_age(d.index, len(d) - 1, pos)
    else:
        pos = len(d) - 1
        age_bars, age_days = _bar_age(d.index, len(d) - 1, pos)

    entry_zone_low = float(max(0.0, support_anchor - atr_v * 0.14))
    entry_zone_high = float(support_anchor + atr_v * 0.18)
    entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=0.34)

    stop_price = float(max(min(last_pivot_low, support_anchor) - atr_v * 0.18, 0.0))
    target_1 = float(max(resistance_anchor, entry_price + atr_v * 1.55))
    target_2 = float(max(target_1 + atr_v * 0.80, entry_price + atr_v * 2.70))
    invalidation_level = float(stop_price)

    if age_bars > max_age_bars:
        status = "INVALID"
        reason = "pullback_too_old"
        valid = False
    elif not setup_detected:
        status = "INVALID"
        reason = "pullback_conditions_not_met"
        valid = False
    elif entry_ready:
        status = "ENTRY"
        reason = "OK"
        valid = True
    else:
        status = "WATCHLIST"
        reason = "pullback_detected_wait_reclaim"
        valid = True

    out.update(
        {
            "index": d.index[pos],
            "position": int(pos),
            "age_bars": age_bars,
            "age_days": age_days,
            "valid": bool(valid),
            "status": status,
            "reason": reason,
            "entry_zone_low": entry_zone_low,
            "entry_zone_high": entry_zone_high,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_1": target_1,
            "target_2": target_2,
            "invalidation_level": invalidation_level,
            "support_anchor": support_anchor,
            "resistance_anchor": resistance_anchor,
        }
    )
    return out

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(d.columns)):
        return out

    last = d.iloc[-1]
    close = _safe_float(last.get("Close"), np.nan)
    open_ = _safe_float(last.get("Open"), np.nan)
    high = _safe_float(last.get("High"), np.nan)
    low = _safe_float(last.get("Low"), np.nan)
    atr_v = _safe_float(last.get("ATR14"), np.nan)
    ema20 = _safe_float(last.get("EMA20"), np.nan)
    ema50 = _safe_float(last.get("EMA50"), np.nan)
    ema200 = _safe_float(last.get("EMA200"), np.nan)
    rsi14 = _safe_float(last.get("RSI14"), np.nan)
    adx14 = _safe_float(last.get("ADX14"), np.nan)
    rel_vol = _safe_float(last.get("REL_VOL"), np.nan)

    if not np.isfinite(close) or close <= 0:
        return out
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close * 0.02, 1.0)

    trend_up = bool(
        (np.isfinite(ema20) and np.isfinite(ema50) and close > ema20 and ema20 >= ema50)
        or (np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(ema200) and close > ema20 and ema20 > ema50 > ema200)
    )

    pivot_highs = d.loc[d.get("Pivot_High_Confirmed", pd.Series(False, index=d.index)).fillna(False), "High"].dropna()
    pivot_lows = d.loc[d.get("Pivot_Low_Confirmed", pd.Series(False, index=d.index)).fillna(False), "Low"].dropna()

    if len(pivot_highs) < 2 or len(pivot_lows) < 1:
        out["reason"] = "not_enough_confirmed_pivots"
        return out

    last_pivot_high = float(pivot_highs.iloc[-1])
    prev_pivot_high = float(pivot_highs.iloc[-2])
    last_pivot_low = float(pivot_lows.iloc[-1])

    bullish_ob_rows = d.loc[d.get("Bullish_OB", pd.Series(False, index=d.index)).fillna(False)]
    if not bullish_ob_rows.empty:
        ob_idx = bullish_ob_rows.index[-1]
        ob_loc = d.index.get_loc(ob_idx)
        ob_low = float(d["Low"].iloc[ob_loc])
        ob_high = float(d["High"].iloc[ob_loc])
    else:
        ob_low = np.nan
        ob_high = np.nan

    block_anchor_candidates = [last_pivot_low, ob_low, ob_high]
    block_anchor_candidates = [v for v in block_anchor_candidates if np.isfinite(v)]
    if not block_anchor_candidates:
        out["reason"] = "no_structural_support_block"
        return out

    support_anchor = float(min(block_anchor_candidates))
    resistance_anchor = float(last_pivot_high)
    mitigation_ok = bool(low <= support_anchor + atr_v * 0.30)
    support_reclaim = bool(close > support_anchor and close > open_)
    structure_ok = bool(last_pivot_high > prev_pivot_high)
    reclaim_ok = bool(np.isfinite(last_pivot_high) and close >= last_pivot_high * 0.995)

    valid = bool(
        trend_up
        and structure_ok
        and mitigation_ok
        and support_reclaim
        and reclaim_ok
    )

    age_bars, age_days = _bar_age(d.index, len(d) - 1, len(d) - 1)
    age_bars, age_days = _bar_age(d.index, len(d) - 1, len(d) - 1)
    # The above can be unreliable on non-integer indices, so prefer the last confirmed pivot position if available.
    pivot_low_positions = np.flatnonzero(d.get("Pivot_Low_Confirmed", pd.Series(False, index=d.index)).fillna(False).to_numpy(dtype=bool))
    if pivot_low_positions.size > 0:
        pos = int(pivot_low_positions[-1])
        age_bars, age_days = _bar_age(d.index, len(d) - 1, pos)
    else:
        pos = len(d) - 1

    entry_zone_low = float(max(0.0, support_anchor - atr_v * 0.18))
    entry_zone_high = float(support_anchor + atr_v * 0.16)
    entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=0.28)
    stop_price = float(max(min(last_pivot_low, support_anchor) - atr_v * 0.15, 0.0))
    target_1 = float(max(last_pivot_high, resistance_anchor if np.isfinite(resistance_anchor) else last_pivot_high, entry_price + atr_v * 1.7))
    target_2 = float(max(target_1 + atr_v * 0.8, entry_price + atr_v * 2.8))
    invalidation_level = float(stop_price)

    if age_bars > max_age_bars:
        valid = False
        reason = "pullback_too_old"
    elif entry_price <= stop_price:
        valid = False
        reason = "invalid_risk"
    else:
        reason = "OK" if valid else "pullback_conditions_not_met"

    status = "ENTRY" if valid else ("WATCHLIST" if structure_ok and mitigation_ok else "INVALID")

    out.update(
        {
            "index": d.index[pos],
            "position": int(pos),
            "age_bars": age_bars,
            "age_days": age_days,
            "valid": bool(valid),
            "status": status,
            "reason": reason,
            "entry_zone_low": entry_zone_low,
            "entry_zone_high": entry_zone_high,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_1": target_1,
            "target_2": target_2,
            "invalidation_level": invalidation_level,
            "support_anchor": support_anchor,
            "resistance_anchor": float(last_pivot_high),
        }
    )
    return out


def _evaluate_latest_reversal_accumulation_setup(d: pd.DataFrame, fresh_bars: int = 3, max_age_bars: int = 18) -> dict:
    """Detect a reversal-accumulation setup using confirmed pivots and reversal candles.

    This is a pure structure setup:
    - confirmed pivot-based base
    - accumulation / basing context
    - bullish reversal confirmation
    - reclaim above the base or trend support
    """
    out = {
        "index": None,
        "position": np.nan,
        "age_bars": np.nan,
        "age_days": np.nan,
        "valid": False,
        "status": "none",
        "reason": "no_setup",
        "entry_zone_low": np.nan,
        "entry_zone_high": np.nan,
        "entry_price": np.nan,
        "stop_price": np.nan,
        "target_1": np.nan,
        "target_2": np.nan,
        "invalidation_level": np.nan,
        "support_anchor": np.nan,
        "resistance_anchor": np.nan,
    }

    if d is None or getattr(d, "empty", True):
        return out

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(d.columns)):
        return out

    last = d.iloc[-1]
    close = _safe_float(last.get("Close"), np.nan)
    open_ = _safe_float(last.get("Open"), np.nan)
    high = _safe_float(last.get("High"), np.nan)
    low = _safe_float(last.get("Low"), np.nan)
    atr_v = _safe_float(last.get("ATR14"), np.nan)
    ema20 = _safe_float(last.get("EMA20"), np.nan)
    ema50 = _safe_float(last.get("EMA50"), np.nan)
    ema200 = _safe_float(last.get("EMA200"), np.nan)
    rsi14 = _safe_float(last.get("RSI14"), np.nan)
    adx14 = _safe_float(last.get("ADX14"), np.nan)
    rel_vol = _safe_float(last.get("REL_VOL"), np.nan)

    if not np.isfinite(close) or close <= 0:
        return out
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close * 0.02, 1.0)

    phase_info = classify_8_phase(d)
    phase = str(phase_info.get("phase", "Unknown"))
    accumulation_phase_ok = phase in {"Early Accumulation", "Accumulation", "Late Accumulation"}
    reclaim_phase_ok = phase in {"Early Markup", "Markup", "Late Markup"}

    pivot_highs = d.loc[d.get("Pivot_High_Confirmed", pd.Series(False, index=d.index)).fillna(False), "High"].dropna()
    pivot_lows = d.loc[d.get("Pivot_Low_Confirmed", pd.Series(False, index=d.index)).fillna(False), "Low"].dropna()

    if len(pivot_highs) < 1 or len(pivot_lows) < 2:
        out["reason"] = "not_enough_confirmed_pivots"
        return out

    last_pivot_high = float(pivot_highs.iloc[-1])
    last_pivot_low = float(pivot_lows.iloc[-1])
    prev_pivot_low = float(pivot_lows.iloc[-2])

    lookback = min(len(d), 20)
    window = d.tail(lookback)
    base_high = float(window["High"].max())
    base_low = float(window["Low"].min())
    base_mid = float((base_high + base_low) / 2.0)
    base_width = float(max(base_high - base_low, 1e-9))
    base_width_pct = float(base_width / max(close, 1e-9))
    compression_ok = bool(base_width_pct <= max(0.065, (atr_v / max(close, 1e-9)) * 3.0))

    breakout_reclaim = bool(
        np.isfinite(last_pivot_high)
        and close > last_pivot_high
        and close > open_
    )
    support_reclaim = bool(
        (np.isfinite(ema20) and close > ema20)
        or (np.isfinite(ema50) and close > ema50)
        or close > base_mid
    )
    higher_low_ok = bool(last_pivot_low >= prev_pivot_low * 0.985)
    sweep_reclaim_ok = bool(low <= prev_pivot_low - atr_v * 0.10 and close >= prev_pivot_low)

    # Structural reversal score: compression + reclaim + higher-low behavior.
    reversal_score = int(compression_ok) + int(support_reclaim) + int(higher_low_ok or sweep_reclaim_ok or breakout_reclaim)

    setup_valid = bool(
        (accumulation_phase_ok or reclaim_phase_ok)
        and compression_ok
        and support_reclaim
        and (higher_low_ok or sweep_reclaim_ok or breakout_reclaim)
    )

    resistance_anchor = float(last_pivot_high)
    age_bars, age_days = _bar_age(d.index, len(d) - 1, int(pivot_lows.index[-1]) if isinstance(pivot_lows.index[-1], (int, np.integer)) else len(d) - 1)
    pivot_low_positions = np.flatnonzero(d.get("Pivot_Low_Confirmed", pd.Series(False, index=d.index)).fillna(False).to_numpy(dtype=bool))
    pos = int(pivot_low_positions[-1]) if pivot_low_positions.size > 0 else len(d) - 1
    if pivot_low_positions.size > 0:
        age_bars, age_days = _bar_age(d.index, len(d) - 1, int(pivot_low_positions[-1]))

    entry_zone_low = float(max(0.0, min(last_pivot_low, ema20 if np.isfinite(ema20) else last_pivot_low, ema50 if np.isfinite(ema50) else last_pivot_low) - atr_v * 0.18))
    entry_zone_high = float(max(entry_zone_low, min(base_mid + atr_v * 0.20, close - atr_v * 0.08)))
    entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=0.30)
    stop_price = float(max(min(last_pivot_low, base_low) - atr_v * 0.14, 0.0))
    target_1 = float(max(last_pivot_high, resistance_anchor if np.isfinite(resistance_anchor) else last_pivot_high, entry_price + atr_v * 1.8))
    target_2 = float(max(target_1 + atr_v * 0.85, entry_price + atr_v * 3.0))
    invalidation_level = float(stop_price)

    if age_bars > max_age_bars:
        setup_valid = False
        reason = "reversal_too_old"
    elif entry_price <= stop_price:
        setup_valid = False
        reason = "invalid_risk"
    else:
        reason = "OK" if setup_valid else "accumulation_conditions_not_met"

    status = "ENTRY" if setup_valid else ("WATCHLIST" if (accumulation_phase_ok or reversal_score >= 1 or compression_ok) else "INVALID")

    out.update(
        {
            "index": d.index[pos],
            "position": int(pos),
            "age_bars": age_bars,
            "age_days": age_days,
            "valid": bool(setup_valid),
            "status": status,
            "reason": reason,
            "entry_zone_low": entry_zone_low,
            "entry_zone_high": entry_zone_high,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_1": target_1,
            "target_2": target_2,
            "invalidation_level": invalidation_level,
            "support_anchor": float(_min_finite([last_pivot_low, ema20, ema50, base_low])),
            "resistance_anchor": float(last_pivot_high),
        }
    )
    return out


def _build_consistent_entry_plan(
    stock_res: dict,
    entry_buffer_atr: float = 0.25,
    stop_loss_atr: float = 1.8,
    target_1_atr: float = 2.2,
    target_2_atr: float = 3.8,
) -> dict:
    """Build one canonical entry / stop / target plan for all tabs.

    Priority:
    1) Pullback continuation
    2) Breakout retest
    3) Unicorn retest (sniper is only an upgrade layer, not a separate setup)
    4) Reversal accumulation
    """
    d = stock_res.get("df")
    last = stock_res.get("last")

    empty_plan = {
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
    }

    if d is None or last is None or getattr(d, "empty", True):
        return empty_plan

    try:
        close = _safe_float(last.get("Close"), np.nan)
        open_price = _safe_float(last.get("Open"), np.nan)
        high_price = _safe_float(last.get("High"), np.nan)
        low_price = _safe_float(last.get("Low"), np.nan)
        atr_v = _safe_float(last.get("ATR14"), np.nan)
        ema20 = _safe_float(last.get("EMA20"), np.nan)
        ema50 = _safe_float(last.get("EMA50"), np.nan)
        ema200 = _safe_float(last.get("EMA200"), np.nan)
        vol_sma20 = _safe_float(last.get("VOL_SMA20"), np.nan)
        rel_vol = _safe_float(last.get("REL_VOL"), np.nan)

        if not np.isfinite(close) or close <= 0:
            return empty_plan
        if not np.isfinite(atr_v) or atr_v <= 0:
            atr_v = max(close * 0.02, 1.0)

        recent_pivot_highs = d.loc[d.get("Pivot_High_Confirmed", pd.Series(False, index=d.index)).fillna(False), "High"].dropna()
        recent_pivot_lows = d.loc[d.get("Pivot_Low_Confirmed", pd.Series(False, index=d.index)).fillna(False), "Low"].dropna()
        recent_swing_low = float(recent_pivot_lows.iloc[-1]) if len(recent_pivot_lows) else float(d["Low"].tail(10).min())
        recent_swing_high = float(recent_pivot_highs.iloc[-1]) if len(recent_pivot_highs) else float(d["High"].tail(20).max())
        prev_swing_high = float(recent_pivot_highs.iloc[-2]) if len(recent_pivot_highs) >= 2 else recent_swing_high

        tradeability_gate_ok = bool(stock_res.get("tradeability_gate_ok", True))
        tradeability_gate_reason = str(stock_res.get("tradeability_gate_reason", "OK"))
        tradeability_score = _safe_float(stock_res.get("tradeability_score"), np.nan)

        # Composite score is ranking only. It never blocks a valid setup.
        final_score = _safe_float(stock_res.get("score"), np.nan)

        breakout_reference = _safe_float(stock_res.get("breakout_reference"), np.nan)
        if not np.isfinite(breakout_reference) or breakout_reference <= 0:
            breakout_reference = recent_swing_high

        breakout_invalidation_level = float(max(0.0, breakout_reference - atr_v * 0.25))
        bullish_close_ok = bool(np.isfinite(close) and np.isfinite(open_price) and close > open_price)
        breakout_retest_ok = bool(
            np.isfinite(breakout_reference)
            and np.isfinite(low_price)
            and low_price <= breakout_reference + atr_v * 0.20
            and close > breakout_reference
            and close > open_price
        )
        breakout_reclaim_ok = bool(
            np.isfinite(ema20) and close > ema20
        )
        # Volume is intentionally not used as a hard breakout gate here.
        breakout_confirmed = bool(
            breakout_retest_ok
            and breakout_reclaim_ok
            and close > breakout_invalidation_level
        )

        unicorn_setup_valid = bool(stock_res.get("unicorn_setup_valid", False))
        unicorn_sniper_valid = bool(stock_res.get("unicorn_sniper_valid", False))
        unicorn_setup_status = str(stock_res.get("unicorn_setup_status", "INVALID")).upper()
        unicorn_sniper_status = str(stock_res.get("unicorn_sniper_status", "INVALID")).upper()

        pullback_state = stock_res.get("pullback_continuation_state", {}) or {}
        pullback_valid = bool(stock_res.get("pullback_continuation_valid", False) or pullback_state.get("valid", False))
        pullback_status = str(stock_res.get("pullback_continuation_status", pullback_state.get("status", "INVALID"))).upper()
        pullback_reference = _safe_float(stock_res.get("pullback_continuation_reference"), np.nan)
        if not np.isfinite(pullback_reference):
            pullback_reference = _safe_float(pullback_state.get("support_anchor"), np.nan)
        pullback_invalidation_level = _safe_float(stock_res.get("pullback_continuation_invalidation"), np.nan)
        if not np.isfinite(pullback_invalidation_level):
            pullback_invalidation_level = _safe_float(pullback_state.get("invalidation_level"), np.nan)

        setup_kind = "None"
        setup_variant = "None"
        entry_trigger = "No_signal"
        plan_reason = "No actionable buy signal"
        projected_flow = _projected_entry_flow("None")
        projected_first_leg = projected_flow["projected_first_leg"]
        projected_rebound_leg = projected_flow["projected_rebound_leg"]
        entry_zone_role = projected_flow["entry_zone_role"]
        entry_zone_label = projected_flow["entry_zone_label"]
        entry_projection_summary = projected_flow["entry_projection_summary"]
        retest_anchor = projected_flow["retest_anchor"]

        if pullback_valid:
            profile = _setup_entry_profile("PULLBACK", entry_buffer_atr, stop_loss_atr, target_1_atr, target_2_atr)
            support_anchor = _safe_float(pullback_state.get("support_anchor"), np.nan)
            resistance_anchor = _safe_float(pullback_state.get("resistance_anchor"), np.nan)
            if not np.isfinite(support_anchor):
                support_anchor = _min_finite([ema20, ema50, recent_swing_low])
            if not np.isfinite(resistance_anchor):
                resistance_anchor = recent_swing_high
            support_ref = _max_finite([support_anchor, ema20, ema50, recent_swing_low], default=np.nan)
            if not np.isfinite(support_ref):
                support_ref = close - atr_v * 0.40
            zone_mid = support_ref
            entry_zone_low = max(0.0, float(zone_mid - profile["entry_buffer_atr"] * atr_v))
            entry_zone_high = float(max(entry_zone_low, min(zone_mid + atr_v * 0.14, close - atr_v * 0.03)))
            entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=profile["entry_bias"])
            stop_candidates = [
                _safe_float(pullback_state.get("stop_price"), np.nan),
                support_ref,
                recent_swing_low,
            ]
            stop_candidates = [v for v in stop_candidates if np.isfinite(v)]
            stop_price = float(max(min(stop_candidates) - atr_v * 0.22, 0.0)) if stop_candidates else max(support_ref - atr_v * 0.22, 0.0)
            setup_kind = "Pullback"
            target_1, target_2 = _setup_take_profit_pair_long(
                "Pullback",
                recent_pivot_highs,
                entry_price,
                atr_v,
                recent_swing_high=recent_swing_high,
                recent_swing_low=recent_swing_low,
                resistance_anchor=resistance_anchor,
                support_anchor=support_ref,
            )
            setup_variant = "Continuation"
            entry_trigger = "Trend_Pullback_Retest"
            plan_reason = "Pullback continuation: support mitigation, reclaim, and liquidity target above"
            projected_flow = _projected_entry_flow(
                "PULLBACK",
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                support_anchor=support_ref,
                resistance_anchor=resistance_anchor,
            )
            projected_first_leg = projected_flow["projected_first_leg"]
            projected_rebound_leg = projected_flow["projected_rebound_leg"]
            entry_zone_role = projected_flow["entry_zone_role"]
            entry_zone_label = projected_flow["entry_zone_label"]
            entry_projection_summary = projected_flow["entry_projection_summary"]
            retest_anchor = projected_flow["retest_anchor"]

        elif breakout_confirmed:
            profile = _setup_entry_profile("BREAKOUT", entry_buffer_atr, stop_loss_atr, target_1_atr, target_2_atr)
            lower_candidates = [
                breakout_reference - profile["entry_buffer_atr"] * atr_v,
                breakout_reference - atr_v * 0.12,
                recent_swing_low - atr_v * 0.08,
            ]
            upper_candidates = [
                breakout_reference + profile["entry_buffer_atr"] * atr_v * 0.25,
                breakout_reference + atr_v * 0.10,
                close,
            ]
            entry_zone_low = float(max(0.0, _min_finite(lower_candidates, default=breakout_reference - atr_v * 0.18)))
            entry_zone_high = max(entry_zone_low, float(min(_max_finite(upper_candidates, default=breakout_reference + atr_v * 0.12), breakout_reference + atr_v * 0.18)))
            entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=profile["entry_bias"])
            stop_candidates = [
                breakout_reference,
                breakout_invalidation_level,
                recent_swing_low,
            ]
            stop_candidates = [v for v in stop_candidates if np.isfinite(v)]
            stop_price = float(max(min(stop_candidates) - atr_v * 0.16, 0.0)) if stop_candidates else max(breakout_reference - atr_v * 0.18, 0.0)
            breakout_range_height = np.nan
            if np.isfinite(recent_swing_high) and np.isfinite(recent_swing_low):
                breakout_range_height = max(recent_swing_high - recent_swing_low, atr_v * 1.0)
            else:
                breakout_range_height = atr_v * 1.0
            setup_kind = "Breakout"
            target_1, target_2 = _setup_take_profit_pair_long(
                "Breakout",
                recent_pivot_highs,
                entry_price,
                atr_v,
                recent_swing_high=recent_swing_high,
                recent_swing_low=recent_swing_low,
                breakout_reference=breakout_reference,
                breakout_range_height=breakout_range_height,
            )
            setup_variant = "Retest"
            entry_trigger = "Breakout_Retest_Above_Former_Resistance"
            plan_reason = "Breakout plan: break, retest, and liquidity expansion above the range"
            projected_flow = _projected_entry_flow(
                "BREAKOUT",
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                breakout_reference=breakout_reference,
                support_anchor=recent_swing_low,
                resistance_anchor=breakout_reference,
            )
            projected_first_leg = projected_flow["projected_first_leg"]
            projected_rebound_leg = projected_flow["projected_rebound_leg"]
            entry_zone_role = projected_flow["entry_zone_role"]
            entry_zone_label = projected_flow["entry_zone_label"]
            entry_projection_summary = projected_flow["entry_projection_summary"]
            retest_anchor = projected_flow["retest_anchor"]

        elif unicorn_setup_valid:
            profile_kind = "SNIPER" if unicorn_sniper_valid else "UNICORN"
            profile = _setup_entry_profile(profile_kind, entry_buffer_atr, stop_loss_atr, target_1_atr, target_2_atr)
            fvg_bottom = _safe_float(stock_res.get("unicorn_fvg_bottom"), np.nan)
            fvg_top = _safe_float(stock_res.get("unicorn_fvg_top"), np.nan)
            sweep_low = _safe_float(stock_res.get("unicorn_sweep_low"), np.nan)
            breaker_bottom = _safe_float(stock_res.get("unicorn_breaker_bottom"), np.nan)
            if not np.isfinite(fvg_bottom):
                fvg_bottom = _min_finite([breaker_bottom, recent_swing_low, close])
            if not np.isfinite(fvg_top):
                fvg_top = max(fvg_bottom + atr_v * 0.10, close)
            entry_zone_low = max(0.0, float(fvg_bottom - profile["entry_buffer_atr"] * atr_v))
            entry_zone_high = float(max(entry_zone_low, min(fvg_top + atr_v * 0.10, close - atr_v * 0.03)))
            entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=profile["entry_bias"])
            stop_candidates = [
                sweep_low,
                breaker_bottom,
                fvg_bottom,
                recent_swing_low,
            ]
            stop_candidates = [v for v in stop_candidates if np.isfinite(v)]
            stop_price = float(max(min(stop_candidates) - atr_v * 0.20, 0.0)) if stop_candidates else max(fvg_bottom - atr_v * 0.20, 0.0)
            setup_kind = "Sniper" if unicorn_sniper_valid else "Unicorn"
            target_1, target_2 = _setup_take_profit_pair_long(
                setup_kind,
                recent_pivot_highs,
                entry_price,
                atr_v,
                recent_swing_high=recent_swing_high,
                recent_swing_low=recent_swing_low,
                fvg_bottom=fvg_bottom,
                fvg_top=fvg_top,
                breaker_bottom=breaker_bottom,
                sweep_low=sweep_low,
            )
            setup_variant = "Sniper" if unicorn_sniper_valid else "Base"
            entry_trigger = "Liquidity_Sweep_MSS_FVG_Retest"
            plan_reason = f"{setup_kind} plan: liquidity sweep, MSS confirmation, and FVG retest"
            projected_flow = _projected_entry_flow(
                profile_kind,
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                fvg_bottom=fvg_bottom,
                fvg_top=fvg_top,
                breaker_bottom=breaker_bottom,
                sweep_low=sweep_low,
                resistance_anchor=recent_swing_high,
            )
            projected_first_leg = projected_flow["projected_first_leg"]
            projected_rebound_leg = projected_flow["projected_rebound_leg"]
            entry_zone_role = projected_flow["entry_zone_role"]
            entry_zone_label = projected_flow["entry_zone_label"]
            entry_projection_summary = projected_flow["entry_projection_summary"]
            retest_anchor = projected_flow["retest_anchor"]

        elif reversal_accumulation_valid:
            profile = _setup_entry_profile("REVERSAL", entry_buffer_atr, stop_loss_atr, target_1_atr, target_2_atr)
            support_anchor = _safe_float(reversal_accumulation_state.get("support_anchor"), np.nan)
            resistance_anchor = _safe_float(reversal_accumulation_state.get("resistance_anchor"), np.nan)
            swept_low = _safe_float(reversal_accumulation_state.get("sweep_low"), np.nan)
            if not np.isfinite(support_anchor):
                support_anchor = _min_finite([ema20, ema50, recent_swing_low])
            if not np.isfinite(resistance_anchor):
                resistance_anchor = recent_swing_high
            support_ref = _max_finite([support_anchor, ema20, ema50, recent_swing_low], default=np.nan)
            if not np.isfinite(support_ref):
                support_ref = close - atr_v * 0.45
            zone_mid = support_ref
            entry_zone_low = max(0.0, float(zone_mid - profile["entry_buffer_atr"] * atr_v))
            entry_zone_high = float(max(entry_zone_low, min(zone_mid + atr_v * 0.16, close - atr_v * 0.03)))
            entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=profile["entry_bias"])
            stop_candidates = [
                _safe_float(reversal_accumulation_state.get("stop_price"), np.nan),
                swept_low,
                support_ref,
                recent_swing_low,
            ]
            stop_candidates = [v for v in stop_candidates if np.isfinite(v)]
            stop_price = float(max(min(stop_candidates) - atr_v * 0.22, 0.0)) if stop_candidates else max(support_ref - atr_v * 0.22, 0.0)
            setup_kind = "Reversal"
            target_1, target_2 = _setup_take_profit_pair_long(
                "Reversal",
                recent_pivot_highs,
                entry_price,
                atr_v,
                recent_swing_high=recent_swing_high,
                recent_swing_low=recent_swing_low,
                support_anchor=support_ref,
                resistance_anchor=resistance_anchor,
                sweep_low=swept_low,
            )
            setup_variant = "Accumulation"
            entry_trigger = "Liquidity_Sweep_MSS_Reclaim"
            plan_reason = "Reversal plan: liquidity sweep, MSS reclaim, and continuation into overhead liquidity"
            projected_flow = _projected_entry_flow(
                "REVERSAL",
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                support_anchor=support_ref,
                resistance_anchor=resistance_anchor,
                sweep_low=swept_low,
            )
            projected_first_leg = projected_flow["projected_first_leg"]
            projected_rebound_leg = projected_flow["projected_rebound_leg"]
            entry_zone_role = projected_flow["entry_zone_role"]
            entry_zone_label = projected_flow["entry_zone_label"]
            entry_projection_summary = projected_flow["entry_projection_summary"]
            retest_anchor = projected_flow["retest_anchor"]


        else:
            empty_plan["plan_reason"] = "No setup valid"
            empty_plan.update(_build_setup_lifecycle_snapshot(stock_res, None, plan_reason=empty_plan["plan_reason"]))
            return empty_plan

        # Keep entry anchored to the structural retrace zone instead of chasing price.
        # If the market is already far away, the lifecycle snapshot will mark it as watchlist/late.

        if stop_price >= entry_price:
            stop_price = max(entry_price - atr_v * 0.90, 0.0)

        risk_per_share = float(max(entry_price - stop_price, 1e-9))
        rr1 = float((target_1 - entry_price) / risk_per_share)
        rr2 = float((target_2 - entry_price) / risk_per_share)
        upside_t1 = float((target_1 / entry_price - 1.0) * 100.0)
        upside_t2 = float((target_2 / entry_price - 1.0) * 100.0)

        setup_rr_floor = (profile["rr_floor_1"], profile["rr_floor_2"])
        setup_max_risk_pct = profile["max_risk_pct"]
        setup_pullback_floor_atr = profile["pullback_floor_atr"]

        risk_pct = float((entry_price - stop_price) / max(entry_price, 1e-9))
        pullback_atr = float((close - entry_price) / max(atr_v, 1e-9))
        distance_to_entry_atr = _setup_distance_to_entry_atr(close, entry_price, atr_v)
        has_sweep = bool(np.isfinite(stock_res.get("unicorn_sweep_low", np.nan)) or setup_kind in {"Unicorn", "Sniper"} and bool(stock_res.get("unicorn_setup", False)))
        has_mss = bool(setup_kind in {"Unicorn", "Sniper"} and (bool(stock_res.get("unicorn_setup_valid", False)) or bool(stock_res.get("unicorn_sniper_valid", False))))
        fill_probability = _estimate_setup_fill_probability(
            setup_kind,
            distance_to_entry_atr,
            age_bars=stock_res.get("setup_age_bars", np.nan),
            setup_valid=True,
            has_liquidity_sweep=has_sweep,
            has_mss=has_mss,
            rr1=rr1,
            rr2=rr2,
            setup_fresh=bool(stock_res.get("unicorn_setup_fresh", False)),
            entry_zone_width_atr=(
                abs(entry_zone_high - entry_zone_low) / max(atr_v, 1e-9) if np.isfinite(entry_zone_low) and np.isfinite(entry_zone_high) else np.nan
            ),
            structure_confluence=_setup_structure_confluence_score(
                entry_price,
                atr_v,
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                recent_swing_low=recent_swing_low,
                recent_swing_high=recent_swing_high,
                sweep_low=_safe_float(stock_res.get("unicorn_sweep_low", np.nan), np.nan),
                breaker_bottom=_safe_float(stock_res.get("unicorn_breaker_bottom", np.nan), np.nan),
                fvg_bottom=_safe_float(stock_res.get("unicorn_fvg_bottom", np.nan), np.nan),
                fvg_top=_safe_float(stock_res.get("unicorn_fvg_top", np.nan), np.nan),
                support_anchor=_safe_float(stock_res.get("support_anchor", np.nan), np.nan),
                resistance_anchor=_safe_float(stock_res.get("resistance_anchor", np.nan), np.nan),
            ),
        )
        entry_valid = bool(np.isfinite(entry_price) and np.isfinite(stop_price) and entry_price > stop_price)

        # Late / far-away retests should fall back to watchlist rather than pretend to be ready.
        fill_prob_floor = 30.0 if setup_kind in {"Unicorn", "Sniper"} else 27.0
        distance_cap_atr = 6.0 if setup_kind in {"Unicorn", "Sniper"} else 6.5
        setup_stale = bool(np.isfinite(distance_to_entry_atr) and distance_to_entry_atr > distance_cap_atr)

        execution_ready = bool(
            stock_res.get("tradeability_gate_ok", True)
            and entry_valid
            and rr1 >= setup_rr_floor[0]
            and rr2 >= setup_rr_floor[1]
            and risk_pct <= setup_max_risk_pct
            and (not np.isfinite(pullback_atr) or pullback_atr >= max(0.0, setup_pullback_floor_atr - 0.02))
            and (not np.isfinite(fill_probability) or fill_probability >= fill_prob_floor)
            and not setup_stale
        )

        watchlist_entry = bool((not execution_ready) and (entry_valid or breakout_confirmed or pullback_valid or unicorn_setup_valid or unicorn_sniper_valid))

        filter_reasons = []
        if not entry_valid:
            filter_reasons.append("invalid_entry")
        if rr1 < setup_rr_floor[0]:
            filter_reasons.append(f"rr1<{setup_rr_floor[0]:.2f}")
        if rr2 < setup_rr_floor[1]:
            filter_reasons.append(f"rr2<{setup_rr_floor[1]:.2f}")
        if risk_pct > setup_max_risk_pct:
            filter_reasons.append(f"risk_pct>{setup_max_risk_pct:.2%}")
        if np.isfinite(pullback_atr) and pullback_atr < setup_pullback_floor_atr:
            filter_reasons.append(f"pullback_atr<{setup_pullback_floor_atr:.2f}")

        if not entry_valid and not (breakout_confirmed or unicorn_setup_valid or pullback_valid or reversal_accumulation_valid):
            empty_plan["plan_reason"] = plan_reason
            empty_plan.update(_build_setup_lifecycle_snapshot(stock_res, None, plan_reason=empty_plan["plan_reason"]))
            return empty_plan

        candidate_allowed = bool(breakout_confirmed or unicorn_setup_valid or pullback_valid or reversal_accumulation_valid or unicorn_sniper_valid)
        candidate_label = "EXECUTION_READY" if execution_ready else ("WATCHLIST_ENTRY" if candidate_allowed else "NONE")
        if setup_stale and candidate_allowed:
            candidate_label = "WATCHLIST_ENTRY"
        execution_reason = (
            "Tradeability gate OK"
            if execution_ready
            else (
                f"Far from entry / low fill probability ({fill_probability:.1f}%)" if setup_stale and candidate_allowed and np.isfinite(fill_probability)
                else (
                    f"Tradeability gate blocked: {stock_res.get('tradeability_gate_reason', 'n/a')}"
                    if not stock_res.get("tradeability_gate_ok", True) and candidate_allowed
                    else ("; ".join(filter_reasons) if filter_reasons else plan_reason)
                )
            )
        )

        return {
            "entry_zone_low": entry_zone_low,
            "entry_zone_high": entry_zone_high,
            "entry_price_plan": entry_price,
            "entry_trigger": entry_trigger,
            "stop_loss_plan": stop_price,
            "target_1": target_1,
            "target_2": target_2,
            "risk_per_share": risk_per_share,
            "risk_reward_1": rr1,
            "risk_reward_2": rr2,
            "upside_to_t1_pct": upside_t1,
            "upside_to_t2_pct": upside_t2,
            "plan_reason": plan_reason,
            "setup_kind": setup_kind,
            "setup_variant": setup_variant,
            "breakout_confirmed": breakout_confirmed,
            "breakout_reference": breakout_reference,
            "breakout_invalidation_level": breakout_invalidation_level,
            "pullback_valid": pullback_valid,
            "pullback_reference": pullback_reference,
            "pullback_invalidation_level": pullback_invalidation_level,
            "entry_valid": entry_valid,
            "entry_mode": setup_kind,
            "tradeability_ok": bool(stock_res.get("tradeability_gate_ok", False)),
            "tradeability_score": float(_safe_float(stock_res.get("tradeability_score"), np.nan)),
            "tradeability_reason": stock_res.get("tradeability_gate_reason", "n/a"),
            "setup_distance_to_entry_atr": float(distance_to_entry_atr) if np.isfinite(distance_to_entry_atr) else np.nan,
            "setup_fill_probability": float(fill_probability) if np.isfinite(fill_probability) else np.nan,
            "execution_status": candidate_label,
            "execution_status_reason": execution_reason,
            "entry_candidate_label": candidate_label,
            "candidate_entry_price": entry_price,
            "candidate_stop_price": stop_price,
            "candidate_target_1": target_1,
            "candidate_target_2": target_2,
            "candidate_entry_zone_low": entry_zone_low,
            "candidate_entry_zone_high": entry_zone_high,
            "candidate_risk_reward_1": rr1,
            "candidate_risk_reward_2": rr2,
            "projected_first_leg": projected_first_leg,
            "projected_rebound_leg": projected_rebound_leg,
            "entry_zone_role": entry_zone_role,
            "entry_zone_label": entry_zone_label,
            "entry_projection_summary": entry_projection_summary,
            "retest_anchor": retest_anchor,
            **_build_setup_lifecycle_snapshot(stock_res, {
                "entry_price_plan": entry_price,
                "stop_loss_plan": stop_price,
                "target_1": target_1,
                "target_2": target_2,
                "entry_zone_low": entry_zone_low,
                "entry_zone_high": entry_zone_high,
                "risk_reward_1": rr1,
                "risk_reward_2": rr2,
                "setup_kind": setup_kind,
            }, plan_reason=plan_reason),
        }
    except Exception as e:
        empty_plan["plan_reason"] = f"Plan error: {e}"
        empty_plan.update(_build_setup_lifecycle_snapshot(stock_res, None, plan_reason=empty_plan["plan_reason"]))
        return empty_plan

def _weekly_mtf_confirmation(df: pd.DataFrame) -> dict:
    """Higher-timeframe confirmation derived from daily OHLCV.

    The goal is not to optimize returns on every swing. The goal is to reduce
    false positives by requiring a weekly trend backdrop for EXECUTION_READY
    while still allowing WATCHLIST_ENTRY candidates to be surfaced.
    """
    result = {
        "weekly_mtf_ok": False,
        "weekly_mtf_score": np.nan,
        "weekly_mtf_reason": "n/a",
    }

    if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 80:
        result["weekly_mtf_reason"] = "Insufficient history for weekly confirmation"
        return result

    try:
        wk = df.copy()
        if not isinstance(wk.index, pd.DatetimeIndex):
            wk.index = pd.to_datetime(wk.index, errors="coerce")
        wk = wk.loc[wk.index.notna(), ["Open", "High", "Low", "Close", "Volume"]].dropna()
        if wk.empty:
            result["weekly_mtf_reason"] = "Weekly resample failed"
            return result

        weekly = wk.resample("W-FRI").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        ).dropna()
        if len(weekly) < 12:
            result["weekly_mtf_reason"] = "Insufficient weekly bars"
            return result

        weekly = _ensure_technical_columns(weekly)
        last = weekly.iloc[-1]

        close_v = _safe_float(last.get("Close"), np.nan)
        ema20_v = _safe_float(last.get("EMA20"), np.nan)
        ema50_v = _safe_float(last.get("EMA50"), np.nan)
        ema200_v = _safe_float(last.get("EMA200"), np.nan)
        macd_hist_v = _safe_float(last.get("MACD_HIST"), np.nan)
        rsi_v = _safe_float(last.get("RSI14"), np.nan)
        adx_v = _safe_float(last.get("ADX14"), np.nan)
        rel_vol_v = _safe_float(last.get("REL_VOL"), np.nan)

        score = 0.0
        reasons: list[str] = []

        if np.isfinite(close_v) and np.isfinite(ema20_v) and close_v > ema20_v:
            score += 28.0
        else:
            reasons.append("Weekly close below EMA20")

        if np.isfinite(ema20_v) and np.isfinite(ema50_v) and ema20_v > ema50_v:
            score += 20.0
        else:
            reasons.append("Weekly EMA20 below EMA50")

        if np.isfinite(ema50_v) and np.isfinite(ema200_v) and ema50_v > ema200_v:
            score += 14.0

        if np.isfinite(macd_hist_v) and macd_hist_v > 0:
            score += 18.0
        else:
            reasons.append("Weekly MACD histogram not positive")

        if np.isfinite(rsi_v) and rsi_v >= 52:
            score += 12.0
        elif np.isfinite(rsi_v):
            reasons.append(f"Weekly RSI {rsi_v:.1f} < 52")

        if np.isfinite(adx_v) and adx_v >= 15:
            score += 4.0

        if np.isfinite(rel_vol_v) and rel_vol_v >= 1.0:
            score += 4.0

        score = float(np.clip(score, 0.0, 100.0))
        weekly_ok = bool(score >= 60.0 and np.isfinite(close_v) and np.isfinite(ema20_v) and close_v > ema20_v)

        result["weekly_mtf_ok"] = weekly_ok
        result["weekly_mtf_score"] = score
        result["weekly_mtf_reason"] = "OK" if weekly_ok else "; ".join(reasons) if reasons else "Weekly alignment incomplete"
        return result
    except Exception as exc:
        result["weekly_mtf_reason"] = f"weekly_mtf_error: {type(exc).__name__}"
        return result

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
    d = df.copy()
    if d.empty or len(d) < min_history_bars:
        return {"valid": False, "reason": "Data historis tidak mencukupi"}

    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    d["EMA200"] = ema(d["Close"], 200)
    d["RSI14"] = rsi(d["Close"], 14)
    d["MACD"], d["MACD_SIGNAL"], d["MACD_HIST"] = macd(d["Close"])
    d["ATR14"] = atr(d, 14)
    d["ADX14"] = adx(d, 14)
    d["BB_MID"], d["BB_UPPER"], d["BB_LOWER"] = bollinger(d["Close"], 20, 2.0)
    d["VOL_SMA20"] = d["Volume"].rolling(20).mean()
    d["REL_VOL"] = d["Volume"] / d["VOL_SMA20"]
    d["VPT"] = (d["Volume"] * d["Close"].pct_change()).cumsum()
    d["OBV"] = obv(d)
    d["OBV_SMA10"] = d["OBV"].rolling(10).mean()
    d["OBV_SLOPE10"] = d["OBV"] - d["OBV"].shift(10)
    d["CMF20"] = chaikin_money_flow(d, 20)
    d["MFI14"] = money_flow_index(d, 14)
    d["STOCH_K"], d["STOCH_D"] = stochastic_oscillator(d, 14, 3, 3)
    d["CCI20"] = cci(d, 20)
    d["ROC12"] = rate_of_change(d["Close"], 12)

    # OBV slope is used later in scoring, so define it before any score calculations.
    obv_slope = float(d["OBV_SLOPE10"].iloc[-1]) if len(d) > 0 and pd.notna(d["OBV_SLOPE10"].iloc[-1]) else 0.0

    d = detect_reversal_signals(d)
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=required_cols).copy()
    if len(d) < max(50, int(min_history_bars * 0.7)):
        return {"valid": False, "reason": "Kebocoran data setelah filtering data inti"}

    last = d.iloc[-1]
    prev = d.iloc[-2]
    dominant_period, time_to_next_bottom, cycle_ok, cycle_info = compute_cycle_features(d["Close"])
    phase_info = classify_8_phase(d)

    adx_last = float(last["ADX14"]) if pd.notna(last["ADX14"]) else np.nan
    time_to_next_top = int(cycle_info.get("time_to_next_top", max(1, dominant_period // 2)))
    phase_age_bars = int(cycle_info.get("phase_age_bars", 0)) if pd.notna(cycle_info.get("phase_age_bars", np.nan)) else np.nan
    phase_age_pct = float(cycle_info.get("phase_age_pct", np.nan)) if pd.notna(cycle_info.get("phase_age_pct", np.nan)) else np.nan
    cycle_reliability = float(cycle_info.get("cycle_reliability", np.nan)) if pd.notna(cycle_info.get("cycle_reliability", np.nan)) else np.nan

    cycle_gate_reason = []
    if phase_info.get("phase") == "Markdown":
        cycle_ok = False
        cycle_gate_reason.append("phase Markdown")
    if np.isfinite(adx_last) and adx_last > 35:
        cycle_ok = False
        cycle_gate_reason.append(f"ADX {adx_last:.0f} > 35")
    if np.isfinite(cycle_reliability) and cycle_reliability < 45:
        cycle_ok = False
        cycle_gate_reason.append(f"CycleRel {cycle_reliability:.0f} < 45")
    cycle_info["cycle_gate_reason"] = ", ".join(cycle_gate_reason) if cycle_gate_reason else "OK"

    macro_context = macro_context or {}
    macro_symbol = str(macro_context.get("benchmark_symbol", "^JKSE"))
    macro_phase = str(macro_context.get("macro_phase", "Unknown"))
    macro_score = float(macro_context.get("macro_score", np.nan)) if pd.notna(macro_context.get("macro_score", np.nan)) else np.nan
    macro_gate_ok = bool(macro_context.get("macro_gate_ok", True))
    macro_gate_reason = str(macro_context.get("macro_gate_reason", "OK"))
    macro_multiplier = float(macro_context.get("macro_multiplier", 1.0)) if pd.notna(macro_context.get("macro_multiplier", 1.0)) else 1.0
    macro_cycle_reliability = float(macro_context.get("macro_cycle_reliability", np.nan)) if pd.notna(macro_context.get("macro_cycle_reliability", np.nan)) else np.nan
    macro_time_to_bottom = macro_context.get("macro_time_to_bottom", np.nan)
    macro_time_to_top = macro_context.get("macro_time_to_top", np.nan)
    macro_phase_age_bars = macro_context.get("macro_phase_age_bars", np.nan)
    macro_phase_age_pct = macro_context.get("macro_phase_age_pct", np.nan)
    market_regime = str(macro_context.get("market_regime", "SIDEWAYS")).upper()
    market_regime_confidence = float(macro_context.get("market_regime_confidence", 0.5)) if pd.notna(macro_context.get("market_regime_confidence", 0.5)) else 0.5
    market_regime_reason = str(macro_context.get("market_regime_reason", "Derived from benchmark"))
    regime_profile = _market_regime_profile(market_regime)

    reversal_names = [
        "Bullish_Engulfing",
        "Hammer",
        "Inverted_Hammer",
        "Morning_Star",
        "EMA20_Reclaim",
        "MACD_Bull_Cross",
        "RSI_Bounce",
        "Breakout_5D",
    ]
    reversal_score = 0
    reversal_hits = []
    for name in reversal_names:
        if bool(d[name].tail(5).any()):
            reversal_score += 1
            reversal_hits.append(name)

    unicorn_setup_confirmed = bool(d["Unicorn_Setup"].tail(8).any())
    unicorn_sniper_confirmed = bool(d["Unicorn_Sniper"].tail(8).any())
    fvg_state = _evaluate_latest_bullish_fvg(d)
    unicorn_state = _evaluate_latest_unicorn_setup(d)
    smc_confirmed = unicorn_setup_confirmed or unicorn_sniper_confirmed

    # User-facing statuses: only a confirmed Unicorn setup can become an actionable entry.
    # Sniper remains a supporting confirmation, not a standalone entry trigger.
    unicorn_setup_state = str(unicorn_state.get("setup_status", "none"))
    unicorn_sniper_state = str(unicorn_state.get("sniper_status", "none"))
    unicorn_setup_status = (
        "ENTRY" if (unicorn_state.get("setup_valid", False) and unicorn_setup_confirmed)
        else ("WATCHLIST" if unicorn_state.get("setup_valid", False) else "INVALID")
    )
    unicorn_sniper_status = (
        "ENTRY" if (unicorn_state.get("sniper_valid", False) and unicorn_sniper_confirmed)
        else ("WATCHLIST" if unicorn_state.get("sniper_valid", False) else "INVALID")
    )

    smc_points = 0
    smc_points += 4 * int(d["Bullish_FVG"].tail(5).any())
    smc_points += 4 * int(d["Bullish_OB"].tail(5).any())
    smc_points += 6 * int(unicorn_setup_confirmed)
    smc_points += 8 * int(unicorn_sniper_confirmed)
    smc_points += 2 * int(len(d) >= 5 and float(last["Close"]) > float(d["Close"].iloc[-5])) if len(d) >= 5 else 0

    trend_points = 0
    trend_points += int(last["Close"] > last["EMA20"])
    trend_points += int(last["EMA20"] > last["EMA50"])
    trend_points += int(last["EMA50"] > last["EMA200"])
    trend_points += int(last["EMA50"] > prev["EMA50"])

    momentum_points = 0
    momentum_points += int(50 <= float(last["RSI14"]) <= 72)
    momentum_points += int(float(last["MACD_HIST"]) > 0)
    momentum_points += int(last["Close"] > last["BB_MID"])
    momentum_points += int(float(last["ADX14"]) >= 18)

    reversal_points = 0
    reversal_points += int(d["EMA20_Reclaim"].tail(5).any())
    reversal_points += int(d["MACD_Bull_Cross"].tail(5).any())
    reversal_points += int(d["RSI_Bounce"].tail(5).any())
    reversal_points += int(d["Breakout_5D"].tail(5).any())

    cmf_last = float(last["CMF20"]) if pd.notna(last["CMF20"]) else 0.0
    mfi_last = float(last["MFI14"]) if pd.notna(last["MFI14"]) else 50.0
    stoch_k_last = float(last["STOCH_K"]) if pd.notna(last["STOCH_K"]) else 50.0
    stoch_d_last = float(last["STOCH_D"]) if pd.notna(last["STOCH_D"]) else 50.0

    # ---------------------------------------------------------------------
    # Refactored scoring model
    # - market_structure_score measures structure + momentum + reversal.
    # - smart_money_score focuses on participation / absorption / volume.
    # This reduces double counting and makes ranking more stable.
    # ---------------------------------------------------------------------
    cmf_score = _score_bucket(float(last.get("CMF20", np.nan)), -0.15, 0.25)
    rel_vol_score = _score_bucket(float(last.get("REL_VOL", np.nan)), 0.75, 2.25)
    obv_slope_score = 72.0 if obv_slope > 0 else 28.0 if obv_slope < 0 else 50.0
    mfi_score = _neutral_mid_score(float(last.get("MFI14", np.nan)), center=55.0, width=25.0)

    # Participation / accumulation proxy, intentionally not using EMA stack again.
    smart_money_score = float(np.clip(
        (cmf_score * 0.34)
        + (rel_vol_score * 0.26)
        + (obv_slope_score * 0.25)
        + (mfi_score * 0.15),
        0.0,
        100.0,
    ))

    # Convert discrete point buckets into 0-100 scores.
    trend_score = float(np.clip((trend_points / 4.0) * 100.0, 0.0, 100.0))
    momentum_score = float(np.clip((momentum_points / 4.0) * 100.0, 0.0, 100.0))
    smc_score = float(np.clip((smc_points / 24.0) * 100.0, 0.0, 100.0))
    reversal_score_pct = float(np.clip((reversal_points / 4.0) * 100.0, 0.0, 100.0))

    # Normalize point-based sub-scores before building the structural score.
    trend_score = float(np.clip((trend_points / 4.0) * 100.0, 0.0, 100.0))
    momentum_score = float(np.clip((momentum_points / 4.0) * 100.0, 0.0, 100.0))
    smc_score = float(np.clip((smc_points / 24.0) * 100.0, 0.0, 100.0))
    reversal_score_pct = float(np.clip((reversal_points / 4.0) * 100.0, 0.0, 100.0))

    # Structural score.  This remains the backbone of the model.
    core_score = float(np.clip(
        (trend_score * 0.35)
        + (momentum_score * 0.25)
        + (smc_score * 0.25)
        + (reversal_score_pct * 0.15),
        0.0,
        100.0,
    ))
    market_structure_score = core_score

    # Relative-strength composite versus the benchmark improves ranking stability
    # in bull markets and prevents weak names from outranking true leaders.
    benchmark_df = macro_context.get("benchmark_df", pd.DataFrame()) if isinstance(macro_context, dict) else pd.DataFrame()
    rs_composite_score = 50.0
    rs_strength_21 = np.nan
    rs_strength_63 = np.nan
    rs_strength_126 = np.nan
    if isinstance(benchmark_df, pd.DataFrame) and not benchmark_df.empty and len(benchmark_df) >= 60 and "Close" in benchmark_df.columns:
        try:
            rs_line = compute_relative_strength(d["Close"], benchmark_df["Close"])
            rs_line = rs_line.replace([np.inf, -np.inf], np.nan).dropna()
            if len(rs_line) >= 30:
                rs_strength_21 = rs_line.pct_change(21).iloc[-1] if len(rs_line) > 21 else np.nan
                rs_strength_63 = rs_line.pct_change(63).iloc[-1] if len(rs_line) > 63 else np.nan
                rs_strength_126 = rs_line.pct_change(126).iloc[-1] if len(rs_line) > 126 else np.nan
                rs_scores = []
                if pd.notna(rs_strength_21):
                    rs_scores.append((_score_bucket(float(rs_strength_21), -0.08, 0.16), 0.30))
                if pd.notna(rs_strength_63):
                    rs_scores.append((_score_bucket(float(rs_strength_63), -0.12, 0.28), 0.40))
                if pd.notna(rs_strength_126):
                    rs_scores.append((_score_bucket(float(rs_strength_126), -0.16, 0.42), 0.30))
                if rs_scores:
                    num = sum(score * weight for score, weight in rs_scores)
                    den = sum(weight for _, weight in rs_scores)
                    rs_composite_score = float(np.clip(num / max(den, 1e-9), 0.0, 100.0))
        except Exception:
            rs_composite_score = 50.0

    # Tradeability proxy rewards setups with enough room to run versus risk.
    trade_stop_atr = 1.8
    trade_target_1_atr = 2.2
    trade_target_2_atr = 3.8
    atr_proxy = _safe_float(last.get("ATR14"), np.nan)
    close_proxy = float(last["Close"])
    ema20_proxy = _safe_float(last.get("EMA20"), np.nan)
    if not np.isfinite(atr_proxy) or atr_proxy <= 0:
        atr_proxy = max(close_proxy * 0.02, 1.0)
    atr_pct = float(atr_proxy / max(close_proxy, 1e-9))
    swing_low_proxy = float(d["Low"].tail(10).min())
    swing_high_proxy = float(d["High"].tail(20).max())
    support_proxy = ema20_proxy if np.isfinite(ema20_proxy) else swing_low_proxy
    avg_value_traded_20d = float((d["Close"].tail(20) * d["Volume"].tail(20)).mean())
    if not np.isfinite(avg_value_traded_20d):
        avg_value_traded_20d = 0.0
    gap_proxy_20d = np.nan
    try:
        prev_close_20 = d["Close"].shift(1).replace(0, np.nan)
        gap_proxy_20d = float((d["Open"] / prev_close_20 - 1.0).abs().tail(20).mean())
    except Exception:
        gap_proxy_20d = np.nan

    entry_proxy_candidates = [
        close_proxy,
        support_proxy,
        swing_low_proxy + atr_proxy * 0.25,
    ]
    entry_proxy_candidates = [v for v in entry_proxy_candidates if np.isfinite(v) and v > 0]
    entry_proxy = float(np.median(entry_proxy_candidates)) if entry_proxy_candidates else close_proxy
    stop_proxy = min(
        swing_low_proxy - atr_proxy * 0.15,
        close_proxy - atr_proxy * trade_stop_atr,
        support_proxy - atr_proxy * 0.35,
    )
    stop_proxy = max(min(stop_proxy, entry_proxy - atr_proxy * 0.75), 0.0)
    risk_proxy = max(entry_proxy - stop_proxy, atr_proxy * 0.60)
    target_1_proxy = max(
        entry_proxy + atr_proxy * trade_target_1_atr,
        swing_high_proxy,
    )
    target_2_proxy = max(
        entry_proxy + atr_proxy * trade_target_2_atr,
        target_1_proxy + atr_proxy * 1.0,
    )
    rr1_proxy = max(0.0, (target_1_proxy - entry_proxy) / max(risk_proxy, 1e-9))
    rr2_proxy = max(0.0, (target_2_proxy - entry_proxy) / max(risk_proxy, 1e-9))
    liquidity_value_score = _score_bucket(avg_value_traded_20d, 3.5e8, 5.0e10)
    volume_participation_score = _score_bucket(float(last.get("REL_VOL", np.nan)), 0.90, 2.60)
    volatility_stability_score = _score_bucket(atr_pct, 0.012, 0.090, invert=True)
    gap_stability_score = _score_bucket(gap_proxy_20d if np.isfinite(gap_proxy_20d) else 0.0, 0.008, 0.050, invert=True)
    rr_tradeability_score = float(np.clip(
        (_score_bucket(rr1_proxy, 0.9, 2.6) * 0.55)
        + (_score_bucket(rr2_proxy, 1.4, 4.8) * 0.30)
        + (_score_bucket(float((close_proxy - support_proxy) / max(atr_proxy, 1e-9)), -0.75, 1.50) * 0.15),
        0.0,
        100.0,
    ))
    tradeability_score = float(np.clip(
        (liquidity_value_score * 0.22)
        + (volume_participation_score * 0.18)
        + (volatility_stability_score * 0.18)
        + (gap_stability_score * 0.12)
        + (rr_tradeability_score * 0.30),
        0.0,
        100.0,
    ))

    # Final score is intentionally simplified for a profit-first scanner.
    # Structural quality and relative strength carry the most weight.
    final_score = float(np.clip(
        (market_structure_score * 0.55)
        + (smart_money_score * 0.18)
        + (rs_composite_score * 0.17)
        + (tradeability_score * 0.10),
        0.0,
        100.0,
    ))

    # Macro remains visible in notes and decisioning, but it no longer crushes the ranking score.
    if np.isfinite(macro_multiplier):
        macro_overlay = 0.0
        if not macro_gate_ok:
            macro_overlay = -4.0 if market_regime == "BEAR" else -3.0 if market_regime == "SIDEWAYS" else -2.0
        elif market_regime == "BULL":
            macro_overlay = 1.5
        elif market_regime == "SIDEWAYS":
            macro_overlay = 0.5
        final_score = float(np.clip(final_score + macro_overlay, 0.0, 100.0))
    future_fundamental_score = np.nan
    future_fundamental_grade = "n/a"
    future_fundamental_direction = "n/a"
    future_fundamental_confidence = np.nan
    future_fundamental_phase = "Unknown"
    future_fundamental_reason = "n/a"
    if future_fundamental_context is not None:
        future_fundamental_score = _safe_float(future_fundamental_context.get("future_fundamental_score"), np.nan)
        future_fundamental_grade = str(future_fundamental_context.get("future_fundamental_grade", "n/a"))
        future_fundamental_direction = str(future_fundamental_context.get("future_fundamental_direction", "n/a"))
        future_fundamental_confidence = _safe_float(future_fundamental_context.get("future_fundamental_confidence"), np.nan)
        future_fundamental_phase = str(future_fundamental_context.get("future_phase", "Unknown"))
        future_fundamental_reason = str(future_fundamental_context.get("future_moat_reason", "n/a"))
        if np.isfinite(future_fundamental_score):
            future_weight = 0.10
            if np.isfinite(future_fundamental_confidence):
                future_weight = float(np.clip(0.08 + (future_fundamental_confidence / 100.0) * 0.08, 0.08, 0.16))
            final_score = float(np.clip((final_score * (1.0 - future_weight)) + (future_fundamental_score * future_weight), 0.0, 100.0))

    tradeability_profile = _tradeability_profile_from_stock_res({"df": d, "last": last, "market_regime": market_regime, "tradeability_score": tradeability_score, "tradeability_components": {"rr_tradeability_score": rr_tradeability_score}, "avg_value_traded_20d": avg_value_traded_20d})
    avg_value_traded_20d = float(tradeability_profile.get("avg_value_traded_20d", avg_value_traded_20d))
    spread_proxy_20d = float(tradeability_profile.get("spread_proxy_20d", np.nan))
    gap_proxy_20d = float(tradeability_profile.get("gap_proxy_20d", np.nan))
    tradeability_score = float(tradeability_profile.get("tradeability_score", tradeability_score))
    liquidity_ok = bool(tradeability_profile.get("liquidity_ok", False))
    tradeability_threshold = float(tradeability_profile.get("tradeability_threshold", 60.0))
    tradeability_value_floor = float(tradeability_profile.get("tradeability_value_floor", 1.0e9))
    spread_cap = float(tradeability_profile.get("tradeability_spread_cap", 0.055))
    gap_cap = float(tradeability_profile.get("tradeability_gap_cap", 0.055))
    tradeability_gate_ok = bool(
        liquidity_ok
        and np.isfinite(tradeability_score)
        and tradeability_score >= tradeability_threshold
        and avg_value_traded_20d >= tradeability_value_floor
        and (not np.isfinite(spread_proxy_20d) or spread_proxy_20d <= spread_cap)
        and (not np.isfinite(gap_proxy_20d) or gap_proxy_20d <= gap_cap)
        and (not np.isfinite(atr_pct) or atr_pct <= 0.16)
    )
    tradeability_gate_reason_bits = []
    if not liquidity_ok:
        tradeability_gate_reason_bits.append("Liquidity below threshold")
    if np.isfinite(tradeability_score) and tradeability_score < tradeability_threshold:
        tradeability_gate_reason_bits.append(
            f"Tradeability score {tradeability_score:.0f} < {tradeability_threshold:.0f}"
        )
    if avg_value_traded_20d < tradeability_value_floor:
        tradeability_gate_reason_bits.append(
            f"Avg value {avg_value_traded_20d/1e9:.2f}B < {tradeability_value_floor/1e9:.2f}B"
        )
    if np.isfinite(spread_proxy_20d) and spread_proxy_20d > spread_cap:
        tradeability_gate_reason_bits.append(f"Spread proxy {spread_proxy_20d:.1%} > {spread_cap:.1%}")
    if np.isfinite(gap_proxy_20d) and gap_proxy_20d > gap_cap:
        tradeability_gate_reason_bits.append(f"Gap proxy {gap_proxy_20d:.1%} > {gap_cap:.1%}")
    if np.isfinite(atr_pct) and atr_pct > 0.16:
        tradeability_gate_reason_bits.append(f"ATR% {atr_pct:.1%} too wide")
    tradeability_gate_reason = "OK" if tradeability_gate_ok else ", ".join(tradeability_gate_reason_bits) if tradeability_gate_reason_bits else "Tradeability gate off"

    ema20_v = _safe_float(last.get("EMA20"), np.nan)
    ema50_v = _safe_float(last.get("EMA50"), np.nan)
    ema200_v = _safe_float(last.get("EMA200"), np.nan)
    macd_hist_v = _safe_float(last.get("MACD_HIST"), np.nan)
    rsi_v = _safe_float(last.get("RSI14"), np.nan)

    trend_ok_strict = bool(
        np.isfinite(ema20_v)
        and np.isfinite(ema50_v)
        and np.isfinite(ema200_v)
        and (float(last["Close"]) > ema20_v)
        and (ema20_v > ema50_v)
        and (ema50_v > ema200_v)
    )
    trend_ok_regime = bool(
        np.isfinite(ema20_v)
        and (float(last["Close"]) > ema20_v)
        and (
            (
                market_regime == "BEAR"
                and (
                    (np.isfinite(ema50_v) and ema20_v > ema50_v)
                    or (np.isfinite(macd_hist_v) and macd_hist_v > 0)
                    or (np.isfinite(rsi_v) and rsi_v >= regime_profile["trend_rsi_floor"])
                )
            )
            or (
                market_regime == "SIDEWAYS"
                and (
                    (np.isfinite(ema50_v) and ema20_v > ema50_v)
                    or (np.isfinite(macd_hist_v) and macd_hist_v > 0)
                    or (np.isfinite(rsi_v) and rsi_v >= regime_profile["trend_rsi_floor"])
                )
            )
            or (
                market_regime == "BULL"
                and (
                    (np.isfinite(ema50_v) and ema20_v > ema50_v)
                    or (np.isfinite(ema50_v) and np.isfinite(ema200_v) and ema50_v > ema200_v)
                    or (np.isfinite(macd_hist_v) and macd_hist_v > 0)
                    or (np.isfinite(rsi_v) and rsi_v >= regime_profile["trend_rsi_floor"])
                )
            )
        )
    )
    weekly_mtf = _weekly_mtf_confirmation(d)
    weekly_mtf_ok = bool(weekly_mtf.get("weekly_mtf_ok", False))
    weekly_mtf_score = _safe_float(weekly_mtf.get("weekly_mtf_score", np.nan), np.nan)
    weekly_mtf_reason = str(weekly_mtf.get("weekly_mtf_reason", "n/a"))

    trend_ok_soft = trend_ok_regime
    trend_ok = bool(trend_ok_regime and weekly_mtf_ok)

    if mode == "Conservative":
        buy_threshold, strong_threshold = 84, 92
    elif mode == "Balanced":
        buy_threshold, strong_threshold = regime_profile["buy_threshold"], regime_profile["strong_threshold"]
    else:
        buy_threshold, strong_threshold = regime_profile["buy_threshold"] - 4.0, regime_profile["strong_threshold"] - 4.0

    regime_score_floor = float(regime_profile["macro_score_floor"])
    score_buffer = float(regime_profile["score_buffer"])

    quality_gate_ok = bool(
        weekly_mtf_ok
        and (
            (market_regime == "BEAR" and market_structure_score >= 66 and smart_money_score >= 64 and rs_composite_score >= 58 and tradeability_score >= 60)
            or (market_regime == "SIDEWAYS" and market_structure_score >= 64 and smart_money_score >= 62 and rs_composite_score >= 60 and tradeability_score >= 62)
            or (market_regime == "BULL" and market_structure_score >= 62 and smart_money_score >= 60 and rs_composite_score >= 62 and tradeability_score >= 64)
        )
    )

    quality_bonus = 0.0
    if quality_gate_ok:
        quality_bonus += 1.0 if market_regime == "BEAR" else 1.4 if market_regime == "SIDEWAYS" else 1.6
    if rs_composite_score >= 60:
        quality_bonus += 1.0
    if tradeability_score >= 60:
        quality_bonus += 0.75
    if smc_confirmed and trend_ok_regime:
        quality_bonus += 0.5
    quality_bonus = float(np.clip(quality_bonus, 0.0, 4.0))
    final_score = float(np.clip(final_score + quality_bonus, 0.0, 100.0))

    # Macro remains visible in notes and decisioning, but it no longer crushes the ranking score.
    if np.isfinite(macro_multiplier):
        macro_overlay = 0.0
        if not macro_gate_ok:
            macro_overlay = -4.0 if market_regime == "BEAR" else -3.0 if market_regime == "SIDEWAYS" else -2.0
        elif market_regime == "BULL":
            macro_overlay = 1.5
        elif market_regime == "SIDEWAYS":
            macro_overlay = 0.5
        final_score = float(np.clip(final_score + macro_overlay, 0.0, 100.0))

    score_support_ok = bool(np.isfinite(final_score) and final_score >= buy_threshold)
    macro_support_ok = bool(
        macro_gate_ok
        or (np.isfinite(macro_score) and macro_score >= regime_score_floor)
        or (final_score >= strong_threshold and tradeability_score >= 50)
    )
    quality_support_ok = bool(quality_gate_ok and final_score >= buy_threshold - 1.5)

    pivot_highs_confirmed = d.loc[d.get("Pivot_High_Confirmed", pd.Series(False, index=d.index)).fillna(False), "High"].dropna()
    breakout_reference = float(pivot_highs_confirmed.iloc[-1]) if len(pivot_highs_confirmed) else np.nan
    if not np.isfinite(breakout_reference) or breakout_reference <= 0:
        breakout_reference = float(d["High"].tail(20).max())

    atr_v = _safe_float(last.get("ATR14"), np.nan)
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(float(last["Close"]) * 0.02, 1.0) if np.isfinite(float(last["Close"])) and float(last["Close"]) > 0 else 1.0

    breakout_invalidation_level = float(max(0.0, breakout_reference - atr_v * 0.25))
    bullish_close_ok = bool(np.isfinite(last["Close"]) and np.isfinite(last["Open"]) and float(last["Close"]) > float(last["Open"]))
    breakout_retest_ok = bool(
        np.isfinite(breakout_reference)
        and np.isfinite(last["Low"])
        and float(last["Low"]) <= breakout_reference + atr_v * 0.20
        and float(last["Close"]) > breakout_reference
        and float(last["Close"]) > float(last["Open"])
    )
    breakout_price_ok = bool(
        breakout_retest_ok
        and (not np.isfinite(ema20_v) or float(last["Close"]) > ema20_v)
    )
    breakout_confirmed = bool(trend_ok_soft and breakout_price_ok)
    breakout_setup_valid = bool(
        breakout_confirmed
        and bullish_close_ok
        and float(last["Close"]) > breakout_invalidation_level
    )
    breakout_setup_status = "ENTRY" if breakout_setup_valid else ("WATCHLIST" if breakout_confirmed else "INVALID")

    pullback_state = _evaluate_latest_pullback_continuation(d)
    pullback_continuation_valid = bool(pullback_state.get("valid", False))
    pullback_continuation_status = str(pullback_state.get("status", "INVALID")).upper()
    pullback_continuation_reference = _safe_float(pullback_state.get("support_anchor"), np.nan)
    pullback_continuation_invalidation = _safe_float(pullback_state.get("invalidation_level"), np.nan)

    reversal_accumulation_state = _evaluate_latest_reversal_accumulation_setup(d)
    reversal_accumulation_valid = bool(reversal_accumulation_state.get("valid", False))
    reversal_accumulation_status = str(reversal_accumulation_state.get("status", "INVALID")).upper()
    reversal_accumulation_reference = _safe_float(reversal_accumulation_state.get("support_anchor"), np.nan)
    reversal_accumulation_invalidation = _safe_float(reversal_accumulation_state.get("invalidation_level"), np.nan)

    setup_valid_any = bool(
        breakout_setup_valid
        or bool(unicorn_state.get("setup_valid", False))
        or bool(unicorn_state.get("sniper_valid", False))
        or pullback_continuation_valid
        or reversal_accumulation_valid
    )

    actionable_entry = bool(
        setup_valid_any
        and tradeability_gate_ok
    )


    soft_entry_candidate = bool(
        setup_valid_any
        and (
            _status_has_entry(unicorn_setup_status)
            or _status_has_entry(unicorn_sniper_status)
            or breakout_setup_valid
            or breakout_confirmed
            or pullback_continuation_valid
            or reversal_accumulation_valid
            or smc_confirmed
        )
    )

    if setup_valid_any:
        if liquidity_ok and np.isfinite(final_score) and final_score >= strong_threshold:
            decision = "STRONG BUY"
        elif liquidity_ok and np.isfinite(final_score) and final_score >= buy_threshold:
            decision = "BUY"
        else:
            decision = "WATCHLIST"
    else:
        decision = "AVOID"
    recent_swing_low = float(d["Low"].tail(10).min())
    recent_support_ema = float(d["EMA20"].iloc[-1])
    ob_zone = np.nan
    ob_rows = d[d["Bullish_OB"]].tail(3)
    if not ob_rows.empty:
        ob_idx = ob_rows.index[-1]
        loc = d.index.get_loc(ob_idx)
        if loc >= 1:
            ob_zone = float((d["Low"].iloc[loc - 1] + d["High"].iloc[loc - 1]) / 2)


    unicorn_zone_rows = d[d["Unicorn_Setup"]].tail(3)
    if not unicorn_zone_rows.empty:
        u_idx = unicorn_zone_rows.index[-1]
        u_loc = d.index.get_loc(u_idx)
        unicorn_fvg_top = float(d["FVG_Top"].iloc[u_loc]) if pd.notna(d["FVG_Top"].iloc[u_loc]) else np.nan
        unicorn_fvg_bottom = float(d["FVG_Bottom"].iloc[u_loc]) if pd.notna(d["FVG_Bottom"].iloc[u_loc]) else np.nan
        unicorn_breaker_top = float(d["Breaker_Top"].iloc[u_loc]) if pd.notna(d["Breaker_Top"].iloc[u_loc]) else np.nan
        unicorn_breaker_bottom = float(d["Breaker_Bottom"].iloc[u_loc]) if pd.notna(d["Breaker_Bottom"].iloc[u_loc]) else np.nan
        unicorn_sweep_low = float(d["Liquidity_Sweep_Low"].iloc[u_loc]) if pd.notna(d["Liquidity_Sweep_Low"].iloc[u_loc]) else np.nan
    else:
        unicorn_fvg_top = np.nan
        unicorn_fvg_bottom = np.nan
        unicorn_breaker_top = np.nan
        unicorn_breaker_bottom = np.nan
        unicorn_sweep_low = np.nan

    entry_plan = _build_consistent_entry_plan(
        {
            "df": d,
            "last": last,
            "decision": decision,
            "score": final_score,
            "trend_ok": trend_ok,
            "tradeability_gate_ok": tradeability_gate_ok,
            "tradeability_gate_reason": tradeability_gate_reason,
            "avg_value_traded_20d": avg_value_traded_20d,
            "atr_pct": atr_pct,
            "trend_ok_regime": trend_ok_regime,
            "quality_gate_ok": quality_gate_ok,
            "quality_support_ok": quality_support_ok,
            "(np.isfinite(final_score) and final_score >= buy_threshold)": (np.isfinite(final_score) and final_score >= buy_threshold),
            "market_regime": market_regime,
            "macro_gate_ok": macro_gate_ok,
            "macro_score": macro_score,
            "tradeability_score": tradeability_score,
            "rs_composite_score": rs_composite_score,
            "smart_money_score": smart_money_score,
            "market_structure_score": market_structure_score,
            "regime_buy_threshold": buy_threshold,
            "regime_strong_threshold": strong_threshold,
            "unicorn_setup_status": unicorn_setup_status,
            "unicorn_sniper_status": unicorn_sniper_status,
            "unicorn_setup_confirmed": unicorn_setup_confirmed,
            "unicorn_sniper_confirmed": unicorn_sniper_confirmed,
            "unicorn_fvg_top": unicorn_fvg_top,
            "unicorn_fvg_bottom": unicorn_fvg_bottom,
            "unicorn_breaker_top": unicorn_breaker_top,
            "unicorn_breaker_bottom": unicorn_breaker_bottom,
            "unicorn_sweep_low": unicorn_sweep_low,
            "reversal_accumulation_status": reversal_accumulation_status,
            "reversal_accumulation_valid": reversal_accumulation_valid,
            "reversal_accumulation_reference": reversal_accumulation_reference,
            "reversal_accumulation_invalidation": reversal_accumulation_invalidation,
            "reversal_accumulation_state": reversal_accumulation_state,
        }
    )
    entry_price = _safe_float(entry_plan.get("entry_price_plan"), np.nan)
    stop_price = _safe_float(entry_plan.get("stop_loss_plan"), np.nan)
    entry_zone_low = _safe_float(entry_plan.get("entry_zone_low"), np.nan)
    entry_zone_high = _safe_float(entry_plan.get("entry_zone_high"), np.nan)
    entry_trigger = str(entry_plan.get("entry_trigger", "No_signal"))
    breakout_confirmed = bool(entry_plan.get("breakout_confirmed", False))
    breakout_reference = _safe_float(entry_plan.get("breakout_reference"), np.nan)
    setup_kind = str(entry_plan.get("setup_kind", "None"))

    obv_slope = float(last["OBV_SLOPE10"]) if pd.notna(last["OBV_SLOPE10"]) else np.nan
    if pd.isna(obv_slope):
        obv_trend = "Flat"
    elif obv_slope > 0:
        obv_trend = "Rising"
    elif obv_slope < 0:
        obv_trend = "Falling"
    else:
        obv_trend = "Flat"

    notes = []
    if not liquidity_ok:
        notes.append("Filter_Likuiditas_Gagal")
    if not tradeability_gate_ok:
        notes.append("Tradeability_Gated_" + tradeability_gate_reason.replace(" ", "_"))
    if not trend_ok:
        notes.append("Struktur_Trend_Bearish")
    if not weekly_mtf_ok:
        notes.append("Weekly_MTF_Belum_Konfirm")
    if not any([
        _status_has_entry(unicorn_setup_status),
        _status_has_entry(unicorn_sniper_status),
        _status_has_entry(str(breakout_setup_status)),
        _status_has_entry(str(pullback_continuation_status)),
        _status_has_entry(str(reversal_accumulation_status)),
    ]):
        notes.append("Setup_Belum_Entry")
    if _status_has_entry(unicorn_setup_status) and not _status_has_entry(unicorn_sniper_status):
        notes.append("Belum_Sniper")
    if _status_has_entry(unicorn_sniper_status) and not _status_has_entry(unicorn_setup_status):
        notes.append("Sniper_Only")
    if _status_has_entry(str(reversal_accumulation_status)):
        notes.append("Reversal_Accumulation_Ready")
    if not cycle_ok:
        notes.append("Siklus_Belum_Menguat")
        if cycle_gate_reason:
            notes.append("Cycle_Gated_" + "_".join(cycle_gate_reason).replace(" ", ""))
    if not macro_gate_ok:
        notes.append("Macro_Gated_" + macro_gate_reason.replace(" ", "_"))
    if reversal_score == 0:
        notes.append("Belum_Ada_Reversal_Strong")
    if tradeability_score < 45:
        notes.append("RR_Kurang_Menarik")
    if breakout_confirmed:
        notes.append("Breakout_Confirmed")
    if breakout_setup_valid:
        notes.append("Breakout_Entry_Ready")
    if pullback_continuation_valid:
        notes.append("Pullback_Continuation_Ready")
    if rs_composite_score >= 70:
        notes.append("RS_Kuat")

    # Keep score views aligned with the refactor.
    trend_score = float(np.clip(trend_score, 0.0, 100.0))
    momentum_score = float(np.clip(momentum_score, 0.0, 100.0))
    smc_score = float(np.clip(smc_score, 0.0, 100.0))
    reversal_score_pct = float(np.clip(reversal_score_pct, 0.0, 100.0))
    market_structure_score = float(np.clip(market_structure_score, 0.0, 100.0))
    core_score = market_structure_score
    risk_score = float(np.clip(
        100.0
        - (18.0 if not liquidity_ok else 0.0)
        - (18.0 if not tradeability_gate_ok else 0.0)
        - (20.0 if not trend_ok else 0.0)
        - (15.0 if not smc_confirmed else 0.0)
        - (12.0 if not cycle_ok else 0.0)
        - (10.0 if not macro_gate_ok else 0.0),
        0.0,
        100.0,
    ))

    return {
        "valid": True,
        "symbol": None,
        "decision": decision,
        "score": float(final_score),
        "core_score": float(core_score),
        "market_structure_score": float(market_structure_score),
        "rs_composite_score": float(rs_composite_score),
        "tradeability_score": float(tradeability_score),
        "tradeability_gate_ok": bool(tradeability_gate_ok),
        "tradeability_gate_reason": tradeability_gate_reason,
        "tradeability_tier": str(tradeability_profile.get("tradeability_tier", "Watch")),
        "avg_value_traded_20d": float(avg_value_traded_20d),
        "spread_proxy_20d": float(spread_proxy_20d) if np.isfinite(spread_proxy_20d) else np.nan,
        "gap_proxy_20d": float(gap_proxy_20d) if np.isfinite(gap_proxy_20d) else np.nan,
        "atr_pct": float(atr_pct),
        "tradeability_components": {
            "liquidity_value_score": float(liquidity_value_score),
            "volume_participation_score": float(volume_participation_score),
            "volatility_stability_score": float(volatility_stability_score),
            "gap_stability_score": float(gap_stability_score),
            "rr_tradeability_score": float(rr_tradeability_score),
        },
        "trend_score": float(trend_score),
        "momentum_score": float(momentum_score),
        "smc_score": float(smc_score),
        "reversal_score_pct": float(reversal_score_pct),
        "risk_score": float(risk_score),
        "close": float(last["Close"]),
        "rsi": float(last["RSI14"]),
        "adx": float(last["ADX14"]) if pd.notna(last["ADX14"]) else np.nan,
        "rel_vol": float(last["REL_VOL"]) if pd.notna(last["REL_VOL"]) else np.nan,
        "smart_money_score": float(smart_money_score),
        "cmf20": float(last["CMF20"]) if pd.notna(last["CMF20"]) else np.nan,
        "mfi14": float(last["MFI14"]) if pd.notna(last["MFI14"]) else np.nan,
        "stoch_k": float(last["STOCH_K"]) if pd.notna(last["STOCH_K"]) else np.nan,
        "stoch_d": float(last["STOCH_D"]) if pd.notna(last["STOCH_D"]) else np.nan,
        "cci20": float(last["CCI20"]) if pd.notna(last["CCI20"]) else np.nan,
        "roc12": float(last["ROC12"]) if pd.notna(last["ROC12"]) else np.nan,
        "dominant_period": int(dominant_period),
        "time_to_bottom": int(time_to_next_bottom),
        "time_to_top": int(time_to_next_top),
        "phase_age_bars": phase_age_bars,
        "phase_age_pct": phase_age_pct,
        "cycle_reliability": cycle_reliability,
        "cycle_gate_reason": cycle_info.get("cycle_gate_reason", ""),
        "cycle_info": cycle_info,
        "macro_symbol": macro_symbol,
        "macro_phase": macro_phase,
        "macro_score": macro_score,
        "macro_gate_ok": macro_gate_ok,
        "macro_gate_reason": macro_gate_reason,
        "macro_multiplier": macro_multiplier,
        "macro_cycle_reliability": macro_cycle_reliability,
        "market_regime": market_regime,
        "market_regime_confidence": market_regime_confidence,
        "market_regime_reason": market_regime_reason,
        "regime_buy_threshold": float(buy_threshold),
        "regime_strong_threshold": float(strong_threshold),
        "macro_time_to_bottom": macro_time_to_bottom,
        "macro_time_to_top": macro_time_to_top,
        "macro_phase_age_bars": macro_phase_age_bars,
        "macro_phase_age_pct": macro_phase_age_pct,
        "future_fundamental_score": float(future_fundamental_score) if pd.notna(future_fundamental_score) else np.nan,
        "future_fundamental_grade": future_fundamental_grade,
        "future_fundamental_direction": future_fundamental_direction,
        "future_fundamental_confidence": float(future_fundamental_confidence) if pd.notna(future_fundamental_confidence) else np.nan,
        "future_fundamental_phase": future_fundamental_phase,
        "future_fundamental_reason": future_fundamental_reason,
        "phase": phase_info["phase"],
        "phase_confidence": float(phase_info["phase_confidence"]),
        "phase_rank": float(phase_info["phase_rank"]),
        "phase_reason": phase_info["phase_reason"],
        "phase_scores": phase_info["phase_scores"],
        "liquidity_ok": liquidity_ok,
        "trend_ok": trend_ok,
        "trend_ok_regime": trend_ok_regime,
        "weekly_mtf_ok": bool(weekly_mtf_ok),
        "weekly_mtf_score": float(weekly_mtf_score) if np.isfinite(weekly_mtf_score) else np.nan,
        "weekly_mtf_reason": weekly_mtf_reason,
        "quality_gate_ok": quality_gate_ok,
        "unicorn_setup": unicorn_setup_confirmed,
        "unicorn_sniper": unicorn_sniper_confirmed,
        "unicorn_entry_style": (
            "Pullback+Continuation"
            if pullback_continuation_valid
            else (
                "Breakout"
                if breakout_setup_valid
                else (
                    "Unicorn+Sniper"
                    if _status_has_entry(unicorn_sniper_status)
                    else (
                        "Unicorn"
                        if _status_has_entry(unicorn_setup_status)
                        else (
                            "Reversal+Accumulation"
                            if reversal_accumulation_valid
                            else ("Watchlist" if unicorn_setup_status == "WATCHLIST" else "None")
                        )
                    )
                )
            )
        ),
        "fvg_present": bool(d["Bullish_FVG"].tail(5).any()),
        "fvg_age_bars": fvg_state.get("age_bars", np.nan),
        "fvg_age_days": fvg_state.get("age_days", np.nan),
        "fvg_status": fvg_state.get("status", "none"),
        "fvg_fresh": fvg_state.get("fresh", False),
        "fvg_valid": fvg_state.get("valid", False),
        "fvg_mitigated": fvg_state.get("mitigated", False),
        "fvg_top": fvg_state.get("top", np.nan),
        "fvg_bottom": fvg_state.get("bottom", np.nan),
        "ob_present": bool(d["Bullish_OB"].tail(5).any()),
        "unicorn_setup_valid": unicorn_state.get("setup_valid", False),
        "unicorn_setup_state": unicorn_setup_state,
        "unicorn_setup_status": unicorn_setup_status,
        "unicorn_setup_age_bars": unicorn_state.get("age_bars", np.nan),
        "unicorn_setup_age_days": unicorn_state.get("age_days", np.nan),
        "unicorn_setup_fresh": unicorn_state.get("setup_fresh", False),
        "unicorn_sniper_valid": unicorn_state.get("sniper_valid", False),
        "unicorn_sniper_state": unicorn_sniper_state,
        "unicorn_sniper_status": unicorn_sniper_status,
        "unicorn_setup_reason": unicorn_state.get("reason", "n/a"),
        "reversal_score": int(reversal_score),
        "reversal_hits": ", ".join(reversal_hits) if reversal_hits else "-",
"obv_trend": obv_trend,
"obv_slope10": obv_slope,
"entry_zone_low": entry_zone_low,
"entry_zone_high": entry_zone_high,
"entry_trigger": entry_trigger,
"entry_price": entry_price,
"entry_price_plan": entry_price,
"stop_price": stop_price,
"stop_loss_plan": stop_price,
"setup_distance_to_entry_atr": entry_plan.get("setup_distance_to_entry_atr", np.nan),
"setup_fill_probability": entry_plan.get("setup_fill_probability", np.nan),
"breakout_confirmed": breakout_confirmed,
"breakout_setup_valid": breakout_setup_valid,
"breakout_setup_status": breakout_setup_status,
"breakout_reference": breakout_reference,
"breakout_invalidation_level": breakout_invalidation_level,
"pullback_continuation_valid": pullback_continuation_valid,
"pullback_continuation_status": pullback_continuation_status,
"pullback_continuation_reference": pullback_continuation_reference,
"pullback_continuation_invalidation": pullback_continuation_invalidation,
"pullback_continuation_state": pullback_state,
"setup_kind": setup_kind,
"setup_variant": entry_plan.get("setup_variant", "None"),
"entry_plan": entry_plan,
"unicorn_setup_confirmed": unicorn_setup_confirmed,
"unicorn_sniper_confirmed": unicorn_sniper_confirmed,
"unicorn_fvg_top": unicorn_fvg_top,
"unicorn_fvg_bottom": unicorn_fvg_bottom,
"unicorn_breaker_top": unicorn_breaker_top,
"unicorn_breaker_bottom": unicorn_breaker_bottom,
"unicorn_sweep_low": unicorn_sweep_low,
        "reversal_accumulation_valid": reversal_accumulation_valid,
        "reversal_accumulation_status": reversal_accumulation_status,
        "reversal_accumulation_reference": reversal_accumulation_reference,
        "reversal_accumulation_invalidation": reversal_accumulation_invalidation,
        "reversal_accumulation_state": reversal_accumulation_state,
        "notes": ",".join(notes) if notes else "SMC_Structure_Clear",
        "df": d,
        "last": last,
    }




def _tradeability_profile_from_stock_res(stock_res: dict) -> dict:
    """Compute a more conservative tradeability snapshot for IDX execution."""
    d = stock_res.get("df")
    last = stock_res.get("last")
    if not isinstance(d, pd.DataFrame) or d.empty or last is None:
        return {
            "avg_value_traded_20d": np.nan,
            "spread_proxy_20d": np.nan,
            "gap_proxy_20d": np.nan,
            "atr_pct": np.nan,
            "tradeability_score": np.nan,
            "tradeability_tier": "n/a",
            "tradeability_gate_ok": False,
            "tradeability_gate_reason": "Insufficient data",
            "liquidity_ok": False,
        }

    close = _safe_float(last.get("Close"), np.nan)
    atr_v = _safe_float(last.get("ATR14"), np.nan)
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close * 0.02, 1.0) if np.isfinite(close) and close > 0 else 1.0
    atr_pct = float(atr_v / max(close, 1e-9)) if np.isfinite(close) and close > 0 else np.nan

    try:
        avg_value_traded_20d = _safe_float(stock_res.get("avg_value_traded_20d"), np.nan)
        if not np.isfinite(avg_value_traded_20d):
            avg_value_traded_20d = float((d["Close"].tail(20) * d["Volume"].tail(20)).mean())
    except Exception:
        avg_value_traded_20d = np.nan
    if not np.isfinite(avg_value_traded_20d):
        avg_value_traded_20d = 0.0

    try:
        range_proxy_20d = float(((d["High"] - d["Low"]) / d["Close"].replace(0, np.nan)).tail(20).median())
    except Exception:
        range_proxy_20d = np.nan
    try:
        atr_proxy_20d = float(((d["ATR14"] / d["Close"].replace(0, np.nan)).tail(20)).median())
    except Exception:
        atr_proxy_20d = np.nan
    try:
        if np.isfinite(range_proxy_20d) and np.isfinite(atr_proxy_20d):
            spread_proxy_20d = float(min(range_proxy_20d, atr_proxy_20d * 0.85))
        elif np.isfinite(range_proxy_20d):
            spread_proxy_20d = float(range_proxy_20d)
        elif np.isfinite(atr_proxy_20d):
            spread_proxy_20d = float(atr_proxy_20d * 0.85)
        else:
            spread_proxy_20d = np.nan
    except Exception:
        spread_proxy_20d = np.nan
    try:
        prev_close = d["Close"].shift(1).replace(0, np.nan)
        gap_proxy_20d = float((d["Open"] / prev_close - 1.0).abs().tail(20).mean())
    except Exception:
        gap_proxy_20d = np.nan

    market_regime = str(stock_res.get("market_regime", "SIDEWAYS") or "SIDEWAYS").upper()
    # Real-money live execution gate: bias toward liquid, clean, low-friction names.
    value_floor = 1.40e9 if market_regime == "BEAR" else (1.10e9 if market_regime == "SIDEWAYS" else 8.00e8)
    spread_cap = 0.075 if market_regime == "BEAR" else (0.085 if market_regime == "SIDEWAYS" else 0.095)
    gap_cap = 0.045 if market_regime == "BEAR" else (0.050 if market_regime == "SIDEWAYS" else 0.055)
    atr_cap = 0.15

    value_score = _score_bucket(avg_value_traded_20d, 4.0e8, 6.0e10)
    volume_score = _score_bucket(_safe_float(last.get("REL_VOL"), np.nan), 0.90, 2.50)
    spread_score = _score_bucket(spread_proxy_20d if np.isfinite(spread_proxy_20d) else 0.20, 0.015, spread_cap, invert=True)
    gap_score = _score_bucket(gap_proxy_20d if np.isfinite(gap_proxy_20d) else 0.20, 0.010, gap_cap, invert=True)
    atr_score = _score_bucket(atr_pct if np.isfinite(atr_pct) else 0.20, 0.018, atr_cap, invert=True)
    rr_score = _safe_float(stock_res.get("tradeability_components", {}).get("rr_tradeability_score"), np.nan)
    if not np.isfinite(rr_score):
        rr_score = _safe_float(stock_res.get("tradeability_score"), np.nan)
    if not np.isfinite(rr_score):
        rr_score = 50.0

    tradeability_score = float(np.clip(
        (value_score * 0.30)
        + (volume_score * 0.22)
        + (spread_score * 0.05)
        + (gap_score * 0.10)
        + (atr_score * 0.13)
        + (rr_score * 0.20),
        0.0,
        100.0,
    ))

    tradeability_threshold = 68.0 if market_regime == "BEAR" else (64.0 if market_regime == "SIDEWAYS" else 60.0)
    gate_ok = bool(
        np.isfinite(close)
        and close >= 100.0
        and np.isfinite(tradeability_score)
        and tradeability_score >= tradeability_threshold
        and avg_value_traded_20d >= value_floor
        and (not np.isfinite(spread_proxy_20d) or spread_proxy_20d <= spread_cap)
        and (not np.isfinite(gap_proxy_20d) or gap_proxy_20d <= gap_cap)
        and (not np.isfinite(atr_pct) or atr_pct <= atr_cap)
    )

    reason_bits = []
    if avg_value_traded_20d < value_floor:
        reason_bits.append(f"Avg value {avg_value_traded_20d/1e9:.2f}B < {value_floor/1e9:.2f}B")
    if np.isfinite(spread_proxy_20d) and spread_proxy_20d > spread_cap:
        reason_bits.append(f"Spread proxy {spread_proxy_20d:.1%} > {spread_cap:.1%}")
    if np.isfinite(gap_proxy_20d) and gap_proxy_20d > gap_cap:
        reason_bits.append(f"Gap proxy {gap_proxy_20d:.1%} > {gap_cap:.1%}")
    if np.isfinite(atr_pct) and atr_pct > atr_cap:
        reason_bits.append(f"ATR% {atr_pct:.1%} too wide")
    if np.isfinite(tradeability_score) and tradeability_score < tradeability_threshold:
        reason_bits.append(f"Score {tradeability_score:.0f} < {tradeability_threshold:.0f}")
    if np.isfinite(close) and close < 100.0:
        reason_bits.append("Price below minimum live threshold")

    if tradeability_score >= 88:
        tier = "Institutional"
    elif tradeability_score >= 76:
        tier = "Good"
    elif tradeability_score >= 66:
        tier = "Watch"
    else:
        tier = "Avoid"

    return {
        "avg_value_traded_20d": float(avg_value_traded_20d),
        "spread_proxy_20d": float(spread_proxy_20d) if np.isfinite(spread_proxy_20d) else np.nan,
        "gap_proxy_20d": float(gap_proxy_20d) if np.isfinite(gap_proxy_20d) else np.nan,
        "atr_pct": float(atr_pct) if np.isfinite(atr_pct) else np.nan,
        "tradeability_score": float(tradeability_score),
        "tradeability_tier": tier,
        "tradeability_gate_ok": gate_ok,
        "tradeability_gate_reason": "OK" if gate_ok else (", ".join(reason_bits) if reason_bits else "Tradeability gate off"),
        "liquidity_ok": bool(np.isfinite(close) and close >= 100.0 and avg_value_traded_20d >= value_floor),
        "tradeability_threshold": float(tradeability_threshold),
        "tradeability_value_floor": float(value_floor),
        "tradeability_spread_cap": float(spread_cap),
        "tradeability_gap_cap": float(gap_cap),
    }


def _build_setup_lifecycle_snapshot(stock_res: dict, entry_plan: dict | None = None, *, plan_reason: str = "") -> dict:
    """Summarize whether a setup is still valid and what to do next."""
    d = stock_res.get("df")
    last = stock_res.get("last")
    close = _safe_float(last.get("Close"), np.nan) if last is not None else np.nan
    entry_plan = entry_plan or {}

    entry_price = _safe_float(entry_plan.get("entry_price_plan"), np.nan)
    stop_price = _safe_float(entry_plan.get("stop_loss_plan"), np.nan)
    target_1 = _safe_float(entry_plan.get("target_1"), np.nan)
    target_2 = _safe_float(entry_plan.get("target_2"), np.nan)
    entry_zone_low = _safe_float(entry_plan.get("entry_zone_low"), np.nan)
    entry_zone_high = _safe_float(entry_plan.get("entry_zone_high"), np.nan)
    rr1 = _safe_float(entry_plan.get("risk_reward_1"), np.nan)
    rr2 = _safe_float(entry_plan.get("risk_reward_2"), np.nan)
    setup_kind = str(entry_plan.get("setup_kind") or stock_res.get("setup_kind") or "None")
    tradeability_ok = bool(stock_res.get("tradeability_gate_ok", False))
    tradeability_reason = str(stock_res.get("tradeability_gate_reason", "n/a"))
    decision = str(stock_res.get("decision", "AVOID") or "AVOID").upper()

    ages = []
    for key in ("unicorn_setup_age_bars", "unicorn_sniper_age_bars", "fvg_age_bars"):
        val = _safe_float(stock_res.get(key), np.nan)
        if np.isfinite(val) and val >= 0:
            ages.append(float(val))
    age_bars = min(ages) if ages else np.nan
    age_limit = {
        "BREAKOUT": 12,
        "SNIPER": 8,
        "UNICORN": 10,
        "PULLBACK": 14,
        "REVERSAL": 14,
    }.get(setup_kind.upper(), 12)

    stage = "NO_SETUP"
    validity_ok = False
    next_action = "WAIT"
    reasons: list[str] = []

    entry_kind = setup_kind.upper()
    has_actionable_entry = bool(np.isfinite(entry_price) and entry_price > 0)
    entry_watch_ready = bool(has_actionable_entry and entry_kind in {"BREAKOUT", "SNIPER", "UNICORN", "REVERSAL", "PULLBACK"})

    bar_low = _safe_float(stock_res.get("last", {}).get("Low") if isinstance(stock_res.get("last", {}), dict) else np.nan, np.nan)
    bar_high = _safe_float(stock_res.get("last", {}).get("High") if isinstance(stock_res.get("last", {}), dict) else np.nan, np.nan)
    entry_zone_touched = bool(
        np.isfinite(bar_low)
        and np.isfinite(bar_high)
        and np.isfinite(entry_zone_low)
        and np.isfinite(entry_zone_high)
        and bar_low <= entry_zone_high
        and bar_high >= entry_zone_low
    )
    entry_level_touched = bool(
        np.isfinite(bar_low)
        and np.isfinite(bar_high)
        and np.isfinite(entry_price)
        and bar_low <= entry_price <= bar_high
    )

    if not tradeability_ok:
        if entry_watch_ready:
            stage = "ENTRY_WATCH"
            next_action = "MANUAL REVIEW / SIZE DOWN"
            validity_ok = True
            reasons.append(f"Soft tradeability block: {tradeability_reason}")
        else:
            stage = "NOT_TRADEABLE"
            next_action = "SKIP"
            reasons.append(tradeability_reason)
    elif not has_actionable_entry:
        stage = "NO_ENTRY"
        next_action = "WAIT"
        reasons.append("No actionable entry")
    else:
        if np.isfinite(age_bars) and age_bars > age_limit:
            stage = "EXPIRED"
            next_action = "REMOVE"
            reasons.append(f"Age {age_bars:.0f} > {age_limit}")
        elif np.isfinite(stop_price) and np.isfinite(close) and close <= stop_price:
            stage = "INVALIDATED"
            next_action = "CUT / DROP"
            reasons.append(f"Close {close:.0f} <= stop {stop_price:.0f}")
        elif np.isfinite(target_2) and np.isfinite(close) and close >= target_2:
            stage = "TARGET_2_HIT"
            next_action = "TAKE PROFIT / TRAIL"
            validity_ok = True
            reasons.append("Target 2 reached")
        elif np.isfinite(target_1) and np.isfinite(close) and close >= target_1:
            stage = "TARGET_1_HIT"
            next_action = "TRAIL / HOLD"
            validity_ok = True
            reasons.append("Target 1 reached")
        elif entry_zone_touched:
            if entry_level_touched:
                stage = "ENTRY_TRIGGERED"
                next_action = "MANAGE RISK"
                reasons.append("Bar traded through the planned entry level")
            else:
                stage = "ENTRY_ZONE"
                next_action = "PREPARE LIMIT ORDER"
                reasons.append("Bar traded through the planned entry zone")
            validity_ok = True
        elif np.isfinite(close) and np.isfinite(entry_zone_high) and close > entry_zone_high:
            stage = "PRE_ENTRY_WATCH"
            next_action = "WAIT FOR RETEST / PULLBACK"
            validity_ok = True
            reasons.append("Setup is valid; awaiting retrace to the planned entry zone")
        else:
            stage = "WATCHLIST"
            next_action = "WAIT"
            validity_ok = True
            reasons.append("Setup still waiting")

    if decision not in {"BUY", "STRONG BUY", "WATCHLIST"} and stage == "WATCHLIST":
        reasons.append(f"Decision={decision}")

    if not validity_ok and stage not in {"INVALIDATED", "EXPIRED", "NOT_TRADEABLE"}:
        validity_ok = False

    distance_to_entry_pct = np.nan
    distance_to_entry_atr = np.nan
    fill_probability = np.nan
    if np.isfinite(close) and close > 0 and np.isfinite(entry_price):
        distance_to_entry_pct = float(((entry_price - close) / close) * 100.0)
        atr_v = _safe_float(stock_res.get("last", {}).get("ATR14") if isinstance(stock_res.get("last", {}), dict) else np.nan, np.nan)
        if not np.isfinite(atr_v) or atr_v <= 0:
            atr_v = _safe_float(stock_res.get("ATR14"), np.nan)
        if not np.isfinite(atr_v) or atr_v <= 0:
            atr_v = max(close * 0.02, 1.0)
        distance_to_entry_atr = float(abs(close - entry_price) / max(atr_v, 1e-9))
        fill_probability = _estimate_setup_fill_probability(
            setup_kind,
            distance_to_entry_atr,
            age_bars=age_bars,
            setup_valid=bool(validity_ok),
            has_liquidity_sweep=bool(stock_res.get("unicorn_setup", False)),
            has_mss=bool(stock_res.get("unicorn_sniper_valid", False)),
            rr1=rr1,
            rr2=rr2,
            setup_fresh=bool(stock_res.get("unicorn_setup_fresh", False)),
            entry_zone_width_atr=(
                abs(_safe_float(stock_res.get("candidate_entry_zone_high", stock_res.get("entry_zone_high", np.nan)), np.nan) - _safe_float(stock_res.get("candidate_entry_zone_low", stock_res.get("entry_zone_low", np.nan)), np.nan)) / max(atr_v, 1e-9)
                if np.isfinite(_safe_float(stock_res.get("candidate_entry_zone_high", stock_res.get("entry_zone_high", np.nan)), np.nan))
                and np.isfinite(_safe_float(stock_res.get("candidate_entry_zone_low", stock_res.get("entry_zone_low", np.nan)), np.nan))
                else np.nan
            ),
            structure_confluence=_setup_structure_confluence_score(
                entry_price,
                atr_v,
                entry_zone_low=_safe_float(stock_res.get("candidate_entry_zone_low", stock_res.get("entry_zone_low", np.nan)), np.nan),
                entry_zone_high=_safe_float(stock_res.get("candidate_entry_zone_high", stock_res.get("entry_zone_high", np.nan)), np.nan),
                recent_swing_low=_safe_float(stock_res.get("recent_swing_low", np.nan), np.nan),
                recent_swing_high=_safe_float(stock_res.get("recent_swing_high", np.nan), np.nan),
                sweep_low=_safe_float(stock_res.get("unicorn_sweep_low", np.nan), np.nan),
                breaker_bottom=_safe_float(stock_res.get("unicorn_breaker_bottom", np.nan), np.nan),
                fvg_bottom=_safe_float(stock_res.get("unicorn_fvg_bottom", np.nan), np.nan),
                fvg_top=_safe_float(stock_res.get("unicorn_fvg_top", np.nan), np.nan),
                support_anchor=_safe_float(stock_res.get("support_anchor", np.nan), np.nan),
                resistance_anchor=_safe_float(stock_res.get("resistance_anchor", np.nan), np.nan),
            ),
        )

    return {
        "setup_lifecycle_stage": stage,
        "setup_validity_ok": bool(validity_ok),
        "setup_validity_reason": "; ".join([r for r in reasons if r]) if reasons else plan_reason or "n/a",
        "setup_next_action": next_action,
        "setup_age_bars": float(age_bars) if np.isfinite(age_bars) else np.nan,
        "setup_age_limit": float(age_limit),
        "setup_distance_to_entry_pct": distance_to_entry_pct,
        "setup_distance_to_entry_atr": distance_to_entry_atr,
        "setup_fill_probability": fill_probability,
        "setup_rr_1": float(rr1) if np.isfinite(rr1) else np.nan,
        "setup_rr_2": float(rr2) if np.isfinite(rr2) else np.nan,
        "projected_first_leg": str(entry_plan.get("projected_first_leg", "n/a")),
        "projected_rebound_leg": str(entry_plan.get("projected_rebound_leg", "n/a")),
        "entry_zone_role": str(entry_plan.get("entry_zone_role", "n/a")),
        "entry_zone_label": str(entry_plan.get("entry_zone_label", "n/a")),
        "entry_projection_summary": str(entry_plan.get("entry_projection_summary", "n/a")),
    }

def _entry_price_from_zone(entry_zone_low: float, entry_zone_high: float, bias: float = 0.22) -> float:
    """Pick a conservative limit-entry price inside a setup zone.

    bias=0.0 means the very bottom of the zone; bias=1.0 means the top.
    We intentionally favor the lower half so the scanner does not turn a
    retest setup into a late chase-entry that sits too close to the close.
    """
    if not np.isfinite(entry_zone_low) or not np.isfinite(entry_zone_high):
        return np.nan
    low = float(entry_zone_low)
    high = float(max(entry_zone_high, entry_zone_low))
    if high <= low:
        return low
    bias = float(np.clip(bias, 0.0, 1.0))
    return float(low + (high - low) * bias)


def build_entry_plan(
    stock_res: dict,
    entry_buffer_atr: float = 0.25,
    stop_loss_atr: float = 1.8,
    target_1_atr: float = 2.2,
    target_2_atr: float = 3.8,
) -> dict:
    """Build a practical trade plan using the canonical shared entry engine."""
    return _build_consistent_entry_plan(
        stock_res,
        entry_buffer_atr=entry_buffer_atr,
        stop_loss_atr=stop_loss_atr,
        target_1_atr=target_1_atr,
        target_2_atr=target_2_atr,
    )

# =========================================================
# Safe public wrappers
# =========================================================

def _safe_cycle_fallback() -> tuple[int, int, bool, dict]:
    return (
        20,
        999,
        False,
        {
            "cycle_reliability": np.nan,
            "time_to_next_top": np.nan,
            "phase_age_bars": np.nan,
            "phase_age_pct": np.nan,
            "cycle_gate_reason": "cycle_fallback",
            "dominant_period": 20,
        },
    )

_orig_compute_cycle_features = compute_cycle_features

def compute_cycle_features(close_series):
    """Safe wrapper that suppresses numeric warnings and never raises."""
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return _orig_compute_cycle_features(close_series)
    except Exception:
        return _safe_cycle_fallback()


_orig_classify_8_phase = classify_8_phase

def classify_8_phase(d: pd.DataFrame) -> dict:
    """Safe wrapper that auto-prepares technical columns when possible."""
    try:
        if isinstance(d, pd.DataFrame):
            required = {"EMA20", "EMA50", "EMA200", "RSI14", "ADX14"}
            if not required.issubset(set(d.columns)):
                d = _ensure_technical_columns(d)
        result = _orig_classify_8_phase(d)
        if not isinstance(result, dict):
            return {
                "phase": "Unknown",
                "phase_confidence": 0.0,
                "phase_rank": 0.0,
                "phase_reason": "Invalid phase result",
                "phase_scores": {},
            }
        return result
    except Exception as exc:
        return {
            "phase": "Unknown",
            "phase_confidence": 0.0,
            "phase_rank": 0.0,
            "phase_reason": f"classify_8_phase_fallback: {type(exc).__name__}",
            "phase_scores": {},
        }


_orig_build_macro_liquidity_gate = build_macro_liquidity_gate

def build_macro_liquidity_gate(*args, **kwargs):
    try:
        return _orig_build_macro_liquidity_gate(*args, **kwargs)
    except Exception as exc:
        benchmark_symbol = kwargs.get("benchmark_symbol") if "benchmark_symbol" in kwargs else (args[1] if len(args) > 1 else "")
        return {
            "benchmark_symbol": benchmark_symbol,
            "macro_phase": "Unknown",
            "macro_phase_confidence": 0.0,
            "macro_period": np.nan,
            "macro_time_to_bottom": np.nan,
            "macro_time_to_top": np.nan,
            "macro_phase_age_bars": np.nan,
            "macro_phase_age_pct": np.nan,
            "macro_cycle_reliability": np.nan,
            "macro_cycle_gate_reason": f"macro_fallback: {type(exc).__name__}",
            "macro_score": 50.0,
            "macro_gate_ok": True,
            "macro_gate_reason": f"macro_fallback: {type(exc).__name__}",
            "macro_multiplier": 1.0,
            "cycle_tuple": _safe_cycle_fallback(),
            "benchmark_df": pd.DataFrame(),
            "benchmark_last": None,
            "benchmark_adx": np.nan,
            "benchmark_cycle_info": {},
        }


_score_stock_smc_base = score_stock_smc

def score_stock_smc(*args, **kwargs):
    try:
        return _score_stock_smc_base(*args, **kwargs)
    except Exception as exc:
        return {
            "valid": False,
            "symbol": kwargs.get("symbol", "n/a"),
            "decision": "REJECT",
            "score": 0.0,
            "core_score": 0.0,
            "market_structure_score": 0.0,
            "rs_composite_score": 50.0,
            "tradeability_score": 0.0,
            "trend_score": 0.0,
            "momentum_score": 0.0,
            "smc_score": 0.0,
            "reversal_score_pct": 0.0,
            "risk_score": 100.0,
            "close": np.nan,
            "rsi": np.nan,
            "adx": np.nan,
            "rel_vol": np.nan,
            "smart_money_score": 0.0,
            "cmf20": np.nan,
            "mfi14": np.nan,
            "stoch_k": np.nan,
            "stoch_d": np.nan,
            "cci20": np.nan,
            "roc12": np.nan,
            "dominant_period": np.nan,
            "time_to_bottom": np.nan,
            "time_to_top": np.nan,
            "phase_age_bars": np.nan,
            "phase_age_pct": np.nan,
            "cycle_reliability": np.nan,
            "cycle_gate_reason": f"score_fallback: {type(exc).__name__}",
            "cycle_info": {},
            "macro_symbol": "",
            "macro_phase": "Unknown",
            "macro_score": 50.0,
            "macro_gate_ok": True,
            "macro_gate_reason": f"score_fallback: {type(exc).__name__}",
            "macro_multiplier": 1.0,
            "macro_cycle_reliability": np.nan,
            "macro_time_to_bottom": np.nan,
            "macro_time_to_top": np.nan,
            "macro_phase_age_bars": np.nan,
            "macro_phase_age_pct": np.nan,
            "future_fundamental_score": np.nan,
            "future_fundamental_grade": "n/a",
            "future_fundamental_direction": "n/a",
            "future_fundamental_confidence": np.nan,
            "future_fundamental_phase": "Unknown",
            "future_fundamental_reason": f"score_fallback: {type(exc).__name__}",
            "phase": "Unknown",
            "phase_confidence": 0.0,
            "phase_rank": 0.0,
            "phase_reason": f"score_fallback: {type(exc).__name__}",
            "phase_scores": {},
            "liquidity_ok": False,
            "trend_ok": False,
            "unicorn_setup": False,
            "unicorn_sniper": False,
            "unicorn_entry_style": "n/a",
            "fvg_present": False,
            "fvg_age_bars": np.nan,
            "fvg_age_days": np.nan,
            "fvg_status": "n/a",
            "fvg_fresh": False,
            "fvg_valid": False,
            "fvg_mitigated": False,
            "fvg_top": np.nan,
            "fvg_bottom": np.nan,
            "ob_present": False,
            "unicorn_setup_valid": False,
            "unicorn_setup_status": "n/a",
            "unicorn_setup_age_bars": np.nan,
            "unicorn_setup_age_days": np.nan,
            "unicorn_setup_fresh": False,
            "unicorn_sniper_valid": False,
            "unicorn_sniper_status": "n/a",
            "unicorn_setup_reason": f"score_fallback: {type(exc).__name__}",
            "reversal_score": 0.0,
            "reversal_hits": 0,
            "obv_trend": "n/a",
            "obv_slope10": 0.0,
            "entry_zone_low": np.nan,
            "entry_zone_high": np.nan,
            "entry_trigger": np.nan,
            "entry_price": np.nan,
            "stop_price": np.nan,
            "unicorn_setup_confirmed": False,
            "unicorn_sniper_confirmed": False,
            "unicorn_fvg_top": np.nan,
            "unicorn_fvg_bottom": np.nan,
            "unicorn_breaker_top": np.nan,
            "unicorn_breaker_bottom": np.nan,
            "unicorn_sweep_low": np.nan,
            "notes": f"score_fallback: {type(exc).__name__}",
            "df": pd.DataFrame(),
            "last": pd.Series(dtype=float),
        }


_orig_build_entry_plan = build_entry_plan

def build_entry_plan(*args, **kwargs):
    try:
        return _orig_build_entry_plan(*args, **kwargs)
    except Exception as exc:
        return {
            "entry_zone_low": np.nan,
            "entry_zone_high": np.nan,
            "entry_price_plan": np.nan,
            "entry_trigger": np.nan,
            "stop_loss_plan": np.nan,
            "target_1": np.nan,
            "target_2": np.nan,
            "risk_per_share": np.nan,
            "risk_reward_1": np.nan,
            "risk_reward_2": np.nan,
            "upside_to_t1_pct": np.nan,
            "upside_to_t2_pct": np.nan,
            "plan_reason": f"entry_plan_fallback: {type(exc).__name__}",
            "setup_kind": "n/a",
        }


_orig_compute_institutional_forward_score = compute_institutional_forward_score

def compute_institutional_forward_score(*args, **kwargs):
    try:
        return _orig_compute_institutional_forward_score(*args, **kwargs)
    except Exception as exc:
        return {
            "ifs_score": np.nan,
            "ifs_grade": "n/a",
            "ifs_breakdown": {},
            "ifs_detail": {"error": f"ifs_fallback: {type(exc).__name__}"},
        }

# =====================================================================
# Conservative quality overrides
# =====================================================================

def _safe_max(*values):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan


def _setup_entry_profile(
    setup_kind: str,
    entry_buffer_atr: float,
    stop_loss_atr: float,
    target_1_atr: float,
    target_2_atr: float,
) -> dict:
    """Conservative profile: fewer signals, better RR, less late chasing."""
    kind = str(setup_kind or "").strip().upper()
    profiles = {
        "BREAKOUT": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.60),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.42),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.05),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.20),
            "entry_bias": 0.20,
            "late_entry_bias": 0.14,
            "chase_limit_atr": 0.18,
            "rr_floor_1": 1.75,
            "rr_floor_2": 2.60,
            "max_risk_pct": 0.060,
            "pullback_floor_atr": 0.22,
        },
        "SNIPER": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.80),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.54),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.05),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.12),
            "entry_bias": 0.16,
            "late_entry_bias": 0.08,
            "chase_limit_atr": 0.16,
            "rr_floor_1": 2.00,
            "rr_floor_2": 3.22,
            "max_risk_pct": 0.065,
            "pullback_floor_atr": 0.30,
        },
        "UNICORN": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.82),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.60),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.03),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.15),
            "entry_bias": 0.14,
            "late_entry_bias": 0.11,
            "chase_limit_atr": 0.20,
            "rr_floor_1": 1.78,
            "rr_floor_2": 2.95,
            "max_risk_pct": 0.065,
            "pullback_floor_atr": 0.25,
        },
        "PULLBACK": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.82),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.72),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.00),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.08),
            "entry_bias": 0.18,
            "late_entry_bias": 0.12,
            "chase_limit_atr": 0.18,
            "rr_floor_1": 1.70,
            "rr_floor_2": 2.70,
            "max_risk_pct": 0.060,
            "pullback_floor_atr": 0.20,
        },
        "REVERSAL": {
            "entry_buffer_atr": max(0.0, float(entry_buffer_atr) * 0.72),
            "stop_loss_atr": max(0.0, float(stop_loss_atr) * 0.72),
            "target_1_atr": max(0.0, float(target_1_atr) * 1.00),
            "target_2_atr": max(0.0, float(target_2_atr) * 1.10),
            "entry_bias": 0.20,
            "late_entry_bias": 0.14,
            "chase_limit_atr": 0.18,
            "rr_floor_1": 1.80,
            "rr_floor_2": 2.85,
            "max_risk_pct": 0.065,
            "pullback_floor_atr": 0.18,
        },
    }
    return profiles.get(kind, profiles["UNICORN"]).copy()


def _estimate_setup_fill_probability(
    setup_kind: str,
    distance_to_entry_atr: float,
    age_bars: float | int | float = np.nan,
    setup_valid: bool = True,
    has_liquidity_sweep: bool = False,
    has_mss: bool = False,
    rr1: float = np.nan,
    rr2: float = np.nan,
    setup_fresh: bool = False,
    entry_zone_width_atr: float = np.nan,
    structure_confluence: float = np.nan,
) -> float:
    """More selective fill model.

    The score rewards setups that sit close to structure and have a wider
    but still controlled limit zone.
    """
    kind = str(setup_kind or "").strip().upper()
    base_map = {
        "BREAKOUT": 62.0,
        "SNIPER": 72.0,
        "UNICORN": 71.0,
        "PULLBACK": 73.0,
        "REVERSAL": 61.0,
    }
    score = base_map.get(kind, 64.0)

    dist = float(distance_to_entry_atr) if np.isfinite(distance_to_entry_atr) else 4.0
    if dist >= 6.0:
        return 0.0

    if kind in {"UNICORN", "SNIPER", "PULLBACK", "REVERSAL"}:
        score -= max(0.0, dist - 0.50) * 8.5
        score -= max(0.0, dist - 1.80) * 6.5
        score -= max(0.0, dist - 3.50) * 8.0
    else:
        score -= max(0.0, dist - 0.50) * 10.0
        score -= max(0.0, dist - 1.50) * 8.0
        score -= max(0.0, dist - 3.50) * 10.0

    if np.isfinite(age_bars):
        age_bars = float(age_bars)
        score -= max(0.0, age_bars - 3.0) * 1.8
        score -= max(0.0, age_bars - 8.0) * 1.3
        score -= max(0.0, age_bars - 14.0) * 0.8

    zone_w = float(entry_zone_width_atr) if np.isfinite(entry_zone_width_atr) else np.nan
    if np.isfinite(zone_w):
        if 0.35 <= zone_w <= 1.50:
            score += 7.5
        elif 1.50 < zone_w <= 2.20:
            score += 4.5
        elif zone_w < 0.25:
            score -= 4.0
        elif zone_w > 2.80:
            score -= 3.0
        else:
            score -= 1.5

    if np.isfinite(structure_confluence):
        score += min(max(float(structure_confluence), 0.0), 30.0) * 0.55

    if setup_fresh:
        score += 8.0
    if has_liquidity_sweep:
        score += 7.0 if kind in {"UNICORN", "SNIPER", "PULLBACK", "REVERSAL"} else 5.0
    if has_mss:
        score += 7.0 if kind in {"UNICORN", "SNIPER", "PULLBACK", "REVERSAL"} else 5.0

    if kind in {"UNICORN", "SNIPER", "PULLBACK"} and dist <= 1.25 and (has_liquidity_sweep or has_mss):
        score += 5.0
    if kind in {"UNICORN", "SNIPER"} and dist <= 0.90 and setup_fresh:
        score += 4.0

    if np.isfinite(rr1):
        rr1 = float(rr1)
        score += np.clip((rr1 - 1.4) * 4.0, -8.0, 8.0)
    if np.isfinite(rr2):
        rr2 = float(rr2)
        score += np.clip((rr2 - 2.4) * 4.0, -8.0, 10.0)

    if not setup_valid:
        score *= 0.45 if kind in {"UNICORN", "SNIPER", "PULLBACK", "REVERSAL"} else 0.40

    return float(np.clip(score, 0.0, 100.0))

_orig_build_consistent_entry_plan = _build_consistent_entry_plan

def _build_consistent_entry_plan(
    stock_res: dict,
    entry_buffer_atr: float = 0.25,
    stop_loss_atr: float = 1.8,
    target_1_atr: float = 2.2,
    target_2_atr: float = 3.8,
) -> dict:
    """Post-process the canonical plan into a stricter, higher-quality plan."""
    plan = _orig_build_consistent_entry_plan(
        stock_res,
        entry_buffer_atr=entry_buffer_atr,
        stop_loss_atr=stop_loss_atr,
        target_1_atr=target_1_atr,
        target_2_atr=target_2_atr,
    )
    if not isinstance(plan, dict) or str(plan.get("setup_kind", "None")).upper() == "NONE":
        return plan

    d = stock_res.get("df")
    last = stock_res.get("last")
    if d is None or getattr(d, "empty", True) or last is None:
        return plan

    close = _safe_float(last.get("Close"), np.nan)
    atr_v = _safe_float(last.get("ATR14"), np.nan)
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close * 0.02 if np.isfinite(close) and close > 0 else 1.0, 1.0)

    setup_kind = str(plan.get("setup_kind", "None"))
    profile = _setup_entry_profile(setup_kind, entry_buffer_atr, stop_loss_atr, target_1_atr, target_2_atr)

    entry_price = _safe_float(plan.get("entry_price_plan"), np.nan)
    stop_price = _safe_float(plan.get("stop_loss_plan"), np.nan)
    target_1 = _safe_float(plan.get("target_1"), np.nan)
    target_2 = _safe_float(plan.get("target_2"), np.nan)

    if not np.isfinite(entry_price) or not np.isfinite(stop_price) or not np.isfinite(target_1) or not np.isfinite(target_2):
        return plan

    if stop_price >= entry_price:
        stop_price = max(entry_price - 0.90 * atr_v, 0.0)

    risk_per_share = float(max(entry_price - stop_price, 1e-9))
    min_t1 = entry_price + max(profile["rr_floor_1"] * risk_per_share, 1.10 * atr_v)
    min_t2 = entry_price + max(profile["rr_floor_2"] * risk_per_share, 2.20 * atr_v)

    # Keep RR floors as validation gates only; do not hard-clamp structural targets.
    target_adjusted = False
    if target_2 <= target_1:
        target_2 = float(target_1 + max(0.85 * atr_v, 0.65 * risk_per_share))
        target_adjusted = True

    rr1 = float((target_1 - entry_price) / risk_per_share)
    rr2 = float((target_2 - entry_price) / risk_per_share)
    upside_t1 = float((target_1 / entry_price - 1.0) * 100.0)
    upside_t2 = float((target_2 / entry_price - 1.0) * 100.0)

    distance_to_entry_atr = _setup_distance_to_entry_atr(close, entry_price, atr_v)
    age_bars = _safe_float(stock_res.get("setup_age_bars"), np.nan)
    fill_probability = _estimate_setup_fill_probability(
        setup_kind,
        distance_to_entry_atr,
        age_bars=age_bars,
        setup_valid=bool(stock_res.get("tradeability_gate_ok", True)),
        has_liquidity_sweep=bool(stock_res.get("unicorn_setup", False)),
        has_mss=bool(stock_res.get("unicorn_sniper_valid", False)),
        rr1=rr1,
        rr2=rr2,
        setup_fresh=bool(stock_res.get("unicorn_setup_fresh", False)),
        entry_zone_width_atr=(
            abs(_safe_float(plan.get("entry_zone_high", np.nan), np.nan) - _safe_float(plan.get("entry_zone_low", np.nan), np.nan)) / max(atr_v, 1e-9)
            if np.isfinite(_safe_float(plan.get("entry_zone_high", np.nan), np.nan))
            and np.isfinite(_safe_float(plan.get("entry_zone_low", np.nan), np.nan))
            else np.nan
        ),
        structure_confluence=_setup_structure_confluence_score(
            entry_price,
            atr_v,
            entry_zone_low=_safe_float(plan.get("entry_zone_low", np.nan), np.nan),
            entry_zone_high=_safe_float(plan.get("entry_zone_high", np.nan), np.nan),
            recent_swing_low=_safe_float(stock_res.get("recent_swing_low", np.nan), np.nan),
            recent_swing_high=_safe_float(stock_res.get("recent_swing_high", np.nan), np.nan),
            sweep_low=_safe_float(stock_res.get("unicorn_sweep_low", np.nan), np.nan),
            breaker_bottom=_safe_float(stock_res.get("unicorn_breaker_bottom", np.nan), np.nan),
            fvg_bottom=_safe_float(stock_res.get("unicorn_fvg_bottom", np.nan), np.nan),
            fvg_top=_safe_float(stock_res.get("unicorn_fvg_top", np.nan), np.nan),
            support_anchor=_safe_float(stock_res.get("support_anchor", np.nan), np.nan),
            resistance_anchor=_safe_float(stock_res.get("resistance_anchor", np.nan), np.nan),
        ),
    )

    tradeability_ok = bool(stock_res.get("tradeability_gate_ok", False))
    tradeability_score = _safe_float(stock_res.get("tradeability_score"), np.nan)
    risk_pct = float((entry_price - stop_price) / max(entry_price, 1e-9))
    pullback_atr = float((close - entry_price) / max(atr_v, 1e-9)) if np.isfinite(close) and close > 0 else np.nan

    fill_floor = 32.0 if setup_kind.upper() in {"UNICORN", "SNIPER"} else 28.0
    distance_cap_atr = 5.5 if setup_kind.upper() in {"UNICORN", "SNIPER"} else 6.0
    setup_stale = bool(np.isfinite(distance_to_entry_atr) and distance_to_entry_atr > distance_cap_atr)

    candidate_allowed = bool(
        setup_kind.upper() in {"BREAKOUT", "SNIPER", "UNICORN", "PULLBACK", "REVERSAL"}
    )

    execution_ready = bool(
        tradeability_ok
        and candidate_allowed
        and np.isfinite(rr1) and rr1 >= profile["rr_floor_1"]
        and np.isfinite(rr2) and rr2 >= profile["rr_floor_2"]
        and risk_pct <= profile["max_risk_pct"]
        and (not np.isfinite(pullback_atr) or pullback_atr >= profile["pullback_floor_atr"])
        and (not np.isfinite(fill_probability) or fill_probability >= fill_floor)
        and not setup_stale
        and (not np.isfinite(tradeability_score) or tradeability_score >= 58.0)
    )

    if target_adjusted:
        plan_reason = str(plan.get("plan_reason", "n/a"))
        plan["plan_reason"] = f"{plan_reason}; tightened targets for RR quality"

    lifecycle = _build_setup_lifecycle_snapshot(
        stock_res,
        {
            "entry_price_plan": entry_price,
            "stop_loss_plan": stop_price,
            "target_1": target_1,
            "target_2": target_2,
            "entry_zone_low": plan.get("entry_zone_low", np.nan),
            "entry_zone_high": plan.get("entry_zone_high", np.nan),
            "risk_reward_1": rr1,
            "risk_reward_2": rr2,
            "setup_kind": setup_kind,
        },
        plan_reason=str(plan.get("plan_reason", "n/a")),
    )

    quality_score = np.nan
    if np.isfinite(rr2) and np.isfinite(fill_probability):
        rr_component = float(np.clip((rr2 - 1.0) / 2.5, 0.0, 1.0))
        fill_component = float(np.clip(fill_probability / 100.0, 0.0, 1.0))
        trade_component = float(np.clip((tradeability_score if np.isfinite(tradeability_score) else 60.0) / 100.0, 0.0, 1.0))
        quality_score = float(np.clip(100.0 * (0.45 * rr_component + 0.30 * fill_component + 0.25 * trade_component), 0.0, 100.0))

    label = "EXECUTION_READY" if execution_ready else ("WATCHLIST_ENTRY" if candidate_allowed else "NONE")
    reason_bits = []
    if not tradeability_ok:
        reason_bits.append(str(stock_res.get("tradeability_gate_reason", "tradeability_blocked")))
    if np.isfinite(rr1) and rr1 < profile["rr_floor_1"]:
        reason_bits.append(f"rr1<{profile['rr_floor_1']:.2f}")
    if np.isfinite(rr2) and rr2 < profile["rr_floor_2"]:
        reason_bits.append(f"rr2<{profile['rr_floor_2']:.2f}")
    if risk_pct > profile["max_risk_pct"]:
        reason_bits.append(f"risk_pct>{profile['max_risk_pct']:.2%}")
    if np.isfinite(fill_probability) and fill_probability < fill_floor:
        reason_bits.append(f"fill<{fill_floor:.0f}")
    if setup_stale:
        reason_bits.append(f"stale>{distance_cap_atr:.1f}ATR")
    if np.isfinite(quality_score) and quality_score < 62.0:
        reason_bits.append(f"quality<{62.0:.0f}")

    plan.update({
        "entry_price_plan": float(entry_price),
        "stop_loss_plan": float(stop_price),
        "target_1": float(target_1),
        "target_2": float(target_2),
        "risk_per_share": float(risk_per_share),
        "risk_reward_1": float(rr1),
        "risk_reward_2": float(rr2),
        "upside_to_t1_pct": float(upside_t1),
        "upside_to_t2_pct": float(upside_t2),
        "setup_distance_to_entry_atr": float(distance_to_entry_atr) if np.isfinite(distance_to_entry_atr) else np.nan,
        "setup_fill_probability": float(fill_probability) if np.isfinite(fill_probability) else np.nan,
        "execution_status": label,
        "execution_status_reason": "; ".join(reason_bits) if reason_bits else str(plan.get("execution_status_reason", "n/a")),
        "entry_candidate_label": label,
        "candidate_entry_price": float(entry_price),
        "candidate_stop_price": float(stop_price),
        "candidate_target_1": float(target_1),
        "candidate_target_2": float(target_2),
        "candidate_risk_reward_1": float(rr1),
        "candidate_risk_reward_2": float(rr2),
        "setup_lifecycle_stage": lifecycle.get("setup_lifecycle_stage", plan.get("setup_lifecycle_stage", "NO_SETUP")),
        "setup_validity_ok": lifecycle.get("setup_validity_ok", plan.get("setup_validity_ok", False)),
        "setup_validity_reason": lifecycle.get("setup_validity_reason", plan.get("setup_validity_reason", "n/a")),
        "setup_next_action": lifecycle.get("setup_next_action", plan.get("setup_next_action", "WAIT")),
        "setup_age_bars": lifecycle.get("setup_age_bars", plan.get("setup_age_bars", np.nan)),
        "setup_age_limit": lifecycle.get("setup_age_limit", plan.get("setup_age_limit", np.nan)),
        "setup_distance_to_entry_pct": lifecycle.get("setup_distance_to_entry_pct", plan.get("setup_distance_to_entry_pct", np.nan)),
        "quality_score": quality_score,
        "execution_ready": execution_ready,
    })
    plan.update(lifecycle)
    return plan


_score_stock_smc_fallback = score_stock_smc

def score_stock_smc(*args, **kwargs):
    """Wrap the base scorer with plan-aware quality penalties and decision tightening."""
    res = _score_stock_smc_fallback(*args, **kwargs)
    if not isinstance(res, dict):
        return res

    try:
        d = res.get("df")
        if d is None or getattr(d, "empty", True):
            return res

        entry_plan = _build_consistent_entry_plan(
            res,
            entry_buffer_atr=float(kwargs.get("entry_buffer_atr", 0.25)) if "entry_buffer_atr" in kwargs else 0.25,
            stop_loss_atr=float(kwargs.get("stop_loss_atr", 1.8)) if "stop_loss_atr" in kwargs else 1.8,
            target_1_atr=float(kwargs.get("target_1_atr", 2.2)) if "target_1_atr" in kwargs else 2.2,
            target_2_atr=float(kwargs.get("target_2_atr", 3.8)) if "target_2_atr" in kwargs else 3.8,
        )
        if isinstance(entry_plan, dict) and entry_plan:
            res["entry_plan"] = entry_plan
            res.update(entry_plan)

            rr2 = _safe_float(entry_plan.get("risk_reward_2"), np.nan)
            fill = _safe_float(entry_plan.get("setup_fill_probability"), np.nan)
            tradeability = _safe_float(res.get("tradeability_score"), np.nan)
            orig_score = _safe_float(res.get("score"), np.nan)

            quality = _safe_float(entry_plan.get("quality_score"), np.nan)
            if not np.isfinite(quality) and np.isfinite(rr2) and np.isfinite(fill):
                rr_component = float(np.clip((rr2 - 1.0) / 2.5, 0.0, 1.0))
                fill_component = float(np.clip(fill / 100.0, 0.0, 1.0))
                trade_component = float(np.clip((tradeability if np.isfinite(tradeability) else 60.0) / 100.0, 0.0, 1.0))
                quality = float(np.clip(100.0 * (0.45 * rr_component + 0.30 * fill_component + 0.25 * trade_component), 0.0, 100.0))
                res["quality_score"] = quality

            if np.isfinite(orig_score) and np.isfinite(quality):
                res["score"] = float(np.clip(0.58 * orig_score + 0.42 * quality, 0.0, 100.0))
            elif np.isfinite(quality):
                res["score"] = float(np.clip(quality, 0.0, 100.0))

            decision = str(res.get("decision", "WATCHLIST") or "WATCHLIST").upper()
            label = str(entry_plan.get("entry_candidate_label", "NONE")).upper()
            if label != "EXECUTION_READY":
                res["decision"] = "WATCHLIST"
            else:
                if (
                    np.isfinite(rr2) and rr2 >= 3.2
                    and np.isfinite(fill) and fill >= 55.0
                    and np.isfinite(tradeability) and tradeability >= 75.0
                    and np.isfinite(quality) and quality >= 75.0
                ):
                    res["decision"] = "STRONG BUY"
                elif np.isfinite(quality) and quality >= 62.0:
                    res["decision"] = "BUY"
                else:
                    res["decision"] = "WATCHLIST"

            if "notes" in res and isinstance(res["notes"], str):
                if np.isfinite(quality):
                    res["notes"] = f"{res['notes']}; Quality={quality:.1f}"
                elif np.isfinite(rr2):
                    res["notes"] = f"{res['notes']}; RR2={rr2:.2f}"

    except Exception:
        # Never let quality post-processing break the base scanner.
        return res

    return res


# =====================================================================
# Final conservative overrides
# =====================================================================

_prev_setup_entry_profile = _setup_entry_profile

def _setup_entry_profile(
    setup_kind: str,
    entry_buffer_atr: float,
    stop_loss_atr: float,
    target_1_atr: float,
    target_2_atr: float,
) -> dict:
    """Final conservative profile.

    This version is deliberately stricter: fewer live signals, tighter invalidation,
    and higher RR floors. It is meant to reduce weak entries that look attractive
    in-sample but degrade in live trading.
    """
    profile = _prev_setup_entry_profile(
        setup_kind,
        entry_buffer_atr=entry_buffer_atr,
        stop_loss_atr=stop_loss_atr,
        target_1_atr=target_1_atr,
        target_2_atr=target_2_atr,
    )
    try:
        kind = str(setup_kind or "").strip().upper()
        if kind in {"BREAKOUT", "UNICORN", "SNIPER", "PULLBACK", "REVERSAL"}:
            profile["rr_floor_1"] = max(float(profile.get("rr_floor_1", 0.0)), 1.90)
            profile["rr_floor_2"] = max(float(profile.get("rr_floor_2", 0.0)), 3.00)
            profile["max_risk_pct"] = min(float(profile.get("max_risk_pct", 0.08)), 0.060)
            profile["chase_limit_atr"] = min(float(profile.get("chase_limit_atr", 0.25)), 0.14)
        if kind in {"UNICORN", "SNIPER"}:
            profile["entry_buffer_atr"] = max(0.0, float(profile.get("entry_buffer_atr", 0.0)) * 0.92)
            profile["stop_loss_atr"] = max(0.0, float(profile.get("stop_loss_atr", 0.0)) * 0.95)
            profile["pullback_floor_atr"] = max(0.0, float(profile.get("pullback_floor_atr", 0.0)) * 0.80)
        elif kind == "BREAKOUT":
            profile["entry_buffer_atr"] = max(0.0, float(profile.get("entry_buffer_atr", 0.0)) * 0.88)
            profile["stop_loss_atr"] = max(0.0, float(profile.get("stop_loss_atr", 0.0)) * 0.90)
    except Exception:
        pass
    return profile

_prev_estimate_setup_fill_probability = _estimate_setup_fill_probability

def _estimate_setup_fill_probability(
    setup_kind: str,
    distance_to_entry_atr: float,
    age_bars: float | int | float = np.nan,
    setup_valid: bool = True,
    has_liquidity_sweep: bool = False,
    has_mss: bool = False,
    rr1: float = np.nan,
    rr2: float = np.nan,
    setup_fresh: bool = False,
    entry_zone_width_atr: float = np.nan,
    structure_confluence: float = np.nan,
) -> float:
    """Final fill model with harder penalties for chase distance and stale setups."""
    score = float(_prev_estimate_setup_fill_probability(
        setup_kind,
        distance_to_entry_atr,
        age_bars=age_bars,
        setup_valid=setup_valid,
        has_liquidity_sweep=has_liquidity_sweep,
        has_mss=has_mss,
        rr1=rr1,
        rr2=rr2,
        setup_fresh=setup_fresh,
        entry_zone_width_atr=entry_zone_width_atr,
        structure_confluence=structure_confluence,
    ))
    try:
        kind = str(setup_kind or "").strip().upper()
        dist = float(distance_to_entry_atr) if np.isfinite(distance_to_entry_atr) else 4.0
        if not setup_valid:
            score *= 0.55
        if dist > 1.20 and kind in {"UNICORN", "SNIPER", "PULLBACK", "REVERSAL"}:
            score -= (dist - 1.20) * 16.0
        if dist > 0.90 and kind == "BREAKOUT":
            score -= (dist - 0.90) * 20.0
        if np.isfinite(age_bars):
            age_bars = float(age_bars)
            if age_bars > 4:
                score -= (age_bars - 4.0) * 2.0
            if age_bars > 10:
                score -= (age_bars - 10.0) * 1.2
        if setup_fresh:
            score += 4.0
        else:
            score -= 4.0
        if np.isfinite(rr2):
            score += float(np.clip((float(rr2) - 2.8) * 3.0, -10.0, 12.0))
        if np.isfinite(rr1):
            score += float(np.clip((float(rr1) - 1.7) * 2.0, -6.0, 8.0))
        if has_liquidity_sweep:
            score += 3.0
        if has_mss:
            score += 3.0
        if np.isfinite(entry_zone_width_atr):
            if 0.30 <= float(entry_zone_width_atr) <= 1.35:
                score += 3.5
            elif float(entry_zone_width_atr) > 2.40:
                score -= 4.0
        score = float(np.clip(score, 0.0, 100.0))
    except Exception:
        pass
    return score

_prev_build_consistent_entry_plan = _build_consistent_entry_plan

def _build_consistent_entry_plan(
    stock_res: dict,
    entry_buffer_atr: float = 0.25,
    stop_loss_atr: float = 1.8,
    target_1_atr: float = 2.2,
    target_2_atr: float = 3.8,
) -> dict:
    """Final entry plan gate.

    Weak plans are demoted to WATCHLIST-like output even if technical structure exists.
    """
    plan = _prev_build_consistent_entry_plan(
        stock_res,
        entry_buffer_atr=entry_buffer_atr,
        stop_loss_atr=stop_loss_atr,
        target_1_atr=target_1_atr,
        target_2_atr=target_2_atr,
    )
    if not isinstance(plan, dict) or not plan:
        return plan

    kind = str(plan.get("setup_kind", "None") or "None").upper()
    if kind == "NONE":
        plan["entry_candidate_label"] = "NONE"
        plan["execution_ready"] = False
        return plan

    try:
        rr1 = _safe_float(plan.get("risk_reward_1"), np.nan)
        rr2 = _safe_float(plan.get("risk_reward_2"), np.nan)
        fill = _safe_float(plan.get("setup_fill_probability"), np.nan)
        quality = _safe_float(plan.get("quality_score"), np.nan)
        dist = _safe_float(plan.get("distance_to_entry_atr"), np.nan)
        risk_pct = _safe_float(plan.get("risk_pct_of_entry"), np.nan)

        # Hard rejection conditions for weak / late / stretched setups.
        hard_reject = False
        if np.isfinite(rr2) and rr2 < 2.65:
            hard_reject = True
        if np.isfinite(fill) and fill < 58.0:
            hard_reject = True
        if np.isfinite(quality) and quality < 68.0:
            hard_reject = True
        if np.isfinite(dist) and dist > 1.35:
            hard_reject = True
        if np.isfinite(risk_pct) and risk_pct > 0.060:
            hard_reject = True

        if hard_reject:
            plan["entry_candidate_label"] = "NONE"
            plan["execution_ready"] = False
            plan["quality_score"] = float(np.clip(min(quality if np.isfinite(quality) else 55.0, 60.0), 0.0, 100.0))
        else:
            plan["entry_candidate_label"] = "EXECUTION_READY"
            plan["execution_ready"] = True
            if np.isfinite(quality):
                plan["quality_score"] = float(np.clip(max(quality, 70.0), 0.0, 100.0))

        # Small adjustment to keep the plan aligned with conservative RR.
        if np.isfinite(rr1) and np.isfinite(rr2) and rr2 < rr1 + 0.55:
            last_obj = stock_res.get("last")
            atr_guess = _safe_float(last_obj.get("ATR14") if hasattr(last_obj, "get") else np.nan, np.nan)
            if not np.isfinite(atr_guess) or atr_guess <= 0:
                atr_guess = _safe_float(stock_res.get("ATR14"), np.nan)
            if not np.isfinite(atr_guess) or atr_guess <= 0:
                atr_guess = 1.0
            t1 = _safe_float(plan.get("target_1"), np.nan)
            if np.isfinite(t1):
                plan["target_2"] = float(t1 + max(0.9 * atr_guess, 1.0))
    except Exception:
        pass
    return plan

_prev_build_entry_plan = build_entry_plan

def build_entry_plan(*args, **kwargs):
    try:
        return _prev_build_entry_plan(*args, **kwargs)
    except Exception as exc:
        return {
            "entry_zone_low": np.nan,
            "entry_zone_high": np.nan,
            "entry_price_plan": np.nan,
            "entry_trigger": np.nan,
            "stop_loss_plan": np.nan,
            "target_1": np.nan,
            "target_2": np.nan,
            "risk_per_share": np.nan,
            "risk_reward_1": np.nan,
            "risk_reward_2": np.nan,
            "upside_to_t1_pct": np.nan,
            "upside_to_t2_pct": np.nan,
            "plan_reason": f"entry_plan_fallback: {type(exc).__name__}",
            "setup_kind": "n/a",
        }

_prev_score_stock_smc = score_stock_smc

def score_stock_smc(*args, **kwargs):
    """Final score wrapper with stricter quality and execution gating."""
    res = _prev_score_stock_smc(*args, **kwargs)
    if not isinstance(res, dict):
        return res
    try:
        entry_plan = res.get("entry_plan") if isinstance(res.get("entry_plan"), dict) else None
        if entry_plan:
            label = str(entry_plan.get("entry_candidate_label", "NONE")).upper()
            quality = _safe_float(entry_plan.get("quality_score"), np.nan)
            rr2 = _safe_float(entry_plan.get("risk_reward_2"), np.nan)
            fill = _safe_float(entry_plan.get("setup_fill_probability"), np.nan)
            if label != "EXECUTION_READY":
                res["decision"] = "WATCHLIST"
            elif np.isfinite(quality) and quality >= 82.0 and np.isfinite(rr2) and rr2 >= 3.15 and np.isfinite(fill) and fill >= 62.0:
                res["decision"] = "STRONG BUY"
            elif np.isfinite(quality) and quality >= 72.0:
                res["decision"] = "BUY"
            else:
                res["decision"] = "WATCHLIST"
            if np.isfinite(quality) and np.isfinite(res.get("score", np.nan)):
                res["score"] = float(np.clip(0.50 * float(res["score"]) + 0.50 * quality, 0.0, 100.0))
            if np.isfinite(fill):
                res["fill_probability"] = float(fill)
    except Exception:
        return res
    return res

_prev_compute_institutional_forward_score = compute_institutional_forward_score

def compute_institutional_forward_score(*args, **kwargs):
    try:
        res = _prev_compute_institutional_forward_score(*args, **kwargs)
        if not isinstance(res, dict):
            return res
        technical_context = kwargs.get("technical_context") or {}
        if isinstance(technical_context, dict):
            entry_plan = technical_context.get("entry_plan")
            if isinstance(entry_plan, dict):
                quality = _safe_float(entry_plan.get("quality_score"), np.nan)
                label = str(entry_plan.get("entry_candidate_label", "NONE")).upper()
                if label != "EXECUTION_READY":
                    res["ifs_score"] = float(np.nan_to_num(_safe_float(res.get("ifs_score"), np.nan), nan=45.0) * 0.90)
                    res.setdefault("ifs_detail", {})["execution_gate"] = "WATCHLIST"
                elif np.isfinite(quality):
                    res["ifs_score"] = float(np.clip(0.55 * _safe_float(res.get("ifs_score"), 50.0) + 0.45 * quality, 0.0, 100.0))
                    res.setdefault("ifs_detail", {})["entry_quality"] = float(quality)
        return res
    except Exception as exc:
        return {
            "ifs_score": np.nan,
            "ifs_grade": "n/a",
            "ifs_breakdown": {},
            "ifs_detail": {"error": f"ifs_fallback: {type(exc).__name__}"},
        }


# =====================================================================
# Ultra-conservative live-trading overrides
# =====================================================================

_prev_setup_entry_profile_2 = _setup_entry_profile

def _setup_entry_profile(
    setup_kind: str,
    entry_buffer_atr: float,
    stop_loss_atr: float,
    target_1_atr: float,
    target_2_atr: float,
) -> dict:
    """Final live-trading profile.

    This version is stricter than the prior conservative layer:
    - only the best long setups are made easy to execute,
    - breakout/reversal are pushed into watchlist territory,
    - pullback and unicorn/sniper require tighter risk.
    """
    profile = _prev_setup_entry_profile_2(
        setup_kind,
        entry_buffer_atr=entry_buffer_atr,
        stop_loss_atr=stop_loss_atr,
        target_1_atr=target_1_atr,
        target_2_atr=target_2_atr,
    )
    try:
        kind = str(setup_kind or "").strip().upper()
        if kind == "PULLBACK":
            profile["entry_buffer_atr"] = max(0.0, float(profile.get("entry_buffer_atr", 0.0)) * 0.90)
            profile["stop_loss_atr"] = max(0.0, float(profile.get("stop_loss_atr", 0.0)) * 0.95)
            profile["rr_floor_1"] = max(float(profile.get("rr_floor_1", 0.0)), 1.95)
            profile["rr_floor_2"] = max(float(profile.get("rr_floor_2", 0.0)), 3.05)
            profile["max_risk_pct"] = min(float(profile.get("max_risk_pct", 0.08)), 0.050)
            profile["chase_limit_atr"] = min(float(profile.get("chase_limit_atr", 0.25)), 0.12)
            profile["pullback_floor_atr"] = max(float(profile.get("pullback_floor_atr", 0.0)), 0.18)
            profile["entry_bias"] = min(float(profile.get("entry_bias", 0.50)), 0.30)
        elif kind in {"UNICORN", "SNIPER"}:
            profile["entry_buffer_atr"] = max(0.0, float(profile.get("entry_buffer_atr", 0.0)) * 0.88)
            profile["stop_loss_atr"] = max(0.0, float(profile.get("stop_loss_atr", 0.0)) * 0.92)
            profile["rr_floor_1"] = max(float(profile.get("rr_floor_1", 0.0)), 2.10)
            profile["rr_floor_2"] = max(float(profile.get("rr_floor_2", 0.0)), 3.30)
            profile["max_risk_pct"] = min(float(profile.get("max_risk_pct", 0.08)), 0.045)
            profile["chase_limit_atr"] = min(float(profile.get("chase_limit_atr", 0.25)), 0.10)
            profile["pullback_floor_atr"] = max(float(profile.get("pullback_floor_atr", 0.0)), 0.22)
            profile["entry_bias"] = min(float(profile.get("entry_bias", 0.50)), 0.20 if kind == "SNIPER" else 0.24)
        else:
            profile["rr_floor_1"] = max(float(profile.get("rr_floor_1", 0.0)), 2.00)
            profile["rr_floor_2"] = max(float(profile.get("rr_floor_2", 0.0)), 3.20)
            profile["max_risk_pct"] = min(float(profile.get("max_risk_pct", 0.08)), 0.040)
            profile["chase_limit_atr"] = min(float(profile.get("chase_limit_atr", 0.25)), 0.10)
    except Exception:
        pass
    return profile


_prev_estimate_setup_fill_probability_2 = _estimate_setup_fill_probability

def _estimate_setup_fill_probability(
    setup_kind: str,
    distance_to_entry_atr: float,
    age_bars: float | int | float = np.nan,
    setup_valid: bool = True,
    has_liquidity_sweep: bool = False,
    has_mss: bool = False,
    rr1: float = np.nan,
    rr2: float = np.nan,
    setup_fresh: bool = False,
    entry_zone_width_atr: float = np.nan,
    structure_confluence: float = np.nan,
) -> float:
    """Stricter fill model that penalizes chasing and stale retraces harder."""
    score = float(_prev_estimate_setup_fill_probability_2(
        setup_kind,
        distance_to_entry_atr,
        age_bars=age_bars,
        setup_valid=setup_valid,
        has_liquidity_sweep=has_liquidity_sweep,
        has_mss=has_mss,
        rr1=rr1,
        rr2=rr2,
        setup_fresh=setup_fresh,
        entry_zone_width_atr=entry_zone_width_atr,
        structure_confluence=structure_confluence,
    ))
    try:
        kind = str(setup_kind or "").strip().upper()
        dist = float(distance_to_entry_atr) if np.isfinite(distance_to_entry_atr) else 4.0
        if kind in {"PULLBACK", "UNICORN", "SNIPER"}:
            if dist > 1.0:
                score -= (dist - 1.0) * 12.0
            if dist > 1.6:
                score -= (dist - 1.6) * 10.0
        else:
            if dist > 0.8:
                score -= (dist - 0.8) * 14.0
            if dist > 1.2:
                score -= (dist - 1.2) * 12.0
        if np.isfinite(age_bars):
            age_bars = float(age_bars)
            if age_bars > 2:
                score -= (age_bars - 2.0) * 2.2
            if age_bars > 6:
                score -= (age_bars - 6.0) * 1.5
        if setup_fresh:
            score += 4.0
        else:
            score -= 5.0
        if np.isfinite(rr1):
            score += float(np.clip((float(rr1) - 1.8) * 2.0, -8.0, 8.0))
        if np.isfinite(rr2):
            score += float(np.clip((float(rr2) - 3.0) * 2.5, -10.0, 12.0))
        if has_liquidity_sweep:
            score += 3.0
        if has_mss:
            score += 3.0
        if np.isfinite(entry_zone_width_atr):
            if 0.30 <= float(entry_zone_width_atr) <= 1.25:
                score += 3.0
            elif float(entry_zone_width_atr) > 2.20:
                score -= 4.0
        if np.isfinite(structure_confluence):
            score += float(np.clip(structure_confluence, 0.0, 30.0)) * 0.15
        if not setup_valid:
            score *= 0.45
        score = float(np.clip(score, 0.0, 100.0))
    except Exception:
        pass
    return score


_prev_build_consistent_entry_plan_2 = _build_consistent_entry_plan

def _build_consistent_entry_plan(
    stock_res: dict,
    entry_buffer_atr: float = 0.25,
    stop_loss_atr: float = 1.8,
    target_1_atr: float = 2.2,
    target_2_atr: float = 3.8,
) -> dict:
    """Final live-trading entry gate.

    Only the best structures are allowed to become EXECUTION_READY.
    Everything else is downgraded to WATCHLIST_ENTRY so the scanner stays useful
    without forcing trades that usually degrade expectancy.
    """
    plan = _prev_build_consistent_entry_plan_2(
        stock_res,
        entry_buffer_atr=entry_buffer_atr,
        stop_loss_atr=stop_loss_atr,
        target_1_atr=target_1_atr,
        target_2_atr=target_2_atr,
    )
    if not isinstance(plan, dict) or not plan:
        return plan

    try:
        kind = str(plan.get("setup_kind", "None") or "None").strip().upper()
        allowed = {"PULLBACK", "UNICORN", "SNIPER"}
        weekly_ok = bool(stock_res.get("weekly_mtf_ok", plan.get("weekly_mtf_ok", False)))
        tradeability_ok = bool(stock_res.get("tradeability_gate_ok", plan.get("tradeability_ok", False)))

        entry_price = _safe_float(plan.get("entry_price_plan"), np.nan)
        stop_price = _safe_float(plan.get("stop_loss_plan"), np.nan)
        target_1 = _safe_float(plan.get("target_1"), np.nan)
        target_2 = _safe_float(plan.get("target_2"), np.nan)
        rr1 = _safe_float(plan.get("risk_reward_1"), np.nan)
        rr2 = _safe_float(plan.get("risk_reward_2"), np.nan)
        fill = _safe_float(plan.get("setup_fill_probability"), np.nan)
        dist = _safe_float(plan.get("distance_to_entry_atr"), np.nan)
        tradeability = _safe_float(stock_res.get("tradeability_score"), np.nan)
        base_score = _safe_float(stock_res.get("score"), np.nan)
        structure = _safe_float(plan.get("structure_confluence"), np.nan)

        risk_pct = np.nan
        if np.isfinite(entry_price) and np.isfinite(stop_price) and entry_price > 0 and stop_price >= 0 and entry_price > stop_price:
            risk_pct = float((entry_price - stop_price) / entry_price)
        plan["risk_pct_of_entry"] = risk_pct
        plan["weekly_mtf_ok"] = weekly_ok
        plan["tradeability_gate_ok"] = tradeability_ok

        if np.isfinite(base_score):
            base = float(base_score)
        else:
            base = 50.0
        if np.isfinite(tradeability):
            trade = float(tradeability)
        else:
            trade = 50.0
        if np.isfinite(fill):
            fill_score = float(fill)
        else:
            fill_score = 50.0
        rr_component = 50.0
        if np.isfinite(rr2):
            rr_component = float(np.clip((rr2 - 1.50) / 2.50 * 100.0, 0.0, 100.0))
        rr1_component = 50.0
        if np.isfinite(rr1):
            rr1_component = float(np.clip((rr1 - 1.20) / 1.80 * 100.0, 0.0, 100.0))
        structure_component = 50.0
        if np.isfinite(structure):
            structure_component = float(np.clip(structure * 3.0, 0.0, 100.0))
        weekly_component = 72.0 if weekly_ok else 35.0
        quality = (
            0.30 * base
            + 0.18 * trade
            + 0.18 * fill_score
            + 0.16 * rr_component
            + 0.08 * rr1_component
            + 0.05 * structure_component
            + 0.05 * weekly_component
        )
        if kind == "PULLBACK":
            quality += 3.0
        elif kind == "SNIPER":
            quality += 5.0
        elif kind == "UNICORN":
            quality += 4.0
        if not tradeability_ok:
            quality -= 6.0
        if not weekly_ok:
            quality -= 8.0
        if np.isfinite(dist):
            quality -= max(0.0, dist - 0.75) * 6.0
        if np.isfinite(risk_pct):
            quality -= max(0.0, risk_pct - 0.045) * 250.0
        plan["quality_score"] = float(np.clip(quality, 0.0, 100.0))

        hard_reject = False
        if kind not in allowed:
            hard_reject = True
        if not tradeability_ok or not weekly_ok:
            hard_reject = True
        if np.isfinite(rr1) and rr1 < (1.95 if kind == "PULLBACK" else 2.10):
            hard_reject = True
        if np.isfinite(rr2) and rr2 < (3.05 if kind == "PULLBACK" else 3.30):
            hard_reject = True
        if np.isfinite(fill) and fill < (62.0 if kind == "PULLBACK" else 65.0):
            hard_reject = True
        if np.isfinite(plan["quality_score"]) and plan["quality_score"] < (74.0 if kind == "PULLBACK" else 78.0):
            hard_reject = True
        if np.isfinite(dist) and dist > (1.10 if kind == "PULLBACK" else 0.95):
            hard_reject = True
        if np.isfinite(risk_pct) and risk_pct > (0.050 if kind == "PULLBACK" else 0.045):
            hard_reject = True

        if hard_reject:
            # Keep the signal visible for review, but do not present it as executable.
            if kind in allowed:
                plan["entry_candidate_label"] = "WATCHLIST_ENTRY"
                plan["execution_ready"] = False
            else:
                plan["entry_candidate_label"] = "NONE"
                plan["execution_ready"] = False
        else:
            plan["entry_candidate_label"] = "EXECUTION_READY"
            plan["execution_ready"] = True
            plan["quality_score"] = float(np.clip(max(plan["quality_score"], 80.0 if kind == "SNIPER" else 77.0), 0.0, 100.0))

        # If T2 is too close to T1, extend it minimally so RR2 remains meaningful.
        if np.isfinite(rr1) and np.isfinite(rr2) and rr2 < rr1 + 0.80:
            last_obj = stock_res.get("last")
            atr_guess = _safe_float(last_obj.get("ATR14") if hasattr(last_obj, "get") else np.nan, np.nan)
            if not np.isfinite(atr_guess) or atr_guess <= 0:
                atr_guess = _safe_float(stock_res.get("ATR14"), np.nan)
            if not np.isfinite(atr_guess) or atr_guess <= 0:
                atr_guess = 1.0
            t1 = _safe_float(plan.get("target_1"), np.nan)
            if np.isfinite(t1):
                plan["target_2"] = float(t1 + max(1.1 * atr_guess, 1.0))
    except Exception:
        pass

    return plan
