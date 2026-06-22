
from __future__ import annotations

import numpy as np
import pandas as pd

from setup_common import (
    _safe_float, _ensure_technical_columns, _bar_age, _entry_price_from_zone,
    _min_finite, _max_finite, _rr
)


def _latest_bullish_fvg(d: pd.DataFrame) -> tuple[int | None, float, float]:
    """Return (position, bottom, top) for the latest bullish FVG.
    bottom < top if valid, otherwise NaN values.
    """
    if len(d) < 3:
        return None, np.nan, np.nan
    for pos in range(len(d) - 1, 1, -1):
        hi_2 = _safe_float(d["High"].iloc[pos - 2], np.nan)
        lo_0 = _safe_float(d["Low"].iloc[pos], np.nan)
        close_0 = _safe_float(d["Close"].iloc[pos], np.nan)
        open_0 = _safe_float(d["Open"].iloc[pos], np.nan)
        if np.isfinite(hi_2) and np.isfinite(lo_0) and lo_0 > hi_2 and close_0 > open_0:
            return pos, hi_2, lo_0
    return None, np.nan, np.nan


def evaluate_unicorn_setup(d: pd.DataFrame, fresh_bars: int = 3, max_age_bars: int = 20) -> dict:
    out = {
        "index": None,
        "position": np.nan,
        "age_bars": np.nan,
        "age_days": np.nan,
        "valid": False,
        "confirmed": False,
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
        "fvg_top": np.nan,
        "fvg_bottom": np.nan,
        "breaker_top": np.nan,
        "breaker_bottom": np.nan,
        "sweep_low": np.nan,
        "setup_kind": "Unicorn",
        "setup_variant": "Base",
        "entry_trigger": "Liquidity_Sweep_MSS_FVG_Retest",
        "lifecycle_stage": "NO_SETUP",
        "sniper_valid": False,
        "sniper_status": "none",
    }
    if d is None or getattr(d, "empty", True):
        return out

    d = _ensure_technical_columns(d.copy())
    if d.empty:
        return out

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(d.columns):
        return out

    last = d.iloc[-1]
    close = _safe_float(last.get("Close"), np.nan)
    open_ = _safe_float(last.get("Open"), np.nan)
    low = _safe_float(last.get("Low"), np.nan)
    atr_v = _safe_float(last.get("ATR14"), np.nan)
    if not np.isfinite(close) or close <= 0:
        return out
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close * 0.02, 1.0)

    piv_low = d.get("Pivot_Low_Confirmed")
    piv_high = d.get("Pivot_High_Confirmed")
    if piv_low is None:
        piv_low = pd.Series(False, index=d.index)
    if piv_high is None:
        piv_high = pd.Series(False, index=d.index)

    pivot_lows = d.loc[piv_low.fillna(False), "Low"].dropna()
    pivot_highs = d.loc[piv_high.fillna(False), "High"].dropna()
    if len(pivot_lows) < 1 or len(pivot_highs) < 1:
        out["reason"] = "not_enough_pivots"
        return out

    last_pivot_low = float(pivot_lows.iloc[-1])
    prev_pivot_low = float(pivot_lows.iloc[-2]) if len(pivot_lows) >= 2 else last_pivot_low
    last_pivot_high = float(pivot_highs.iloc[-1])

    fvg_pos, fvg_bottom, fvg_top = _latest_bullish_fvg(d)
    if fvg_pos is None or not np.isfinite(fvg_bottom) or not np.isfinite(fvg_top):
        out["reason"] = "no_fvg"
        return out

    # Liquidity sweep + MSS + fresh FVG is the core Unicorn concept.
    sweep_low = float(min(d["Low"].tail(6).min(), last_pivot_low))
    breaker_bottom = float(min(last_pivot_low, prev_pivot_low))
    breaker_top = float(max(last_pivot_high, close))
    sweep_confirmed = bool(low <= prev_pivot_low - atr_v * 0.10)
    mss_confirmed = bool(close > last_pivot_high and close > open_)
    fvg_fresh = bool((len(d) - 1 - fvg_pos) <= max_age_bars)

    valid = bool(sweep_confirmed and mss_confirmed and fvg_fresh)
    sniper_valid = bool(valid and close <= fvg_top and close >= fvg_bottom and (len(d) - 1 - fvg_pos) <= fresh_bars)

    pos = int(fvg_pos)
    age_bars, age_days = _bar_age(d.index, len(d) - 1, pos)

    if valid:
        status = "fresh" if age_bars <= fresh_bars else "valid"
        reason = "unicorn_valid"
    else:
        status = "invalid"
        reason = "unicorn_conditions_not_met"

    entry_zone_low = float(max(0.0, fvg_bottom - atr_v * 0.18))
    entry_zone_high = float(max(entry_zone_low, min(fvg_top + atr_v * 0.16, close + atr_v * 0.08)))
    entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=0.30)
    stop_price = float(max(min(sweep_low, breaker_bottom, fvg_bottom) - atr_v * 0.10, 0.0))
    target_1 = float(max(last_pivot_high, fvg_top + atr_v * 1.3, entry_price + atr_v * 1.8))
    target_2 = float(max(target_1 + atr_v * 0.9, entry_price + atr_v * 3.2))
    invalidation = float(stop_price)

    out.update({
        "index": d.index[pos],
        "position": int(pos),
        "age_bars": age_bars,
        "age_days": age_days,
        "valid": valid,
        "confirmed": valid,
        "status": status,
        "reason": reason,
        "entry_zone_low": entry_zone_low,
        "entry_zone_high": entry_zone_high,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_1": target_1,
        "target_2": target_2,
        "invalidation_level": invalidation,
        "support_anchor": sweep_low,
        "resistance_anchor": last_pivot_high,
        "fvg_top": float(fvg_top),
        "fvg_bottom": float(fvg_bottom),
        "breaker_top": breaker_top,
        "breaker_bottom": breaker_bottom,
        "sweep_low": sweep_low,
        "setup_kind": "Unicorn",
        "setup_variant": "Sniper" if sniper_valid else "Base",
        "entry_trigger": "Liquidity_Sweep_MSS_FVG_Retest",
        "lifecycle_stage": "ENTRY_ZONE" if valid else "WATCHLIST",
        "sniper_valid": sniper_valid,
        "sniper_status": "valid" if sniper_valid else ("not_confirmed" if valid else "invalid"),
        "risk_reward_1": _rr(entry_price, stop_price, target_1),
        "risk_reward_2": _rr(entry_price, stop_price, target_2),
    })
    return out
