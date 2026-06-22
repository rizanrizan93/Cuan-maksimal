
from __future__ import annotations

import numpy as np
import pandas as pd

from setup_common import (
    _safe_float, _safe_text, _ensure_technical_columns, _bar_age,
    _entry_price_from_zone, _min_finite, _max_finite, _nearest_level_above,
    _rr
)


def evaluate_breakout_setup(d: pd.DataFrame, fresh_bars: int = 3, max_age_bars: int = 20) -> dict:
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
        "breakout_reference": np.nan,
        "breakout_range_height": np.nan,
        "setup_kind": "Breakout",
        "setup_variant": "Retest",
        "entry_trigger": "Breakout_Retest",
        "lifecycle_stage": "NO_SETUP",
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
    ema20 = _safe_float(last.get("EMA20"), np.nan)
    ema50 = _safe_float(last.get("EMA50"), np.nan)

    if not np.isfinite(close) or close <= 0:
        return out
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close * 0.02, 1.0)

    piv_high = d.get("Pivot_High_Confirmed")
    if piv_high is None:
        piv_high = pd.Series(False, index=d.index)
    pivot_highs = d.loc[piv_high.fillna(False), "High"].dropna()
    if len(pivot_highs) < 2:
        out["reason"] = "not_enough_pivot_highs"
        return out

    breakout_reference = float(pivot_highs.iloc[-1])
    prev_high = float(pivot_highs.iloc[-2])
    recent_low = float(d["Low"].tail(10).min())
    recent_high = float(d["High"].tail(20).max())
    range_height = max(recent_high - recent_low, atr_v)

    breakout_retest_ok = bool(
        np.isfinite(breakout_reference)
        and low <= breakout_reference + atr_v * 0.20
        and close > breakout_reference
        and close > open_
    )
    reclaim_ok = bool(np.isfinite(ema20) and close > ema20 and (not np.isfinite(ema50) or ema20 >= ema50))
    structure_ok = bool(breakout_reference > prev_high)

    valid = bool(breakout_retest_ok and reclaim_ok and structure_ok)

    pivot_pos = np.flatnonzero(piv_high.fillna(False).to_numpy(dtype=bool))
    pos = int(pivot_pos[-1]) if pivot_pos.size else len(d) - 1
    age_bars, age_days = _bar_age(d.index, len(d) - 1, pos)

    if valid:
        status = "fresh" if age_bars <= fresh_bars else "valid"
        reason = "breakout_retest_valid"
    else:
        status = "invalid"
        reason = "breakout_conditions_not_met"

    entry_zone_low = float(max(0.0, breakout_reference - atr_v * 0.18))
    entry_zone_high = float(max(entry_zone_low, min(breakout_reference + atr_v * 0.14, close + atr_v * 0.08)))
    entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=0.28)
    stop_price = float(max(min(breakout_reference, recent_low) - atr_v * 0.12, 0.0))
    target_1 = float(max(recent_high, breakout_reference + range_height * 0.75, entry_price + atr_v * 1.6))
    target_2 = float(max(target_1 + atr_v * 0.9, entry_price + atr_v * 3.0))
    invalidation = float(max(0.0, breakout_reference - atr_v * 0.25))

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
        "support_anchor": recent_low,
        "resistance_anchor": breakout_reference,
        "breakout_reference": breakout_reference,
        "breakout_range_height": range_height,
        "setup_kind": "Breakout",
        "setup_variant": "Retest",
        "entry_trigger": "Breakout_Retest_Above_Former_Resistance",
        "lifecycle_stage": "ENTRY_ZONE" if valid else "WATCHLIST",
        "risk_reward_1": _rr(entry_price, stop_price, target_1),
        "risk_reward_2": _rr(entry_price, stop_price, target_2),
    })
    return out
