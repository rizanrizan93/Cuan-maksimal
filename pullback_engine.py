
from __future__ import annotations

import numpy as np
import pandas as pd

from setup_common import (
    _safe_float, _ensure_technical_columns, _bar_age, _entry_price_from_zone,
    _min_finite, _max_finite, _rr
)


def evaluate_pullback_setup(d: pd.DataFrame, fresh_bars: int = 3, max_age_bars: int = 18) -> dict:
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
        "pullback_reference": np.nan,
        "setup_kind": "Pullback",
        "setup_variant": "Continuation",
        "entry_trigger": "Trend_Pullback_Retest",
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
    ema200 = _safe_float(last.get("EMA200"), np.nan)
    if not np.isfinite(close) or close <= 0:
        return out
    if not np.isfinite(atr_v) or atr_v <= 0:
        atr_v = max(close * 0.02, 1.0)

    trend_up = bool(
        (np.isfinite(ema20) and np.isfinite(ema50) and close > ema20 and ema20 >= ema50)
        or (np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(ema200) and close > ema20 and ema20 > ema50 > ema200)
    )
    piv_high = d.get("Pivot_High_Confirmed")
    piv_low = d.get("Pivot_Low_Confirmed")
    if piv_high is None:
        piv_high = pd.Series(False, index=d.index)
    if piv_low is None:
        piv_low = pd.Series(False, index=d.index)

    pivot_highs = d.loc[piv_high.fillna(False), "High"].dropna()
    pivot_lows = d.loc[piv_low.fillna(False), "Low"].dropna()
    if len(pivot_highs) < 1 or len(pivot_lows) < 1:
        out["reason"] = "not_enough_pivots"
        return out

    last_pivot_high = float(pivot_highs.iloc[-1])
    last_pivot_low = float(pivot_lows.iloc[-1])
    prev_pivot_low = float(pivot_lows.iloc[-2]) if len(pivot_lows) >= 2 else last_pivot_low

    support_anchor = _min_finite([ema20, ema50, last_pivot_low, prev_pivot_low], default=np.nan)
    if not np.isfinite(support_anchor):
        support_anchor = float(d["Low"].tail(12).min())

    resistance_anchor = _max_finite([last_pivot_high, float(d["High"].tail(20).max())], default=np.nan)
    if not np.isfinite(resistance_anchor):
        resistance_anchor = last_pivot_high

    mitigation_ok = bool(low <= support_anchor + atr_v * 0.35)
    reclaim_ok = bool(close > support_anchor and close > open_)
    structure_ok = bool(np.isfinite(last_pivot_high) and last_pivot_high >= prev_pivot_low)
    valid = bool(trend_up and mitigation_ok and reclaim_ok and structure_ok)

    pivot_pos = np.flatnonzero(piv_low.fillna(False).to_numpy(dtype=bool))
    pos = int(pivot_pos[-1]) if pivot_pos.size else len(d) - 1
    age_bars, age_days = _bar_age(d.index, len(d) - 1, pos)

    if valid:
        status = "fresh" if age_bars <= fresh_bars else "valid"
        reason = "pullback_continuation_valid"
    else:
        status = "invalid"
        reason = "pullback_conditions_not_met"

    entry_zone_low = float(max(0.0, support_anchor - atr_v * 0.22))
    entry_zone_high = float(max(entry_zone_low, min(support_anchor + atr_v * 0.18, close - atr_v * 0.05)))
    entry_price = _entry_price_from_zone(entry_zone_low, entry_zone_high, bias=0.35)
    stop_price = float(max(min(last_pivot_low, support_anchor) - atr_v * 0.14, 0.0))
    target_1 = float(max(resistance_anchor, entry_price + atr_v * 1.7))
    target_2 = float(max(target_1 + atr_v * 0.9, entry_price + atr_v * 3.0))
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
        "support_anchor": support_anchor,
        "resistance_anchor": resistance_anchor,
        "pullback_reference": support_anchor,
        "setup_kind": "Pullback",
        "setup_variant": "Continuation",
        "entry_trigger": "Trend_Pullback_Retest",
        "lifecycle_stage": "ENTRY_ZONE" if valid else "WATCHLIST",
        "risk_reward_1": _rr(entry_price, stop_price, target_1),
        "risk_reward_2": _rr(entry_price, stop_price, target_2),
    })
    return out
