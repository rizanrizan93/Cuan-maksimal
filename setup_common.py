
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, hilbert, periodogram


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, (list, tuple, set, dict)):
            return float(default)
        v = float(value)
        if np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return float(default)


def _safe_text(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default
    except Exception:
        return default


def _min_finite(values, default=np.nan):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(min(finite)) if finite else default


def _max_finite(values, default=np.nan):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(max(finite)) if finite else default


def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def zlema(series: pd.Series, period: int = 20) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lag = max(int((period - 1) / 2), 1)
    adj = s + (s - s.shift(lag))
    return adj.ewm(span=period, adjust=False).mean()


def highpass_filter(series: pd.Series, period: int = 48) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.dropna().size < 10:
        return s.copy()
    trend = zlema(s, max(int(period), 5))
    return s - trend


def linear_forecast_pad(series: pd.Series, length: int = 10) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.dropna().size < 3:
        return s.copy()
    x = np.arange(len(s))
    valid = np.isfinite(s.to_numpy(dtype=float))
    if valid.sum() < 3:
        return s.copy()
    coef = np.polyfit(x[valid], s.to_numpy(dtype=float)[valid], 1)
    pad_x = np.arange(len(s) + max(int(length), 0))
    forecast = np.polyval(coef, pad_x)
    return pd.Series(forecast[: len(s)], index=s.index)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = pd.to_numeric(series, errors="coerce").diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ = ema(series, fast)
    slow_ = ema(series, slow)
    line = fast_ - slow_
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_smoothed = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_smoothed
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_smoothed
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = pd.to_numeric(series, errors="coerce").rolling(window).mean()
    std = pd.to_numeric(series, errors="coerce").rolling(window).std(ddof=0)
    return mid, mid + n_std * std, mid - n_std * std


def obv(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    direction = np.sign(close.diff()).fillna(0.0)
    direction[direction == 0] = 0.0
    return (direction * vol).cumsum()


def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    denom = (high - low).replace(0, np.nan)
    mfm = (((close - low) - (high - close)) / denom).clip(-1, 1)
    mfv = mfm * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    positive = raw_mf.where(tp > tp.shift(1), 0.0)
    negative = raw_mf.where(tp < tp.shift(1), 0.0)
    pmf = positive.rolling(period).sum()
    nmf = negative.rolling(period).sum().replace(0, np.nan)
    mr = pmf / nmf
    return 100 - (100 / (1 + mr))


def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
    low_min = pd.to_numeric(df["Low"], errors="coerce").rolling(k_period).min()
    high_max = pd.to_numeric(df["High"], errors="coerce").rolling(k_period).max()
    close = pd.to_numeric(df["Close"], errors="coerce")
    k = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    return k, d


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (pd.to_numeric(df["High"], errors="coerce") + pd.to_numeric(df["Low"], errors="coerce") + pd.to_numeric(df["Close"], errors="coerce")) / 3.0
    sma = tp.rolling(period).mean()
    md = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * md.replace(0, np.nan))


def rate_of_change(series: pd.Series, period: int = 12) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return (s / s.shift(period) - 1.0) * 100.0


def _ensure_technical_columns(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if out.empty:
        return out
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)
    out["RSI14"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(out["Close"])
    out["ATR14"] = atr(out, 14)
    out["ADX14"] = adx(out, 14)
    out["BB_MID"], out["BB_UPPER"], out["BB_LOWER"] = bollinger(out["Close"], 20, 2.0)
    out["VOL_SMA20"] = out["Volume"].rolling(20).mean()
    out["REL_VOL"] = out["Volume"] / out["VOL_SMA20"].replace(0, np.nan)
    out["OBV"] = obv(out)
    out["OBV_SMA10"] = out["OBV"].rolling(10).mean()
    out["OBV_SLOPE10"] = out["OBV"] - out["OBV"].shift(10)
    out["CMF20"] = chaikin_money_flow(out, 20)
    out["MFI14"] = money_flow_index(out, 14)
    out["STOCH_K"], out["STOCH_D"] = stochastic_oscillator(out, 14, 3, 3)
    out["CCI20"] = cci(out, 20)
    out["ROC12"] = rate_of_change(out["Close"], 12)
    return out


def _bar_age(index: pd.Index, current_pos: int, anchor_pos: int) -> tuple[int, float]:
    age_bars = max(int(current_pos - anchor_pos), 0)
    try:
        anchor_ts = pd.Timestamp(index[anchor_pos])
        current_ts = pd.Timestamp(index[current_pos])
        age_days = float(max((current_ts - anchor_ts).days, 0))
    except Exception:
        age_days = float(age_bars)
    return age_bars, age_days


def _entry_price_from_zone(low: float, high: float, bias: float = 0.5) -> float:
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high):
        return np.nan
    if high < low:
        low, high = high, low
    bias = float(min(max(bias, 0.0), 1.0))
    return float(low + (high - low) * bias)


def _nearest_level_above(levels: pd.Series | np.ndarray | list, price: float) -> float:
    vals = [float(v) for v in list(levels) if v is not None and np.isfinite(v) and float(v) > price]
    return float(min(vals)) if vals else np.nan


def _nearest_level_below(levels: pd.Series | np.ndarray | list, price: float) -> float:
    vals = [float(v) for v in list(levels) if v is not None and np.isfinite(v) and float(v) < price]
    return float(max(vals)) if vals else np.nan


def _rr(entry: float, stop: float, target: float) -> float:
    risk = max(float(entry) - float(stop), 1e-9)
    return float((float(target) - float(entry)) / risk)


def _pivot_series(d: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[pd.Series, pd.Series]:
    highs = pd.to_numeric(d["High"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(d["Low"], errors="coerce").to_numpy(dtype=float)
    idx_high = argrelextrema(highs, np.greater_equal, order=max(int(left), 1))[0]
    idx_low = argrelextrema(lows, np.less_equal, order=max(int(left), 1))[0]
    ph = pd.Series(False, index=d.index)
    pl = pd.Series(False, index=d.index)
    ph.iloc[idx_high] = True
    pl.iloc[idx_low] = True
    # apply right-side confirmation by removing the last few bars
    if right > 0:
        ph.iloc[-right:] = False
        pl.iloc[-right:] = False
    return ph, pl


def compute_relative_strength(stock_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    s = pd.to_numeric(stock_close, errors="coerce").astype(float)
    b = pd.to_numeric(bench_close, errors="coerce").astype(float)
    ratio = (s / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return ratio / ratio.rolling(50).mean() * 100.0


def format_score_delta(delta: float) -> str:
    try:
        d = float(delta)
    except Exception:
        return "n/a"
    if not np.isfinite(d):
        return "n/a"
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}"


def score_to_grade(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        return "n/a"
    if not np.isfinite(s):
        return "n/a"
    if s >= 90:
        return "A+"
    if s >= 80:
        return "A"
    if s >= 70:
        return "B+"
    if s >= 60:
        return "B"
    if s >= 50:
        return "C"
    return "D"


def _phase_from_trend(d: pd.DataFrame) -> tuple[str, float]:
    last = d.iloc[-1]
    close = _safe_float(last.get("Close"), np.nan)
    ema20_v = _safe_float(last.get("EMA20"), np.nan)
    ema50_v = _safe_float(last.get("EMA50"), np.nan)
    ema200_v = _safe_float(last.get("EMA200"), np.nan)
    adx_v = _safe_float(last.get("ADX14"), np.nan)
    if not np.isfinite(close):
        return "Unknown", 0.0
    if np.isfinite(ema20_v) and np.isfinite(ema50_v) and np.isfinite(ema200_v):
        if close > ema20_v > ema50_v > ema200_v:
            return "Markup", min(1.0, 0.6 + (adx_v / 100.0 if np.isfinite(adx_v) else 0.0))
        if close < ema20_v < ema50_v < ema200_v:
            return "Markdown", min(1.0, 0.6 + (adx_v / 100.0 if np.isfinite(adx_v) else 0.0))
        if close > ema20_v and ema20_v >= ema50_v:
            return "Early Markup", 0.75
        if close < ema20_v and ema20_v <= ema50_v:
            return "Early Markdown", 0.75
    return "Sideways", 0.45


def compute_cycle_features(close: pd.Series) -> tuple[int, int, bool, dict]:
    series = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    n = len(series)
    if n < 30:
        return 20, 999, False, {
            "dominant_period": 20,
            "cycle_reliability": 0.0,
            "time_to_next_bottom": 999,
            "time_to_next_top": np.nan,
            "phase_age_bars": np.nan,
            "phase_age_pct": np.nan,
            "cycle_gate_reason": "insufficient_history",
        }

    x = series.to_numpy(dtype=float)
    x = np.log(np.clip(x, 1e-9, None))
    x = x - np.nanmean(x)
    freqs, power = periodogram(x, scaling="spectrum")
    mask = (freqs > 0) & np.isfinite(power)
    freqs = freqs[mask]
    power = power[mask]
    if len(freqs) == 0:
        dom = 20
        rel = 0.0
    else:
        periods = np.where(freqs > 0, 1.0 / freqs, np.nan)
        valid = np.isfinite(periods) & (periods >= 5) & (periods <= max(5, n // 2))
        if valid.any():
            periods = periods[valid]
            power = power[valid]
            dom = int(round(float(periods[np.argmax(power)])))
            rel = float(np.clip(np.nanmax(power) / (np.nanmean(power) + 1e-9), 0.0, 1.0))
        else:
            dom = 20
            rel = 0.0

    dom = int(np.clip(dom, 8, max(20, n // 2)))
    phase_age = int(n % max(dom, 1))
    time_to_bottom = int(max(dom - phase_age, 0))
    time_to_top = int(max(dom // 2 - phase_age, 0))
    phase_age_pct = float((phase_age / max(dom, 1)) * 100.0)

    if rel >= 0.7:
        gate_ok = True
        reason = "cycle_reliable"
    elif rel >= 0.45:
        gate_ok = True
        reason = "cycle_moderate"
    else:
        gate_ok = False
        reason = "cycle_weak"

    return dom, time_to_bottom, gate_ok, {
        "dominant_period": dom,
        "cycle_reliability": float(np.clip(rel, 0.0, 1.0)),
        "time_to_next_bottom": time_to_bottom,
        "time_to_next_top": time_to_top,
        "phase_age_bars": phase_age,
        "phase_age_pct": phase_age_pct,
        "cycle_gate_reason": reason,
    }


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
    if bench_df is None or bench_df.empty or len(bench_df) < 40:
        return neutral

    d = _ensure_technical_columns(bench_df.copy())
    if d.empty:
        return neutral

    last = d.iloc[-1]
    phase, phase_conf = _phase_from_trend(d)
    dom, time_to_bottom, cycle_ok, cycle_info = compute_cycle_features(d["Close"])
    adx_v = _safe_float(last.get("ADX14"), np.nan)
    close = _safe_float(last.get("Close"), np.nan)
    ema20_v = _safe_float(last.get("EMA20"), np.nan)
    ema50_v = _safe_float(last.get("EMA50"), np.nan)
    ema200_v = _safe_float(last.get("EMA200"), np.nan)

    if np.isfinite(close) and np.isfinite(ema20_v) and np.isfinite(ema50_v) and np.isfinite(ema200_v):
        if close > ema20_v > ema50_v > ema200_v:
            regime = "BULL"
            regime_conf = 0.85
            reason = "Price above stacked averages"
            multiplier = 1.10 if (np.isfinite(adx_v) and adx_v >= 25) else 1.05
        elif close < ema20_v < ema50_v < ema200_v:
            regime = "BEAR"
            regime_conf = 0.85
            reason = "Price below stacked averages"
            multiplier = 0.90
        elif close > ema20_v and ema20_v >= ema50_v:
            regime = "UPTREND"
            regime_conf = 0.70
            reason = "Short trend positive"
            multiplier = 1.05
        elif close < ema20_v and ema20_v <= ema50_v:
            regime = "DOWNTREND"
            regime_conf = 0.70
            reason = "Short trend negative"
            multiplier = 0.95
        else:
            regime = "SIDEWAYS"
            regime_conf = 0.55
            reason = "Mixed structure"
            multiplier = 1.0
    else:
        regime = "SIDEWAYS"
        regime_conf = 0.5
        reason = "Insufficient benchmark structure"
        multiplier = 1.0

    macro_score = 50.0
    if regime == "BULL":
        macro_score = 72.0
    elif regime == "UPTREND":
        macro_score = 63.0
    elif regime == "SIDEWAYS":
        macro_score = 50.0
    elif regime == "DOWNTREND":
        macro_score = 42.0
    elif regime == "BEAR":
        macro_score = 30.0

    if np.isfinite(adx_v):
        macro_score += min(max((adx_v - 18) * 1.2, -10.0), 12.0)

    gate_ok = bool(regime not in {"BEAR"})
    gate_reason = "OK" if gate_ok else "Benchmark regime bearish"

    return {
        "benchmark_symbol": benchmark_symbol,
        "macro_phase": phase,
        "macro_phase_confidence": float(phase_conf),
        "macro_period": int(dom),
        "macro_time_to_bottom": int(time_to_bottom),
        "macro_time_to_top": int(cycle_info.get("time_to_next_top", np.nan)) if cycle_info else np.nan,
        "macro_phase_age_bars": cycle_info.get("phase_age_bars", np.nan),
        "macro_phase_age_pct": cycle_info.get("phase_age_pct", np.nan),
        "macro_cycle_reliability": cycle_info.get("cycle_reliability", 0.0),
        "macro_cycle_gate_reason": cycle_info.get("cycle_gate_reason", "unknown"),
        "macro_score": float(np.clip(macro_score, 0.0, 100.0)),
        "macro_gate_ok": gate_ok,
        "macro_gate_reason": gate_reason,
        "macro_multiplier": float(multiplier),
        "market_regime": regime,
        "market_regime_confidence": float(regime_conf),
        "market_regime_reason": reason,
        "cycle_tuple": (dom, time_to_bottom, cycle_ok, cycle_info),
        "benchmark_df": d,
    }


def compute_institutional_forward_score(
    symbol: str,
    price_df: pd.DataFrame,
    bench_df: pd.DataFrame | None = None,
    current_fundamental: dict | None = None,
    future_context: dict | None = None,
    technical_context: dict | None = None,
) -> dict:
    d = _ensure_technical_columns(price_df.copy()) if price_df is not None else pd.DataFrame()
    if d.empty:
        return {
            "ifs_score": 50.0,
            "ifs_grade": score_to_grade(50.0),
            "ifs_breakdown": {},
            "ifs_detail": {"reason": "no_price_data", "symbol": symbol},
        }

    last = d.iloc[-1]
    close = _safe_float(last.get("Close"), np.nan)
    ema20_v = _safe_float(last.get("EMA20"), np.nan)
    ema50_v = _safe_float(last.get("EMA50"), np.nan)
    ema200_v = _safe_float(last.get("EMA200"), np.nan)
    rsi_v = _safe_float(last.get("RSI14"), np.nan)
    adx_v = _safe_float(last.get("ADX14"), np.nan)
    cmf_v = _safe_float(last.get("CMF20"), np.nan)
    mfi_v = _safe_float(last.get("MFI14"), np.nan)
    rel_vol = _safe_float(last.get("REL_VOL"), np.nan)

    trend_score = 50.0
    if np.isfinite(close) and np.isfinite(ema20_v) and np.isfinite(ema50_v) and np.isfinite(ema200_v):
        if close > ema20_v > ema50_v > ema200_v:
            trend_score = 88.0
        elif close > ema20_v and ema20_v >= ema50_v:
            trend_score = 74.0
        elif close < ema20_v < ema50_v < ema200_v:
            trend_score = 22.0
        else:
            trend_score = 50.0

    momentum_score = 50.0
    if np.isfinite(rsi_v):
        momentum_score += (rsi_v - 50.0) * 0.9
    if np.isfinite(mfi_v):
        momentum_score += (mfi_v - 50.0) * 0.2

    flow_score = 50.0
    if np.isfinite(cmf_v):
        flow_score += cmf_v * 100.0 * 0.3
    if np.isfinite(rel_vol):
        flow_score += min(max((rel_vol - 1.0) * 12.0, -8.0), 10.0)

    fundamentals = current_fundamental or {}
    future = future_context or {}
    fundamental_score = _safe_float(fundamentals.get("fundamental_score"), np.nan)
    if not np.isfinite(fundamental_score):
        fundamental_score = 50.0
    ifs_forward = 50.0
    if np.isfinite(future.get("expected_eps_growth_next_q", np.nan)):
        ifs_forward += min(max(_safe_float(future.get("expected_eps_growth_next_q"), 0.0), -20.0), 20.0) * 0.6
    if np.isfinite(future.get("expected_revenue_growth_next_q", np.nan)):
        ifs_forward += min(max(_safe_float(future.get("expected_revenue_growth_next_q"), 0.0), -20.0), 20.0) * 0.4
    if np.isfinite(future.get("expected_margin_next_q", np.nan)):
        ifs_forward += (float(future.get("expected_margin_next_q")) - 50.0) * 0.2

    macro = build_macro_liquidity_gate(bench_df, "^JKSE") if bench_df is not None else build_macro_liquidity_gate(pd.DataFrame(), "^JKSE")
    macro_mult = _safe_float(macro.get("macro_multiplier"), 1.0)
    macro_score = _safe_float(macro.get("macro_score"), 50.0)

    ifs_score = (
        0.28 * trend_score
        + 0.18 * momentum_score
        + 0.16 * flow_score
        + 0.14 * fundamental_score
        + 0.12 * ifs_forward
        + 0.12 * macro_score
    )
    ifs_score = float(np.clip(ifs_score * macro_mult, 0.0, 100.0))
    grade = score_to_grade(ifs_score)

    return {
        "ifs_score": ifs_score,
        "ifs_grade": grade,
        "ifs_breakdown": {
            "trend_score": float(trend_score),
            "momentum_score": float(momentum_score),
            "flow_score": float(flow_score),
            "fundamental_score": float(fundamental_score),
            "forward_score": float(ifs_forward),
            "macro_score": float(macro_score),
            "macro_multiplier": float(macro_mult),
        },
        "ifs_detail": {
            "symbol": symbol,
            "close": close,
            "rsi14": rsi_v,
            "adx14": adx_v,
            "cmf20": cmf_v,
            "mfi14": mfi_v,
        },
    }
