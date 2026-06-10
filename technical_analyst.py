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



def _safe_float(value, default=np.nan):
    """Convert a value to float safely, returning default for invalid/non-finite inputs."""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return float(result)

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


def _estimate_trend_lag_bars(
    dominant_period: float,
    composite_confidence: float | None = None,
    cycle_reliability: float | None = None,
) -> int:
    """Estimate the effective lag of the trend view.

    The old implementation tied lag to the historical sample length, which made
    it collapse to the same value across many symbols.  This version scales the
    lag from the estimated dominant cycle so it can vary by stock while still
    remaining bounded and easy to interpret.
    """
    if not np.isfinite(dominant_period):
        return 0

    # A trend view should lag less than half a cycle, but more than a few bars.
    base_lag = float(dominant_period) * 0.40

    comp = float(composite_confidence) if composite_confidence is not None and np.isfinite(composite_confidence) else np.nan
    rel = float(cycle_reliability) if cycle_reliability is not None and np.isfinite(cycle_reliability) else np.nan

    if np.isfinite(comp):
        # Stronger cycle agreement permits a slightly faster estimate.
        base_lag *= float(np.interp(comp, [0.0, 100.0], [1.08, 0.92]))

    if np.isfinite(rel):
        # More reliable cycles allow a slightly more responsive lag; noisy cycles
        # stay a bit slower to avoid overreacting.
        base_lag *= float(np.interp(rel, [0.0, 100.0], [1.06, 0.94]))

    return int(np.clip(round(base_lag), 3, 30))


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
            "trend_lag_bars": np.nan,
            "trend_lag_source": "n/a",
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

    trend_lag_bars = _estimate_trend_lag_bars(dominant_period, composite_confidence=composite_conf, cycle_reliability=reliability)

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
        "trend_lag_source": "dominant_period",
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
        "macro_tradeability": 0.55,
        "macro_regime_strength": 50.0,
        "macro_regime_tag": "Neutral",
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
    ema20_last = float(last["EMA20"]) if pd.notna(last["EMA20"]) else np.nan
    ema50_last = float(last["EMA50"]) if pd.notna(last["EMA50"]) else np.nan
    ema200_last = float(last["EMA200"]) if pd.notna(last["EMA200"]) else np.nan
    close_last = float(last["Close"]) if pd.notna(last["Close"]) else np.nan

    macro_phase = str(phase_info.get("phase", "Unknown"))
    macro_phase_confidence = float(phase_info.get("phase_confidence", 0.0))
    macro_time_to_top = int(cycle_info.get("time_to_next_top", np.nan)) if pd.notna(cycle_info.get("time_to_next_top", np.nan)) else np.nan
    macro_phase_age_bars = int(cycle_info.get("phase_age_bars", np.nan)) if pd.notna(cycle_info.get("phase_age_bars", np.nan)) else np.nan
    macro_phase_age_pct = float(cycle_info.get("phase_age_pct", np.nan)) if pd.notna(cycle_info.get("phase_age_pct", np.nan)) else np.nan

    macro_score = 100.0
    reasons: list[str] = []
    regime_tag = "Neutral"

    if macro_phase in {"Early Accumulation", "Accumulation", "Late Accumulation"}:
        macro_score += 4.0
        regime_tag = "Accumulation"
    elif macro_phase in {"Early Markup", "Markup"}:
        macro_score += 6.0
        regime_tag = "Markup"
    elif macro_phase == "Late Markup":
        macro_score -= 8.0
        regime_tag = "Late Markup"
    elif macro_phase == "Distribution":
        macro_score -= 22.0
        regime_tag = "Distribution"
        reasons.append("IHSG phase Distribution")
    elif macro_phase == "Markdown":
        macro_score -= 42.0
        regime_tag = "Markdown"
        reasons.append("IHSG phase Markdown")

    if np.isfinite(close_last) and np.isfinite(ema50_last) and close_last < ema50_last:
        macro_score -= 8.0
        reasons.append("IHSG below EMA50")
    if np.isfinite(ema50_last) and np.isfinite(ema200_last) and ema50_last < ema200_last:
        macro_score -= 8.0
        reasons.append("IHSG EMA50 below EMA200")
    if np.isfinite(adx_last) and adx_last > 38:
        macro_score -= 20.0
        reasons.append(f"IHSG ADX {adx_last:.0f} > 38")
    elif np.isfinite(adx_last) and adx_last > 28:
        macro_score -= 10.0
        reasons.append(f"IHSG ADX {adx_last:.0f} elevated")

    if np.isfinite(cycle_reliability) and cycle_reliability < 45:
        macro_score -= 15.0
        reasons.append(f"IHSG CycleRel {cycle_reliability:.0f} < 45")
    elif np.isfinite(cycle_reliability) and cycle_reliability < 60:
        macro_score -= 8.0
        reasons.append(f"IHSG CycleRel {cycle_reliability:.0f} moderate")

    macro_score = float(np.clip(macro_score, 0.0, 100.0))
    macro_regime_strength = float(np.clip(macro_score, 0.0, 100.0))
    macro_tradeability = float(np.clip((macro_score / 100.0) * 0.45 + 0.35, 0.35, 0.95))

    bullish_phase = macro_phase in {"Early Accumulation", "Accumulation", "Late Accumulation", "Early Markup", "Markup", "Late Markup"}
    severe_weakness = (macro_phase == "Markdown") or (np.isfinite(adx_last) and adx_last > 38 and macro_phase in {"Distribution", "Markdown"})
    macro_gate_ok = bullish_phase and (not severe_weakness) and (not (np.isfinite(cycle_reliability) and cycle_reliability < 45)) and (macro_score >= 55.0)
    macro_gate_reason = "OK" if macro_gate_ok else ", ".join(reasons) if reasons else "Macro gate off"

    if macro_gate_ok:
        macro_multiplier = 1.0 if macro_score >= 75 else 0.96 if macro_score >= 65 else 0.92
    else:
        macro_multiplier = 0.84 if macro_score >= 70 else 0.74 if macro_score >= 55 else 0.62 if macro_score >= 40 else 0.52

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
        "macro_tradeability": macro_tradeability,
        "macro_regime_strength": macro_regime_strength,
        "macro_regime_tag": regime_tag,
        "cycle_tuple": cycle_tuple,
        "benchmark_df": d,
    }


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

    # Final score is intentionally blended from two different views:
    # structure (what the chart is doing) and participation (how strong the flow is).
    if flow_used:
        final_score = float(np.clip(
            (market_structure_score * 0.54)
            + (smart_money_score * 0.24)
            + (float(flow_val) * 0.22),
            0.0,
            100.0,
        ))
    else:
        final_score = float(np.clip(
            (market_structure_score * 0.64)
            + (smart_money_score * 0.36),
            0.0,
            100.0,
        ))

    if np.isfinite(macro_multiplier):
        final_score = float(np.clip(final_score * macro_multiplier, 0.0, 100.0))

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
            final_score = float(np.clip((final_score * 0.74) + (future_fundamental_score * 0.26), 0.0, 100.0))

    liquidity_ok = (d["Volume"].tail(20).mean() >= min_avg_volume) and (min_price <= float(last["Close"]) <= max_price)
    trend_ok = (last["Close"] > last["EMA20"]) and (last["EMA50"] > last["EMA200"])
    future_quality_ok = (not np.isfinite(future_fundamental_score)) or (future_fundamental_score >= 60.0)
    future_quality_strong = np.isfinite(future_fundamental_score) and (future_fundamental_score >= 68.0)
    strong_confluence = bool(future_quality_ok and smart_money_score >= 56.0 and cycle_ok)

    if mode == "Conservative":
        buy_threshold, strong_threshold = 78, 86
    elif mode == "Balanced":
        buy_threshold, strong_threshold = 68, 81
    else:
        buy_threshold, strong_threshold = 58, 72

    trend_extension_ok = bool(
        liquidity_ok
        and macro_gate_ok
        and not trend_ok
        and (unicorn_setup_status == "ENTRY" or unicorn_sniper_status == "ENTRY")
        and future_quality_strong
        and strong_confluence
        and reversal_score >= 2
        and smart_money_score >= 62.0
        and (final_score >= buy_threshold + 7)
    )

    # Unicorn setup stays the highest-quality path.
    # Sniper retests can still qualify, and a small number of high-conviction
    # counter-trend recoveries are allowed only when the score stack is strong.
    if not macro_gate_ok:
        if liquidity_ok and trend_ok and unicorn_setup_status == "ENTRY" and unicorn_sniper_status == "ENTRY" and (final_score >= strong_threshold + 4) and strong_confluence and future_quality_strong:
            decision = "STRONG BUY"
        elif liquidity_ok and trend_ok and ((unicorn_setup_status == "ENTRY") or (unicorn_sniper_status == "ENTRY")) and (final_score >= buy_threshold + 4) and strong_confluence:
            decision = "BUY"
        elif liquidity_ok and ((unicorn_setup_status == "ENTRY") or (unicorn_sniper_status == "ENTRY")) and future_quality_ok and (final_score >= buy_threshold + 2):
            decision = "WATCHLIST"
        elif liquidity_ok and smc_confirmed and (final_score >= buy_threshold - 4):
            decision = "WATCHLIST"
        else:
            decision = "AVOID"
    else:
        if liquidity_ok and trend_ok and unicorn_setup_status == "ENTRY" and unicorn_sniper_status == "ENTRY" and (final_score >= strong_threshold) and strong_confluence and future_quality_strong:
            decision = "STRONG BUY"
        elif liquidity_ok and trend_ok and ((unicorn_setup_status == "ENTRY") or (unicorn_sniper_status == "ENTRY")) and (final_score >= buy_threshold) and future_quality_ok:
            decision = "BUY"
        elif trend_extension_ok:
            decision = "BUY"
        elif liquidity_ok and ((unicorn_setup_status == "ENTRY") or (unicorn_sniper_status == "ENTRY")) and future_quality_ok and (final_score >= buy_threshold - 4) and strong_confluence:
            decision = "BUY"
        elif liquidity_ok and trend_ok and smc_confirmed and (final_score >= buy_threshold):
            decision = "WATCHLIST"
        elif liquidity_ok and smc_confirmed and (final_score >= buy_threshold - 6) and strong_confluence:
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

    if decision in {"BUY", "STRONG BUY"}:
        atr_safe = _safe_float(last.get("ATR14"), np.nan)
        close_v = _safe_float(last.get("Close"), np.nan)
        if not np.isfinite(atr_safe) or atr_safe <= 0:
            atr_safe = max(close_v * 0.02, 1.0)

        entry_candidates = [
            float(close_v),
            float(recent_support_ema),
            float(recent_swing_low + atr_safe * 0.25),
        ]
        if np.isfinite(ob_zone):
            entry_candidates.append(float(ob_zone))

        entry_candidates = [v for v in entry_candidates if np.isfinite(v) and v > 0]
        if entry_candidates:
            entry_price = float(np.median(entry_candidates))
            entry_zone_low = max(0.0, float(min(entry_candidates) - atr_safe * 0.15))
            entry_zone_high = max(entry_zone_low, float(max(entry_candidates) + atr_safe * 0.15))
        else:
            entry_price = float(close_v)
            entry_zone_low = max(0.0, entry_price - atr_safe * 0.15)
            entry_zone_high = entry_price + atr_safe * 0.15

        structural_stop = min(
            recent_swing_low - atr_safe * 0.15,
            close_v - atr_safe * stop_loss_atr,
            recent_support_ema - atr_safe * 0.35,
        )
        stop_price = max(min(structural_stop, entry_price - atr_safe * 0.75), 0.0)

        entry_trigger = "Pullback_to_support" if entry_price <= close_v else "Breakout_confirmation"
    else:
        entry_price = np.nan
        stop_price = np.nan
        entry_zone_low = np.nan
        entry_zone_high = np.nan
        entry_trigger = "No_signal"

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
    if not trend_ok:
        notes.append("Struktur_Trend_Bearish")
    if unicorn_setup_status != "ENTRY":
        notes.append("Unicorn_Belum_Entry")
    if unicorn_setup_status == "ENTRY" and unicorn_sniper_status != "ENTRY":
        notes.append("Belum_Sniper")
    if not cycle_ok:
        notes.append("Siklus_Belum_Menguat")
        if cycle_gate_reason:
            notes.append("Cycle_Gated_" + "_".join(cycle_gate_reason).replace(" ", ""))
    if not macro_gate_ok:
        notes.append("Macro_Gated_" + macro_gate_reason.replace(" ", "_"))
    if reversal_score == 0:
        notes.append("Belum_Ada_Reversal_Strong")
    if trend_extension_ok:
        notes.append("Trend_Recovering_With_Strong_Confluence")

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
        "unicorn_setup": unicorn_setup_confirmed,
        "unicorn_sniper": unicorn_sniper_confirmed,
        "unicorn_entry_style": "Sniper" if unicorn_sniper_status == "ENTRY" else ("Basic" if unicorn_setup_status == "ENTRY" else ("Watchlist" if unicorn_setup_status == "WATCHLIST" else "None")),
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
        "stop_price": stop_price,
        "unicorn_setup_confirmed": unicorn_setup_confirmed,
        "unicorn_sniper_confirmed": unicorn_sniper_confirmed,
        "unicorn_fvg_top": unicorn_fvg_top,
        "unicorn_fvg_bottom": unicorn_fvg_bottom,
        "unicorn_breaker_top": unicorn_breaker_top,
        "unicorn_breaker_bottom": unicorn_breaker_bottom,
        "unicorn_sweep_low": unicorn_sweep_low,
        "notes": ",".join(notes) if notes else "SMC_Structure_Clear",
        "df": d,
        "last": last,
    }

def build_entry_plan(
    stock_res: dict,
    entry_buffer_atr: float = 0.25,
    stop_loss_atr: float = 1.8,
    target_1_atr: float = 2.2,
    target_2_atr: float = 3.8,
) -> dict:
    """Build a practical trade plan anchored to ICT Unicorn / Sniper structure."""
    d = stock_res.get("df")
    last = stock_res.get("last")
    decision = str(stock_res.get("decision", "AVOID"))

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
    }

    if d is None or last is None or getattr(d, "empty", True):
        return empty_plan

    unicorn_setup_entry = str(stock_res.get("unicorn_setup_status", "INVALID")).upper() == "ENTRY"
    unicorn_sniper_entry = str(stock_res.get("unicorn_sniper_status", "INVALID")).upper() == "ENTRY"
    if not unicorn_setup_entry and not unicorn_sniper_entry:
        setup_status = str(stock_res.get("unicorn_setup_status", "No Unicorn entry"))
        sniper_status = str(stock_res.get("unicorn_sniper_status", setup_status))
        empty_plan["plan_reason"] = setup_status if setup_status != "INVALID" else sniper_status
        return empty_plan

    try:
        atr_v = _safe_float(last.get("ATR14"), np.nan)
        close = _safe_float(last.get("Close"), np.nan)
        ema20 = _safe_float(last.get("EMA20"), np.nan)
        ema50 = _safe_float(last.get("EMA50"), np.nan)
        ema200 = _safe_float(last.get("EMA200"), np.nan)
        if not np.isfinite(close) or close <= 0:
            return empty_plan

        if not np.isfinite(atr_v) or atr_v <= 0:
            atr_v = max(close * 0.02, 1.0)

        recent_swing_low = float(d["Low"].tail(10).min())
        recent_swing_high = float(d["High"].tail(20).max())
        recent_support = float(min(recent_swing_low, ema20 if np.isfinite(ema20) else recent_swing_low))

        fvg_top = _safe_float(stock_res.get("unicorn_fvg_top"), np.nan)
        fvg_bottom = _safe_float(stock_res.get("unicorn_fvg_bottom"), np.nan)
        breaker_top = _safe_float(stock_res.get("unicorn_breaker_top"), np.nan)
        breaker_bottom = _safe_float(stock_res.get("unicorn_breaker_bottom"), np.nan)
        sweep_low = _safe_float(stock_res.get("unicorn_sweep_low"), np.nan)

        # Prefer the structure already computed in the signal engine.
        entry_candidates = [close, ema20, ema50 if np.isfinite(ema50) else np.nan, recent_support]
        if unicorn_sniper_entry:
            if np.isfinite(fvg_bottom):
                entry_candidates.append(fvg_bottom + atr_v * 0.20)
            if np.isfinite(fvg_top):
                entry_candidates.append(min(fvg_top, close + atr_v * 0.25))
            if np.isfinite(breaker_top):
                entry_candidates.append(breaker_top)
        else:
            if np.isfinite(fvg_bottom):
                entry_candidates.append(fvg_bottom + atr_v * 0.35)
            if np.isfinite(fvg_top):
                entry_candidates.append((fvg_top + fvg_bottom) / 2 if np.isfinite(fvg_bottom) else fvg_top)
            if np.isfinite(breaker_top):
                entry_candidates.append(breaker_top)

        entry_candidates = [v for v in entry_candidates if np.isfinite(v) and v > 0]
        if not entry_candidates:
            entry_candidates = [close]

        entry_price = float(np.median(entry_candidates))
        entry_buffer = max(float(entry_buffer_atr), 0.0) * atr_v

        if unicorn_sniper_entry:
            lower_anchor = max(
                v for v in [fvg_bottom, breaker_bottom, close - atr_v * 0.35] if np.isfinite(v)
            ) if any(np.isfinite(v) for v in [fvg_bottom, breaker_bottom]) else max(0.0, close - atr_v * 0.35)
            upper_anchor = min(
                v for v in [fvg_top, breaker_top, close + atr_v * 0.20] if np.isfinite(v)
            ) if any(np.isfinite(v) for v in [fvg_top, breaker_top]) else close + atr_v * 0.20
            entry_zone_low = max(0.0, float(min(lower_anchor, entry_price) - entry_buffer))
            entry_zone_high = max(entry_zone_low, float(max(upper_anchor, entry_price) + entry_buffer))
            entry_trigger = "Unicorn_Sniper_Retest"
            setup_kind = "Sniper"

            stop_candidates = [close - atr_v * stop_loss_atr]
            if np.isfinite(sweep_low):
                stop_candidates.append(sweep_low - atr_v * 0.10)
            if np.isfinite(breaker_bottom):
                stop_candidates.append(breaker_bottom - atr_v * 0.05)
            if np.isfinite(fvg_bottom):
                stop_candidates.append(fvg_bottom - atr_v * 0.05)
            stop_price = max(min(stop_candidates), 0.0)

            target_1 = float(max(recent_swing_high, entry_price + atr_v * target_1_atr))
            target_2 = float(max(target_1 + atr_v * 0.80, entry_price + atr_v * target_2_atr))
            plan_reason = "Sniper plan using Unicorn retest + EMA stack + structural protection"

        else:
            if np.isfinite(fvg_bottom):
                entry_zone_low = max(0.0, float(fvg_bottom - entry_buffer))
            else:
                entry_zone_low = max(0.0, float(min(entry_price, recent_support) - atr_v * 0.10 - entry_buffer))
            if np.isfinite(fvg_top):
                entry_zone_high = float(max(entry_zone_low, min(fvg_top + entry_buffer, entry_price + atr_v * 0.25 + entry_buffer)))
            else:
                entry_zone_high = float(max(entry_zone_low, entry_price + atr_v * 0.20 + entry_buffer))
            entry_trigger = "Unicorn_FVG_Retest"
            setup_kind = "Basic"

            stop_candidates = [close - atr_v * stop_loss_atr, recent_swing_low - atr_v * 0.15]
            if np.isfinite(sweep_low):
                stop_candidates.append(sweep_low - atr_v * 0.10)
            if np.isfinite(breaker_bottom):
                stop_candidates.append(breaker_bottom - atr_v * 0.05)
            stop_price = max(min(stop_candidates), 0.0)

            target_1 = float(max(recent_swing_high, entry_price + atr_v * target_1_atr))
            target_2 = float(max(target_1 + atr_v * 0.80, entry_price + atr_v * target_2_atr))
            plan_reason = "Basic Unicorn plan using FVG retest and breaker protection"

        if stop_price >= entry_price:
            stop_price = max(entry_price - atr_v * 0.90, 0.0)

        risk_per_share = float(max(entry_price - stop_price, 1e-9))
        rr1 = float((target_1 - entry_price) / risk_per_share)
        rr2 = float((target_2 - entry_price) / risk_per_share)
        upside_t1 = float((target_1 / entry_price - 1.0) * 100.0)
        upside_t2 = float((target_2 / entry_price - 1.0) * 100.0)

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
        }
    except Exception as e:
        empty_plan["plan_reason"] = f"Plan error: {e}"
        return empty_plan



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
            "trend_lag_source": "fallback",
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


def _classify_8_phase_impl(d: pd.DataFrame) -> dict:
    """Classify the market phase using trend, momentum, and flow context."""
    unknown = {
        "phase": "Unknown",
        "phase_confidence": 0.0,
        "phase_rank": 0.0,
        "phase_reason": "insufficient_data",
        "phase_scores": {},
    }

    try:
        if not isinstance(d, pd.DataFrame) or d.empty:
            return unknown

        required = {"EMA20", "EMA50", "EMA200", "RSI14", "ADX14"}
        if not required.issubset(set(d.columns)):
            d = _ensure_technical_columns(d)

        d = d.replace([np.inf, -np.inf], np.nan).copy()
        if "Close" not in d.columns:
            return unknown

        d = d.dropna(subset=["Close"]).copy()
        if len(d) < 60:
            return unknown

        last = d.iloc[-1]
        prev = d.iloc[-2] if len(d) >= 2 else last

        def g(row, col, default=np.nan):
            return _safe_float(row.get(col, default), default)

        close = g(last, "Close")
        ema20 = g(last, "EMA20")
        ema50 = g(last, "EMA50")
        ema200 = g(last, "EMA200")
        rsi = g(last, "RSI14")
        adx = g(last, "ADX14")
        cmf = g(last, "CMF20", 0.0)
        obv_slope = g(last, "OBV_SLOPE10", 0.0)
        rel_vol = g(last, "REL_VOL", 1.0)
        ema20_prev = g(prev, "EMA20")
        ema50_prev = g(prev, "EMA50")
        ema200_prev = g(prev, "EMA200")

        close_vs_ema20 = close / ema20 - 1.0 if np.isfinite(close) and np.isfinite(ema20) and ema20 != 0 else np.nan
        close_vs_ema50 = close / ema50 - 1.0 if np.isfinite(close) and np.isfinite(ema50) and ema50 != 0 else np.nan
        close_vs_ema200 = close / ema200 - 1.0 if np.isfinite(close) and np.isfinite(ema200) and ema200 != 0 else np.nan
        mom5 = d["Close"].pct_change(5).iloc[-1] if len(d) > 5 else np.nan
        mom20 = d["Close"].pct_change(20).iloc[-1] if len(d) > 20 else np.nan
        mom60 = d["Close"].pct_change(60).iloc[-1] if len(d) > 60 else np.nan
        accel = (mom20 - mom60) if pd.notna(mom20) and pd.notna(mom60) else np.nan

        def rng_score(value, low, high):
            if not np.isfinite(value):
                return 0.0
            mid = (low + high) / 2.0
            half = max(1e-9, (high - low) / 2.0)
            return float(np.clip(100.0 * (1.0 - min(1.0, abs(value - mid) / half)), 0.0, 100.0))

        def bool_score(condition, points):
            return float(points if condition else 0.0)

        scores = {}

        # 1) Markdown
        score = 40.0
        score += bool_score(close < ema20, 10.0)
        score += bool_score(ema20 < ema50, 12.0)
        score += bool_score(ema50 < ema200, 12.0)
        score += bool_score(pd.notna(rsi) and rsi < 45.0, 10.0)
        score += bool_score(pd.notna(cmf) and cmf < 0.0, 8.0)
        score += bool_score(pd.notna(obv_slope) and obv_slope < 0.0, 8.0)
        score += bool_score(pd.notna(mom20) and mom20 < 0.0, 10.0)
        score += bool_score(pd.notna(adx) and adx >= 25.0, 0.0)
        scores["Markdown"] = float(np.clip(score, 0.0, 100.0))

        # 2) Early Accumulation
        score = 38.0
        score += bool_score(close < ema50, 10.0)
        score += bool_score(np.isfinite(close_vs_ema200) and close_vs_ema200 > -0.18, 8.0)
        score += bool_score(pd.notna(rsi) and 30.0 <= rsi <= 52.0, 12.0)
        score += bool_score(pd.notna(cmf) and cmf > 0.0, 10.0)
        score += bool_score(pd.notna(obv_slope) and obv_slope > 0.0, 10.0)
        score += bool_score(pd.notna(mom5) and mom5 > 0.0 and pd.notna(mom20) and mom20 < 0.0, 10.0)
        score += bool_score(pd.notna(adx) and adx < 30.0, 6.0)
        scores["Early Accumulation"] = float(np.clip(score, 0.0, 100.0))

        # 3) Accumulation
        score = 40.0
        score += bool_score(np.isfinite(close_vs_ema50) and abs(close_vs_ema50) <= 0.08, 12.0)
        score += bool_score(pd.notna(ema20) and pd.notna(ema20_prev) and ema20 > ema20_prev, 10.0)
        score += bool_score(pd.notna(ema50) and pd.notna(ema50_prev) and ema50 > ema50_prev, 8.0)
        score += bool_score(pd.notna(rsi) and 40.0 <= rsi <= 60.0, 12.0)
        score += bool_score(pd.notna(cmf) and cmf > 0.0, 10.0)
        score += bool_score(pd.notna(obv_slope) and obv_slope > 0.0, 10.0)
        score += bool_score(pd.notna(mom20) and pd.notna(mom60) and mom20 >= mom60, 8.0)
        score += bool_score(pd.notna(adx) and adx < 30.0, 5.0)
        scores["Accumulation"] = float(np.clip(score, 0.0, 100.0))

        # 4) Late Accumulation
        score = 42.0
        score += bool_score(close > ema20, 10.0)
        score += bool_score(pd.notna(ema20) and pd.notna(ema50) and ema20 >= ema50, 10.0)
        score += bool_score(pd.notna(ema50) and pd.notna(ema200) and ema50 <= ema200, 5.0)
        score += bool_score(pd.notna(rsi) and 45.0 <= rsi <= 65.0, 12.0)
        score += bool_score(pd.notna(cmf) and cmf > 0.0, 8.0)
        score += bool_score(pd.notna(obv_slope) and obv_slope > 0.0, 8.0)
        score += bool_score(pd.notna(mom20) and mom20 > 0.0, 10.0)
        score += bool_score(pd.notna(adx) and 18.0 <= adx <= 35.0, 5.0)
        scores["Late Accumulation"] = float(np.clip(score, 0.0, 100.0))

        # 5) Early Markup
        score = 45.0
        score += bool_score(close > ema20, 12.0)
        score += bool_score(pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50, 12.0)
        score += bool_score(pd.notna(ema50) and pd.notna(ema200) and ema50 >= ema200, 8.0)
        score += bool_score(pd.notna(rsi) and 50.0 <= rsi <= 70.0, 12.0)
        score += bool_score(pd.notna(cmf) and cmf > 0.0, 8.0)
        score += bool_score(pd.notna(obv_slope) and obv_slope > 0.0, 8.0)
        score += bool_score(pd.notna(mom20) and mom20 > 0.0, 10.0)
        score += bool_score(pd.notna(adx) and adx >= 20.0, 5.0)
        scores["Early Markup"] = float(np.clip(score, 0.0, 100.0))

        # 6) Markup
        score = 46.0
        score += bool_score(close > ema20, 10.0)
        score += bool_score(pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50, 10.0)
        score += bool_score(pd.notna(ema50) and pd.notna(ema200) and ema50 > ema200, 10.0)
        score += bool_score(pd.notna(rsi) and 55.0 <= rsi <= 80.0, 12.0)
        score += bool_score(pd.notna(cmf) and cmf > 0.0, 8.0)
        score += bool_score(pd.notna(obv_slope) and obv_slope > 0.0, 8.0)
        score += bool_score(pd.notna(mom20) and mom20 > 0.0, 10.0)
        score += bool_score(pd.notna(adx) and adx >= 25.0, 8.0)
        score += bool_score(np.isfinite(close_vs_ema20) and close_vs_ema20 < 0.12, 6.0)
        scores["Markup"] = float(np.clip(score, 0.0, 100.0))

        # 7) Late Markup
        score = 44.0
        score += bool_score(close > ema20, 10.0)
        score += bool_score(pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50, 10.0)
        score += bool_score(pd.notna(ema50) and pd.notna(ema200) and ema50 > ema200, 10.0)
        score += bool_score(pd.notna(rsi) and rsi >= 60.0, 10.0)
        score += bool_score(pd.notna(cmf) and cmf > 0.0, 6.0)
        score += bool_score(pd.notna(obv_slope) and obv_slope > 0.0, 6.0)
        score += bool_score(pd.notna(mom20) and mom20 > 0.0, 8.0)
        score += bool_score(pd.notna(accel) and accel < 0.0, 10.0)
        score += bool_score(pd.notna(adx) and adx >= 25.0, 6.0)
        score += bool_score(np.isfinite(close_vs_ema20) and close_vs_ema20 >= 0.12, 4.0)
        scores["Late Markup"] = float(np.clip(score, 0.0, 100.0))

        # 8) Distribution
        score = 40.0
        score += bool_score(close > ema200, 8.0)
        score += bool_score(pd.notna(rsi) and 45.0 <= rsi <= 70.0, 10.0)
        score += bool_score(pd.notna(cmf) and cmf < 0.0, 10.0)
        score += bool_score(pd.notna(obv_slope) and obv_slope < 0.0, 10.0)
        score += bool_score(pd.notna(mom20) and mom20 < 0.0, 10.0)
        score += bool_score(pd.notna(accel) and accel < 0.0, 8.0)
        score += bool_score(pd.notna(adx) and adx >= 20.0, 6.0)
        score += bool_score(close > ema20, 4.0)
        scores["Distribution"] = float(np.clip(score, 0.0, 100.0))

        best_phase = max(scores.items(), key=lambda kv: kv[1])[0]
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_score = float(ordered[0][1])
        second_score = float(ordered[1][1]) if len(ordered) > 1 else 0.0
        phase_confidence = float(np.clip((top_score - second_score) * 1.8 + max(0.0, top_score - 40.0), 0.0, 100.0))

        if top_score >= 72.0:
            reason = f"{best_phase}: strong alignment"
        elif top_score >= 60.0:
            reason = f"{best_phase}: moderate alignment"
        else:
            reason = f"{best_phase}: weak alignment"

        if best_phase in {"Markup", "Early Markup", "Late Markup"} and pd.notna(rel_vol) and rel_vol >= 1.5:
            reason += ", volume supportive"
        if best_phase in {"Markdown", "Distribution"} and pd.notna(accel) and accel < 0:
            reason += ", momentum decelerating"

        rank_map = {
            "Early Accumulation": 1.0,
            "Accumulation": 2.0,
            "Late Accumulation": 3.0,
            "Early Markup": 4.0,
            "Markup": 5.0,
            "Late Markup": 6.0,
            "Distribution": 7.0,
            "Markdown": 8.0,
            "Unknown": 0.0,
        }

        return {
            "phase": best_phase,
            "phase_confidence": phase_confidence,
            "phase_rank": rank_map.get(best_phase, 0.0),
            "phase_reason": reason,
            "phase_scores": scores,
        }
    except Exception as exc:
        return {
            "phase": "Unknown",
            "phase_confidence": 0.0,
            "phase_rank": 0.0,
            "phase_reason": f"classify_8_phase_error: {type(exc).__name__}",
            "phase_scores": {},
        }


_orig_classify_8_phase = _classify_8_phase_impl

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


_orig_score_stock_smc = score_stock_smc

def score_stock_smc(*args, **kwargs):
    try:
        return _orig_score_stock_smc(*args, **kwargs)
    except Exception as exc:
        return {
            "valid": False,
            "symbol": kwargs.get("symbol", "n/a"),
            "decision": "REJECT",
            "score": 0.0,
            "core_score": 0.0,
            "market_structure_score": 0.0,
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
