"""IDX edge lab

A lightweight research/backtest layer for the scanner.

Goals:
- keep scanner logic honest via out-of-sample testing
- measure edge in R-multiples, win rate, expectancy, profit factor, drawdown
- support walk-forward calibration on IDX / IHSG sample universes

This module is intentionally self-contained and uses only pandas/numpy plus the
project's existing data / technical helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

try:
    from data_engine import load_ticker_data, normalize_ticker
except Exception:  # pragma: no cover - fallback for lean local environments
    def normalize_ticker(symbol: str) -> str:
        s = str(symbol).strip().upper()
        if not s or s == "NAN":
            return ""
        if s.startswith("^"):
            return s
        return s if s.endswith(".JK") else f"{s}.JK"

    def load_ticker_data(symbol: str, months: int) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception:
            return pd.DataFrame()
        try:
            months = max(1, int(months))
        except Exception:
            months = 12
        base = str(symbol).strip()
        candidates = [base]
        if base and not base.startswith("^"):
            if base.endswith(".JK"):
                candidates.append(base[:-3])
            else:
                candidates.append(f"{base}.JK")
        seen = set()
        candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]
        for candidate in candidates:
            try:
                df = yf.download(candidate, period=f"{months}mo", interval="1d", auto_adjust=False, progress=False, threads=False)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except Exception:
                continue
        return pd.DataFrame()

try:
    from technical_analyst import _ensure_technical_columns, build_macro_liquidity_gate
except Exception:  # pragma: no cover
    def _ensure_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def build_macro_liquidity_gate(*args, **kwargs) -> dict:
        return {"benchmark_symbol": kwargs.get("benchmark_symbol", "^JKSE"), "market_regime": "SIDEWAYS", "macro_gate_ok": False, "macro_gate_reason": "technical_analyst_unavailable"}


@dataclass(frozen=True)
class StrategyParams:
    """Parameters for scanner-aligned research backtests.

    These are intentionally conservative. The goal is not the highest trade count,
    but the best chance of preserving edge after fees, slippage, and regime changes.
    """

    min_rel_vol: float = 1.10
    min_adx: float = 18.0
    max_adx: float = 35.0
    min_rsi: float = 50.0
    max_rsi: float = 72.0
    min_score: float = 58.0
    stop_atr: float = 1.8
    target1_atr: float = 2.2
    target2_atr: float = 3.8
    entry_buffer_atr: float = 0.25
    max_hold_bars: int = 20
    fee_bps: float = 20.0
    slippage_bps: float = 10.0
    max_risk_pct: float = 0.10
    require_bull_stack: bool = True
    require_cmf_positive: bool = True
    require_close_above_ema20: bool = True
    use_breakout_signal: bool = True
    use_pullback_signal: bool = True


@dataclass
class Trade:
    symbol: str
    entry_i: int
    exit_i: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    r_multiple: float
    outcome: str
    bars_held: int
    setup_kind: str


def _safe_float(value, default=np.nan) -> float:
    try:
        v = float(value)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = _ensure_technical_columns(df.copy())
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _benchmark_context(benchmark_df: pd.DataFrame | None, benchmark_symbol: str = "^JKSE") -> dict:
    if benchmark_df is None or benchmark_df.empty:
        return {"benchmark_symbol": benchmark_symbol, "market_regime": "SIDEWAYS", "macro_gate_ok": False, "macro_gate_reason": "benchmark_empty"}
    return build_macro_liquidity_gate(benchmark_df.copy(), benchmark_symbol)


def _signal_candidates(row: pd.Series, params: StrategyParams) -> list[tuple[str, float]]:
    """Return candidate long setups with a simple quality score."""
    close = _safe_float(row.get("Close"))
    ema20 = _safe_float(row.get("EMA20"))
    ema50 = _safe_float(row.get("EMA50"))
    ema200 = _safe_float(row.get("EMA200"))
    atr14 = _safe_float(row.get("ATR14"), max(close * 0.02, 1.0))
    rsi14 = _safe_float(row.get("RSI14"), 50.0)
    adx14 = _safe_float(row.get("ADX14"), 0.0)
    rel_vol = _safe_float(row.get("REL_VOL"), 1.0)
    cmf20 = _safe_float(row.get("CMF20"), 0.0)
    macd_hist = _safe_float(row.get("MACD_HIST"), 0.0)
    obv_slope = _safe_float(row.get("OBV_SLOPE10"), 0.0)

    if not np.isfinite(close) or close <= 0:
        return []
    if not np.isfinite(atr14) or atr14 <= 0:
        atr14 = max(close * 0.02, 1.0)

    bull_stack = np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(ema200) and (ema20 > ema50 > ema200)
    above_ema20 = np.isfinite(ema20) and close > ema20
    above_ema50 = np.isfinite(ema50) and close > ema50
    breakout_20d = bool(row.get("Close", np.nan) > row.get("High20Prev", np.nan)) if pd.notna(row.get("High20Prev", np.nan)) else False
    pullback_zone = np.isfinite(ema20) and np.isfinite(ema50) and (close >= ema20 - 0.35 * atr14) and (close <= ema20 + 0.25 * atr14)

    signals: list[tuple[str, float]] = []

    trend_ok = (
        (not params.require_bull_stack or bull_stack)
        and (not params.require_close_above_ema20 or above_ema20)
        and (not params.require_cmf_positive or cmf20 > 0)
        and (adx14 >= params.min_adx)
        and (adx14 <= params.max_adx)
        and (rsi14 >= params.min_rsi)
        and (rsi14 <= params.max_rsi)
        and (rel_vol >= params.min_rel_vol)
    )
    if trend_ok:
        score = 50.0
        score += 12.0 if bull_stack else 0.0
        score += 10.0 if above_ema20 else 0.0
        score += 8.0 if above_ema50 else 0.0
        score += 8.0 if cmf20 > 0 else 0.0
        score += 6.0 if macd_hist > 0 else 0.0
        score += 6.0 if obv_slope > 0 else 0.0
        score += min(10.0, max(0.0, (rel_vol - params.min_rel_vol) * 6.0))
        signals.append(("Trend", float(np.clip(score, 0.0, 100.0))))

    if params.use_breakout_signal and breakout_20d and rel_vol >= params.min_rel_vol and close > ema20:
        score = 62.0 + min(18.0, (rel_vol - params.min_rel_vol) * 8.0)
        if bull_stack:
            score += 10.0
        if adx14 >= 22:
            score += 4.0
        if cmf20 > 0:
            score += 4.0
        signals.append(("Breakout", float(np.clip(score, 0.0, 100.0))))

    if params.use_pullback_signal and pullback_zone and above_ema50 and (cmf20 > 0 or macd_hist > 0) and rsi14 >= 45:
        score = 58.0
        score += 8.0 if bull_stack else 0.0
        score += 6.0 if above_ema20 else 0.0
        score += 4.0 if adx14 >= 18 else 0.0
        score += 4.0 if obv_slope > 0 else 0.0
        signals.append(("Pullback", float(np.clip(score, 0.0, 100.0))))

    return signals


def _build_trade_plan_from_row(
    df: pd.DataFrame,
    i: int,
    params: StrategyParams,
    setup_kind: str,
) -> tuple[float, float, float, float] | None:
    """Return entry, stop, target1, target2 using only info available on bar i."""
    if i <= 0 or i >= len(df) - 1:
        return None

    row = df.iloc[i]
    close = _safe_float(row.get("Close"))
    atr14 = _safe_float(row.get("ATR14"), max(close * 0.02, 1.0))
    ema20 = _safe_float(row.get("EMA20"))
    ema50 = _safe_float(row.get("EMA50"))
    low20 = _safe_float(row.get("Low20Prev"))
    high20 = _safe_float(row.get("High20Prev"))

    if not np.isfinite(close) or close <= 0:
        return None
    if not np.isfinite(atr14) or atr14 <= 0:
        atr14 = max(close * 0.02, 1.0)

    next_open = _safe_float(df.iloc[i + 1].get("Open"), np.nan)
    if not np.isfinite(next_open) or next_open <= 0:
        return None

    if setup_kind == "Breakout":
        entry = max(next_open, close + 0.10 * atr14)
        stop = min([
            v for v in [high20 - 2.2 * atr14 if np.isfinite(high20) else np.nan, ema20 - 0.35 * atr14 if np.isfinite(ema20) else np.nan, close - params.stop_atr * atr14]
            if np.isfinite(v)
        ], default=close - params.stop_atr * atr14)
        target1 = max(entry + params.target1_atr * atr14, close + 1.25 * atr14)
        target2 = max(entry + params.target2_atr * atr14, target1 + 0.9 * atr14)
    elif setup_kind == "Pullback":
        entry_zone = [v for v in [ema20 - params.entry_buffer_atr * atr14 if np.isfinite(ema20) else np.nan, ema50 + 0.08 * atr14 if np.isfinite(ema50) else np.nan] if np.isfinite(v)]
        if not entry_zone:
            return None
        entry = float(np.mean(entry_zone))
        entry = min(entry, close - 0.20 * atr14)
        stop = min([
            v for v in [low20 - 0.15 * atr14 if np.isfinite(low20) else np.nan, entry - params.stop_atr * atr14, ema50 - 0.25 * atr14 if np.isfinite(ema50) else np.nan]
            if np.isfinite(v)
        ], default=entry - params.stop_atr * atr14)
        target1 = max(entry + params.target1_atr * atr14, close + 0.75 * atr14)
        target2 = max(entry + params.target2_atr * atr14, target1 + 0.8 * atr14)
    else:
        entry = max(next_open, close)
        stop = min([
            v for v in [low20 - 0.20 * atr14 if np.isfinite(low20) else np.nan, ema20 - 0.30 * atr14 if np.isfinite(ema20) else np.nan, entry - params.stop_atr * atr14]
            if np.isfinite(v)
        ], default=entry - params.stop_atr * atr14)
        target1 = max(entry + params.target1_atr * atr14, close + 1.0 * atr14)
        target2 = max(entry + params.target2_atr * atr14, target1 + 0.75 * atr14)

    if not all(np.isfinite(x) for x in [entry, stop, target1, target2]):
        return None
    if stop >= entry:
        stop = entry - 0.90 * atr14
    risk = entry - stop
    if risk <= 0:
        return None

    # hard risk controls
    if risk / entry > params.max_risk_pct:
        return None
    rr1 = (target1 - entry) / risk
    rr2 = (target2 - entry) / risk
    if rr1 < 1.25 or rr2 < 1.90:
        return None

    return float(entry), float(stop), float(target1), float(target2)


def _simulate_single_trade(
    df: pd.DataFrame,
    i: int,
    plan: tuple[float, float, float, float],
    params: StrategyParams,
    symbol: str,
    setup_kind: str,
) -> Trade | None:
    entry, stop, target1, target2 = plan
    entry_fill = entry * (1.0 + params.slippage_bps / 10000.0)
    stop_fill = stop * (1.0 - params.slippage_bps / 10000.0)
    target_fill = target2 * (1.0 - params.slippage_bps / 10000.0)
    fee = params.fee_bps / 10000.0

    risk = entry_fill - stop_fill
    if risk <= 0:
        return None

    exit_price = target_fill
    exit_i = min(len(df) - 1, i + params.max_hold_bars)
    outcome = "TIME"

    for j in range(i + 1, min(len(df), i + params.max_hold_bars + 1)):
        bar = df.iloc[j]
        high = _safe_float(bar.get("High"))
        low = _safe_float(bar.get("Low"))
        if not np.isfinite(high) or not np.isfinite(low):
            continue

        stop_hit = low <= stop_fill
        target_hit = high >= target_fill
        # Conservative sequencing: if both are touched in the same bar, assume stop first.
        if stop_hit:
            exit_price = stop_fill
            exit_i = j
            outcome = "STOP"
            break
        if target_hit:
            exit_price = target_fill
            exit_i = j
            outcome = "TARGET"
            break
    else:
        if exit_i < len(df):
            exit_price = _safe_float(df.iloc[exit_i].get("Close"), exit_price)

    gross_r = (exit_price - entry_fill) / risk
    # Approximate round-trip cost in R terms.
    cost_r = 2.0 * fee * entry_fill / risk
    net_r = gross_r - cost_r
    if np.isnan(net_r):
        return None

    return Trade(
        symbol=symbol,
        entry_i=i + 1,
        exit_i=exit_i,
        entry_price=float(entry_fill),
        exit_price=float(exit_price),
        stop_price=float(stop_fill),
        target_price=float(target_fill),
        r_multiple=float(net_r),
        outcome=outcome,
        bars_held=int(exit_i - (i + 1) + 1),
        setup_kind=setup_kind,
    )


def run_backtest(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    params: StrategyParams | None = None,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_symbol: str = "^JKSE",
) -> dict:
    """Backtest a single symbol using scanner-aligned long-only rules."""
    params = params or StrategyParams()
    if df is None or df.empty or len(df) < 120:
        return {"symbol": symbol, "valid": False, "reason": "data_too_short"}

    d = _prepare_features(df)
    if d.empty:
        return {"symbol": symbol, "valid": False, "reason": "feature_prep_failed"}

    d["High20Prev"] = d["High"].rolling(20).max().shift(1)
    d["Low20Prev"] = d["Low"].rolling(20).min().shift(1)

    trades: list[Trade] = []
    open_trade: Trade | None = None

    # Optional macro gate only changes whether we take signals; it does not overwrite results.
    macro_ctx = _benchmark_context(benchmark_df, benchmark_symbol)
    market_regime = str(macro_ctx.get("market_regime", "SIDEWAYS")).upper()
    macro_gate_ok = bool(macro_ctx.get("macro_gate_ok", True))

    for i in range(60, len(d) - 2):
        # Close trade if one is open.
        if open_trade is not None:
            if i <= open_trade.exit_i:
                continue
            open_trade = None

        row = d.iloc[i]
        candidates = _signal_candidates(row, params)
        if not candidates:
            continue

        # Macro filter for long-only setups.
        if market_regime == "BEAR" and not macro_gate_ok:
            continue

        # Keep the highest quality setup only.
        setup_kind, setup_score = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
        if setup_score < params.min_score:
            continue

        plan = _build_trade_plan_from_row(d, i, params, setup_kind)
        if plan is None:
            continue

        trade = _simulate_single_trade(d, i, plan, params, symbol, setup_kind)
        if trade is None:
            continue

        trades.append(trade)
        open_trade = trade

    return {
        "symbol": symbol,
        "valid": True,
        "params": asdict(params),
        "macro_context": macro_ctx,
        "trades": trades,
        "trade_count": len(trades),
        "equity_r": pd.Series([t.r_multiple for t in trades], dtype=float).cumsum() if trades else pd.Series(dtype=float),
        **trade_metrics(trades),
    }


def trade_metrics(trades: Iterable[Trade]) -> dict:
    trades = list(trades)
    if not trades:
        return {
            "trade_count": 0,
            "winrate": np.nan,
            "expectancy_r": np.nan,
            "profit_factor": np.nan,
            "avg_r": np.nan,
            "median_r": np.nan,
            "max_drawdown_r": np.nan,
            "avg_bars_held": np.nan,
            "best_r": np.nan,
            "worst_r": np.nan,
        }

    rs = np.array([t.r_multiple for t in trades], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    equity = np.cumsum(rs)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    dd = equity - peak[1:]

    return {
        "trade_count": int(len(trades)),
        "winrate": float((rs > 0).mean()),
        "expectancy_r": float(rs.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size else np.inf,
        "avg_r": float(rs.mean()),
        "median_r": float(np.median(rs)),
        "max_drawdown_r": float(dd.min()) if len(dd) else 0.0,
        "avg_bars_held": float(np.mean([t.bars_held for t in trades])),
        "best_r": float(rs.max()),
        "worst_r": float(rs.min()),
    }


def _iter_param_grid(grid: dict[str, list]) -> Iterator[StrategyParams]:
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        kwargs = dict(zip(keys, values))
        yield StrategyParams(**kwargs)


def optimize_params(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    param_grid: dict[str, list],
    benchmark_symbol: str = "^JKSE",
    min_trades: int = 8,
) -> tuple[StrategyParams, dict]:
    """Grid-search parameters on a training slice.

    Objective prefers positive expectancy, sane drawdown, and a minimum trade sample.
    """
    best_params = StrategyParams()
    best_stats: dict = {"score": -np.inf}

    for params in _iter_param_grid(param_grid):
        res = run_backtest(df, params=params, benchmark_df=benchmark_df, benchmark_symbol=benchmark_symbol)
        if not res.get("valid", False):
            continue
        tc = int(res.get("trade_count", 0) or 0)
        if tc < min_trades:
            continue
        exp_r = _safe_float(res.get("expectancy_r"), np.nan)
        pf = _safe_float(res.get("profit_factor"), np.nan)
        dd = abs(_safe_float(res.get("max_drawdown_r"), np.nan))
        wr = _safe_float(res.get("winrate"), np.nan)
        if not np.isfinite(exp_r) or not np.isfinite(pf) or not np.isfinite(dd) or not np.isfinite(wr):
            continue
        score = (exp_r * 100.0) + (pf * 8.0) + (wr * 30.0) - (dd * 12.0) + min(15.0, tc * 0.7)
        if score > best_stats.get("score", -np.inf):
            best_stats = {**res, "score": float(score)}
            best_params = params

    return best_params, best_stats


def walk_forward_test(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    benchmark_symbol: str = "^JKSE",
    train_bars: int = 252,
    test_bars: int = 63,
    step_bars: int | None = None,
    param_grid: dict[str, list] | None = None,
    min_trades: int = 8,
) -> pd.DataFrame:
    """Walk-forward optimizer + out-of-sample evaluator.

    Returns a fold-level dataframe with both in-sample and out-of-sample metrics.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if step_bars is None:
        step_bars = test_bars
    if param_grid is None:
        param_grid = {
            "min_rel_vol": [1.05, 1.10, 1.20],
            "min_adx": [16.0, 18.0, 20.0],
            "max_adx": [30.0, 35.0],
            "min_rsi": [48.0, 50.0, 52.0],
            "max_rsi": [68.0, 72.0],
            "min_score": [55.0, 58.0, 62.0],
            "stop_atr": [1.6, 1.8, 2.0],
            "target1_atr": [2.0, 2.2, 2.4],
            "target2_atr": [3.4, 3.8, 4.2],
            "entry_buffer_atr": [0.18, 0.25, 0.32],
            "max_hold_bars": [15, 20, 25],
            "fee_bps": [20.0],
            "slippage_bps": [10.0],
            "max_risk_pct": [0.08, 0.10],
        }

    d = _prepare_features(df)
    if d.empty:
        return pd.DataFrame()
    d["High20Prev"] = d["High"].rolling(20).max().shift(1)
    d["Low20Prev"] = d["Low"].rolling(20).min().shift(1)

    rows = []
    start = max(120, train_bars)
    while start + test_bars <= len(d):
        train = d.iloc[:start].copy()
        test = d.iloc[start : start + test_bars].copy()
        bench_train = benchmark_df.iloc[:start].copy() if benchmark_df is not None and not benchmark_df.empty else None
        bench_test = benchmark_df.iloc[start : start + test_bars].copy() if benchmark_df is not None and not benchmark_df.empty else None

        best_params, in_sample = optimize_params(train, bench_train, param_grid, benchmark_symbol=benchmark_symbol, min_trades=min_trades)
        oos = run_backtest(test, params=best_params, benchmark_df=bench_test, benchmark_symbol=benchmark_symbol)

        rows.append(
            {
                "fold_start": int(start - train_bars),
                "fold_end": int(start + test_bars),
                "train_bars": int(len(train)),
                "test_bars": int(len(test)),
                "best_params": asdict(best_params),
                "train_trade_count": int(in_sample.get("trade_count", 0) or 0),
                "train_expectancy_r": in_sample.get("expectancy_r", np.nan),
                "train_profit_factor": in_sample.get("profit_factor", np.nan),
                "train_winrate": in_sample.get("winrate", np.nan),
                "train_max_drawdown_r": in_sample.get("max_drawdown_r", np.nan),
                "test_trade_count": int(oos.get("trade_count", 0) or 0),
                "test_expectancy_r": oos.get("expectancy_r", np.nan),
                "test_profit_factor": oos.get("profit_factor", np.nan),
                "test_winrate": oos.get("winrate", np.nan),
                "test_max_drawdown_r": oos.get("max_drawdown_r", np.nan),
                "test_avg_r": oos.get("avg_r", np.nan),
                "test_best_r": oos.get("best_r", np.nan),
                "test_worst_r": oos.get("worst_r", np.nan),
            }
        )
        start += step_bars

    return pd.DataFrame(rows)


def universe_backtest(
    tickers: Iterable[str],
    months: int = 24,
    benchmark_symbol: str = "^JKSE",
    params: StrategyParams | None = None,
) -> pd.DataFrame:
    """Convenience helper to backtest a ticker universe."""
    params = params or StrategyParams()
    benchmark = load_ticker_data(benchmark_symbol, months)
    rows = []
    for raw in tickers:
        sym = normalize_ticker(raw)
        if not sym:
            continue
        df = load_ticker_data(sym, months)
        res = run_backtest(df, symbol=sym, params=params, benchmark_df=benchmark, benchmark_symbol=benchmark_symbol)
        if res.get("valid", False):
            rows.append(
                {
                    "symbol": sym,
                    "trade_count": res.get("trade_count", 0),
                    "winrate": res.get("winrate", np.nan),
                    "expectancy_r": res.get("expectancy_r", np.nan),
                    "profit_factor": res.get("profit_factor", np.nan),
                    "max_drawdown_r": res.get("max_drawdown_r", np.nan),
                    "avg_r": res.get("avg_r", np.nan),
                    "best_r": res.get("best_r", np.nan),
                    "worst_r": res.get("worst_r", np.nan),
                }
            )
    return pd.DataFrame(rows)


def summarize_walk_forward(folds: pd.DataFrame) -> dict:
    if folds is None or folds.empty:
        return {"valid": False, "reason": "no_folds"}

    test_expectancy = folds["test_expectancy_r"].dropna()
    test_pf = folds["test_profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    test_wr = folds["test_winrate"].dropna()
    test_dd = folds["test_max_drawdown_r"].dropna()

    return {
        "valid": True,
        "fold_count": int(len(folds)),
        "avg_test_expectancy_r": float(test_expectancy.mean()) if not test_expectancy.empty else np.nan,
        "median_test_expectancy_r": float(test_expectancy.median()) if not test_expectancy.empty else np.nan,
        "avg_test_profit_factor": float(test_pf.mean()) if not test_pf.empty else np.nan,
        "avg_test_winrate": float(test_wr.mean()) if not test_wr.empty else np.nan,
        "avg_test_max_drawdown_r": float(test_dd.mean()) if not test_dd.empty else np.nan,
        "positive_expectancy_fold_ratio": float((folds["test_expectancy_r"] > 0).mean()) if "test_expectancy_r" in folds else np.nan,
    }
