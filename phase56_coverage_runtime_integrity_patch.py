from __future__ import annotations

"""Phase 5.6 transport-only coverage integrity fixes for EMIR.

This patch deliberately does *not* alter scoring weights, ranking thresholds,
real-money authorization, ownership/KSEI semantics, or Future Fundamental
formulae.  It fixes two data-flow defects:

1. Strict, already-verified forward evidence loaded from the durable evidence
   stores was discarded for RADAR_ONLY tickers because the main scan passed an
   empty event frame outside the deep-review shortlist.  We cache the strict
   official events from the existing one-shot direct-evidence load and merge
   those events into the existing Future Fundamental call for the matching
   ticker.  Public/research events are not promoted by this patch.

2. ``build_execution_plan`` returned immediately when accumulation Plan A was
   geometrically invalid, before evaluating the independent breakout/retest
   Plan B.  We rescue only that exact state and build the same conservative
   breakout geometry.  Missing/zero ATR remains missing and no authorization
   flag is created here.
"""

from functools import wraps
from typing import Any, Mapping

import numpy as np
import pandas as pd

PATCH_VERSION = "1.0.0-phase5.6-coverage-runtime-integrity"
_STRICT_FORWARD_BY_TICKER: dict[str, pd.DataFrame] = {}


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _merge_strict_events(existing: Any, strict: Any) -> pd.DataFrame:
    """Merge strict official events without converting research into direct evidence."""
    left = existing.copy() if isinstance(existing, pd.DataFrame) else pd.DataFrame()
    right = strict.copy() if isinstance(strict, pd.DataFrame) else pd.DataFrame()
    if right.empty:
        return left
    if left.empty:
        out = right
    else:
        out = pd.concat([left, right], ignore_index=True, sort=False)
    dedupe = [column for column in ("ticker", "title", "url") if column in out.columns]
    if dedupe:
        out = out.drop_duplicates(dedupe, keep="first")
    return out.reset_index(drop=True)


def _cache_strict_forward_events(bundle: Mapping[str, Any] | None) -> None:
    _STRICT_FORWARD_BY_TICKER.clear()
    if not isinstance(bundle, Mapping):
        return
    frame = bundle.get("official_forward_events")
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return
    local = frame.copy()
    local["ticker"] = local["ticker"].map(_ticker)
    # The upstream loader has already fail-closed on source verification,
    # entity match, >=2-source quorum and freshness for governed rows.  Keep an
    # additional defensive source_verified filter when the field is present.
    if "source_verified" in local.columns:
        verified = local["source_verified"].fillna(False).astype(bool)
        local = local.loc[verified].copy()
    for ticker, group in local.groupby("ticker", sort=False):
        if ticker:
            _STRICT_FORWARD_BY_TICKER[ticker] = group.reset_index(drop=True)


def _wrap_direct_loader(resumable: Any) -> None:
    original = getattr(resumable, "load_verified_direct_evidence", None)
    if not callable(original) or getattr(original, "__phase56_strict_forward_cache_v1__", False):
        return

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        bundle = original(*args, **kwargs)
        _cache_strict_forward_events(bundle)
        return bundle

    wrapped.__phase56_strict_forward_cache_v1__ = True
    setattr(resumable, "load_verified_direct_evidence", wrapped)


def _wrap_future_calculator(resumable: Any) -> None:
    original = getattr(resumable, "calculate_future_fundamental", None)
    if not callable(original) or getattr(original, "__phase56_strict_forward_reuse_v1__", False):
        return

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        ticker = _ticker(kwargs.get("ticker") if "ticker" in kwargs else (args[0] if args else ""))
        strict = _STRICT_FORWARD_BY_TICKER.get(ticker)
        if isinstance(strict, pd.DataFrame) and not strict.empty:
            if "events" in kwargs:
                kwargs = dict(kwargs)
                kwargs["events"] = _merge_strict_events(kwargs.get("events"), strict)
            elif len(args) >= 2:
                mutable = list(args)
                mutable[1] = _merge_strict_events(mutable[1], strict)
                args = tuple(mutable)
        return original(*args, **kwargs)

    wrapped.__phase56_strict_forward_reuse_v1__ = True
    setattr(resumable, "calculate_future_fundamental", wrapped)


def _rr(entry: float, target: float, stop: float) -> float:
    risk = entry - stop
    if not all(np.isfinite(value) for value in (entry, target, stop)) or risk <= 0:
        return np.nan
    return (target - entry) / risk


def _breakout_rescue(
    features: Mapping[str, Any],
    ready: bool,
    lifecycle: str,
    orderbook: Mapping[str, Any] | None,
    *,
    auto_eod_ready: bool = False,
    engine: Any = None,
) -> dict[str, Any] | None:
    """Evaluate independent Plan B after Plan A failed geometry.

    This mirrors the existing breakout/retest construction.  R-multiple targets
    remain explicitly research-only when no observed structural resistance pair
    satisfies the floors.
    """
    if engine is None:
        import narrative_flow_engine as engine

    close = _finite(features.get("last_price"), np.nan)
    atr = _finite(features.get("atr14"), np.nan)
    ema20 = _finite(features.get("ema20"), np.nan)
    high20 = _finite(features.get("high20"), np.nan)
    low20 = _finite(features.get("low20"), np.nan)
    if not all(np.isfinite(value) for value in (close, atr, ema20, high20, low20)) or atr <= 0:
        return None

    orderbook = dict(orderbook or {})
    precise_trigger = _finite(orderbook.get("precise_trigger_price"), np.nan)
    orderbook_verified = str(orderbook.get("orderbook_provenance_state") or "") == "DIRECT_SOURCE_VERIFIED"
    round_idx = getattr(engine, "round_idx")
    idx_tick = getattr(engine, "idx_tick")

    if str(lifecycle) == "MOMENTUM_TRIGGERED":
        raw_trigger = precise_trigger if orderbook_verified and np.isfinite(precise_trigger) else max(close, high20)
        trigger = round_idx(raw_trigger, "up")
        entry_low = round_idx(max(ema20, trigger - 0.65 * atr), "down")
        entry_high = round_idx(trigger + 0.20 * atr, "up")
    else:
        raw_trigger = precise_trigger if orderbook_verified and np.isfinite(precise_trigger) else high20
        trigger = round_idx(raw_trigger, "up")
        entry_low = round_idx(max(low20, ema20 - 0.45 * atr), "down")
        entry_high = round_idx(min(trigger, max(close, ema20 + 0.20 * atr)), "up")
    if entry_high < entry_low:
        entry_low, entry_high = entry_high, entry_low

    breakout_entry = float(trigger)
    tick = float(idx_tick(max(breakout_entry, 1.0)))
    structure_stop = min(
        entry_high - tick,
        breakout_entry - max(1.25 * atr, 0.02 * breakout_entry),
    )
    breakout_stop = float(round_idx(max(structure_stop, breakout_entry * 0.94), "down"))
    if breakout_stop >= breakout_entry:
        breakout_stop = float(round_idx(breakout_entry - tick, "down"))
    risk = breakout_entry - breakout_stop
    if not np.isfinite(risk) or risk <= 0:
        return None

    observed_resistance = sorted({
        float(round_idx(value, "up"))
        for value in (
            _finite(features.get("previous_high20"), np.nan),
            _finite(features.get("prior_high20"), np.nan),
            _finite(features.get("prior_high55"), np.nan),
            _finite(features.get("prior_high120"), np.nan),
            _finite(features.get("prior_high252"), np.nan),
        )
        if np.isfinite(value) and value > 0
    })
    candidates = [value for value in observed_resistance if value > breakout_entry]
    first = next((value for value in candidates if _rr(breakout_entry, value, breakout_stop) >= 1.8), np.nan)
    second = next(
        (value for value in candidates if np.isfinite(first) and value > first and _rr(breakout_entry, value, breakout_stop) >= 2.5),
        np.nan,
    )
    structural = bool(np.isfinite(first) and np.isfinite(second))
    if structural:
        tp1, tp2 = float(first), float(second)
    else:
        tp1 = float(round_idx(breakout_entry + 1.8 * risk, "up"))
        tp2 = float(round_idx(breakout_entry + 3.0 * risk, "up"))

    valid = bool(breakout_stop < breakout_entry < tp1 < tp2)
    if not valid:
        return None
    rr1 = _rr(breakout_entry, tp1, breakout_stop)
    rr2 = _rr(breakout_entry, tp2, breakout_stop)
    min_rr_pass = bool(np.isfinite(rr1) and rr1 >= 1.8)

    if ready and orderbook_verified:
        state = "EMIR_PRECISE_TRIGGER_READY"
    elif auto_eod_ready:
        state = "AUTO_EOD_PROXY_TRIGGER_READY"
    elif ready:
        state = "THESIS_READY_WAIT_DIRECT_BID_OFFER_TRIGGER"
    else:
        state = "RESEARCH_SCENARIO_ONLY"

    return {
        "execution_state": state,
        "execution_plan_semantics": "DUAL_PLAN_SEPARATED_V2_BREAKOUT_RESCUE",
        "entry_low": np.nan,
        "entry_high": np.nan,
        "stop_loss": np.nan,
        "tp1": np.nan,
        "tp2": np.nan,
        "accumulation_entry_low": np.nan,
        "accumulation_entry_high": np.nan,
        "accumulation_stop_loss": np.nan,
        "accumulation_tp1": np.nan,
        "accumulation_tp2": np.nan,
        "accumulation_geometry_valid": False,
        "accumulation_rr_tp1": np.nan,
        "accumulation_rr_tp2": np.nan,
        "accumulation_min_rr_pass": False,
        "trigger": breakout_entry,
        "breakout_entry": breakout_entry,
        "breakout_stop_loss": breakout_stop,
        "breakout_tp1": tp1,
        "breakout_tp2": tp2,
        "breakout_geometry_valid": True,
        "breakout_rr_tp1": round(float(rr1), 2),
        "breakout_rr_tp2": round(float(rr2), 2),
        "breakout_min_rr_pass": min_rr_pass,
        "preferred_execution_path": "BREAKOUT_RETEST",
        "execution_entry_low": breakout_entry,
        "execution_entry_high": breakout_entry,
        "execution_entry_reference": breakout_entry,
        "execution_trigger": breakout_entry,
        "execution_stop_loss": breakout_stop,
        "execution_tp1": tp1,
        "execution_tp2": tp2,
        "execution_rr_tp1": round(float(rr1), 2),
        "execution_rr_tp2": round(float(rr2), 2),
        "execution_geometry_valid": True,
        "execution_min_rr_pass": min_rr_pass,
        "execution_geometry_state": "VALID_SELECTED_PATH_BREAKOUT_RESCUED",
        "hard_stop_distance_pct": round(100.0 * risk / breakout_entry, 2),
        "risk_doctrine_state": "SEPARATE_ACCUMULATION_VS_BREAKOUT_RISK_GEOMETRY_V2",
        "execution_target_basis": "OBSERVED_PRIOR_RESISTANCE" if structural else "R_MULTIPLE_FALLBACK_RESEARCH_ONLY",
        "execution_targets_structural": structural,
        "observed_resistance_candidates": observed_resistance,
        "phase56_geometry_rescue_state": "PLAN_A_INVALID_PLAN_B_EVALUATED",
    }


def _wrap_execution_builder(engine: Any) -> None:
    original = getattr(engine, "build_execution_plan", None)
    if not callable(original) or getattr(original, "__phase56_plan_b_rescue_v1__", False):
        return

    @wraps(original)
    def wrapped(features: Mapping[str, Any], ready: bool, lifecycle: str, orderbook: Mapping[str, Any] | None = None, *, auto_eod_ready: bool = False):
        result = original(features, ready, lifecycle, orderbook, auto_eod_ready=auto_eod_ready)
        if not isinstance(result, dict) or result.get("execution_geometry_state") != "ACCUMULATION_GEOMETRY_INVALID":
            return result
        rescue = _breakout_rescue(
            features, ready, lifecycle, orderbook,
            auto_eod_ready=auto_eod_ready,
            engine=engine,
        )
        return rescue if rescue is not None else result

    wrapped.__phase56_plan_b_rescue_v1__ = True
    setattr(engine, "build_execution_plan", wrapped)


def install() -> dict[str, str]:
    import narrative_flow_engine
    import resumable_scan

    _wrap_direct_loader(resumable_scan)
    _wrap_future_calculator(resumable_scan)
    _wrap_execution_builder(narrative_flow_engine)
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "future_fundamental": "STRICT_PERSISTED_FORWARD_REUSED_OUTSIDE_DEEP_REVIEW",
        "execution_geometry": "PLAN_B_EVALUATED_AFTER_PLAN_A_INVALID",
        "authorization": "UNCHANGED",
    }


__all__ = [
    "PATCH_VERSION",
    "install",
    "_breakout_rescue",
    "_cache_strict_forward_events",
    "_merge_strict_events",
]
