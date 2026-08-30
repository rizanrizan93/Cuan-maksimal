from __future__ import annotations

from html import escape
from typing import Any, Mapping

import numpy as np
import pandas as pd
import top3_dashboard_legacy as _legacy

from release_contract import SCANNER_RELEASE_VERSION
from top3_dashboard_legacy import *  # noqa: F401,F403

SMART_MONEY_COST_BASIS_VERSION = "1.0.0"
SCANNER_VERSION = SCANNER_RELEASE_VERSION


def _num(value: Any, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _cost_state(distance_pct: float) -> str:
    if not np.isfinite(distance_pct):
        return "COST_UNAVAILABLE"
    if distance_pct < -3.0:
        return "UNDER_PROXY_COST"
    if distance_pct <= 5.0:
        return "AT_PROXY_COST"
    if distance_pct <= 15.0:
        return "EARLY_MARKUP"
    if distance_pct <= 35.0:
        return "MARKUP"
    return "EXTENDED_MARKUP"


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    total = float(weights.sum())
    if total <= 0:
        return float(np.quantile(values, q))
    cumulative = np.cumsum(weights) / total
    return float(np.interp(float(q), cumulative, values))


def _ohlcv_cost_basis(frame: pd.DataFrame) -> dict[str, Any]:
    base = {
        "estimated_smart_money_cost": np.nan,
        "estimated_smart_money_cost_low": np.nan,
        "estimated_smart_money_cost_high": np.nan,
        "smart_money_cost_distance_pct": np.nan,
        "smart_money_cost_state": "COST_UNAVAILABLE",
        "smart_money_cost_confidence_pct": 0.0,
        "smart_money_cost_evidence_type": "COST_UNAVAILABLE",
        "smart_money_cost_note": "No defensible accumulation-cost proxy available.",
        "smart_money_cost_basis_version": SMART_MONEY_COST_BASIS_VERSION,
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return base
    required = {"High", "Low", "Close", "Volume"}
    if not required.issubset(frame.columns):
        return base

    local = frame.tail(120).copy()
    high = pd.to_numeric(local["High"], errors="coerce")
    low = pd.to_numeric(local["Low"], errors="coerce")
    close = pd.to_numeric(local["Close"], errors="coerce")
    volume = pd.to_numeric(local["Volume"], errors="coerce")
    valid = high.notna() & low.notna() & close.notna() & volume.notna() & volume.gt(0)
    if int(valid.sum()) < 20:
        return base

    high, low, close, volume = high[valid], low[valid], close[valid], volume[valid]
    typical = (high + low + close) / 3.0
    span = (high - low).replace(0.0, np.nan)
    location = ((close - low) / span).clip(0.0, 1.0).fillna(0.5)
    returns = close.pct_change().fillna(0.0)
    avg_volume = volume.rolling(20, min_periods=3).mean().replace(0.0, np.nan)
    vol_ratio = (volume / avg_volume).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    recent = pd.DataFrame({
        "typical": typical,
        "close": close,
        "volume": volume,
        "location": location,
        "returns": returns,
        "vol_ratio": vol_ratio,
    }).tail(60)
    mask = (
        ((recent["returns"] >= -0.004) & (recent["location"] >= 0.58) & recent["vol_ratio"].between(0.60, 1.90))
        | ((recent["returns"] < 0.0) & (recent["location"] >= 0.55) & (recent["vol_ratio"] >= 1.05))
    )
    selected = recent.loc[mask].dropna(subset=["typical", "volume"])
    if len(selected) < 5:
        selected = recent.tail(20).dropna(subset=["typical", "volume"])
        fallback = True
    else:
        fallback = False
    if selected.empty or float(selected["volume"].sum()) <= 0:
        return base

    values = selected["typical"].to_numpy(dtype=float)
    weights = selected["volume"].to_numpy(dtype=float)
    mid = float(np.average(values, weights=weights))
    low_zone = _weighted_quantile(values, weights, 0.25)
    high_zone = _weighted_quantile(values, weights, 0.75)
    if not (np.isfinite(low_zone) and np.isfinite(high_zone)) or low_zone <= 0 or high_zone <= 0:
        low_zone, high_zone = mid * 0.975, mid * 1.025
    if high_zone < low_zone:
        low_zone, high_zone = high_zone, low_zone

    last = float(close.iloc[-1])
    distance = 100.0 * (last / mid - 1.0) if mid > 0 else np.nan
    participation = min(1.0, len(selected) / 15.0)
    volume_share = min(1.0, float(selected["volume"].sum()) / max(float(recent["volume"].sum()), 1.0) * 2.5)
    confidence = min(72.0, 28.0 + 24.0 * participation + 20.0 * volume_share)
    if fallback:
        confidence = min(confidence, 46.0)

    return {
        "estimated_smart_money_cost": round(mid, 4),
        "estimated_smart_money_cost_low": round(low_zone, 4),
        "estimated_smart_money_cost_high": round(high_zone, 4),
        "smart_money_cost_distance_pct": round(distance, 2) if np.isfinite(distance) else np.nan,
        "smart_money_cost_state": _cost_state(distance),
        "smart_money_cost_confidence_pct": round(confidence, 1),
        "smart_money_cost_evidence_type": "OHLCV_ACCUMULATION_COST_PROXY" if not fallback else "OHLCV_TRADED_VALUE_FALLBACK_PROXY",
        "smart_money_cost_note": (
            "60D accumulation/absorption bars weighted by traded volume; proxy only, not broker or beneficial-owner identity."
            if not fallback else
            "20D traded-value fallback because accumulation-bar sample was insufficient; lower confidence."
        ),
        "smart_money_cost_basis_version": SMART_MONEY_COST_BASIS_VERSION,
    }


def _row_fallback_cost_basis(row: Mapping[str, Any]) -> dict[str, Any]:
    last = _num(row.get("last_price"), np.nan)
    evidence_text = str(row.get("broker_inventory_evidence_type") or "").upper()
    direct_mid = _num(
        row.get("dominant_broker_avg_cost",
                row.get("broker_avg_buy_price",
                        row.get("verified_broker_average_cost"))),
        np.nan,
    )
    if "DIRECT" in evidence_text and np.isfinite(direct_mid):
        low = _num(row.get("dominant_broker_cost_low"), direct_mid * 0.985)
        high = _num(row.get("dominant_broker_cost_high"), direct_mid * 1.015)
        distance = 100.0 * (last / direct_mid - 1.0) if np.isfinite(last) and direct_mid > 0 else np.nan
        return {
            "estimated_smart_money_cost": round(direct_mid, 4),
            "estimated_smart_money_cost_low": round(min(low, high), 4),
            "estimated_smart_money_cost_high": round(max(low, high), 4),
            "smart_money_cost_distance_pct": round(distance, 2) if np.isfinite(distance) else np.nan,
            "smart_money_cost_state": _cost_state(distance),
            "smart_money_cost_confidence_pct": min(100.0, max(78.0, _num(row.get("broker_inventory_coverage_pct"), 78.0))),
            "smart_money_cost_evidence_type": "DIRECT_BROKER_EVIDENCE",
            "smart_money_cost_note": "Direct broker-level average cost evidence; broker code is not beneficial-owner identity.",
            "smart_money_cost_basis_version": SMART_MONEY_COST_BASIS_VERSION,
        }

    defended = _num(row.get("defended_level"), np.nan)
    ema20 = _num(row.get("ema20"), np.nan)
    entry_low = _num(row.get("entry_low"), np.nan)
    entry_high = _num(row.get("entry_high"), np.nan)
    anchors = [value for value in (defended, ema20) if np.isfinite(value) and value > 0]
    if anchors:
        mid = float(np.median(anchors))
        low, high = mid * 0.975, mid * 1.025
        basis = "DEFENDED_LEVEL_EMA20_PROXY"
    elif np.isfinite(entry_low) and np.isfinite(entry_high) and entry_low > 0 and entry_high > 0:
        low, high = min(entry_low, entry_high), max(entry_low, entry_high)
        mid = (low + high) / 2.0
        basis = "ENTRY_ZONE_PROXY"
    else:
        return _ohlcv_cost_basis(pd.DataFrame())
    distance = 100.0 * (last / mid - 1.0) if np.isfinite(last) and mid > 0 else np.nan
    confidence = min(52.0, 0.5 * _num(row.get("broker_inventory_coverage_pct"), 0.0) + 20.0)
    return {
        "estimated_smart_money_cost": round(mid, 4),
        "estimated_smart_money_cost_low": round(low, 4),
        "estimated_smart_money_cost_high": round(high, 4),
        "smart_money_cost_distance_pct": round(distance, 2) if np.isfinite(distance) else np.nan,
        "smart_money_cost_state": _cost_state(distance),
        "smart_money_cost_confidence_pct": round(confidence, 1),
        "smart_money_cost_evidence_type": basis,
        "smart_money_cost_note": "Persisted-scan fallback proxy; not broker or beneficial-owner identity.",
        "smart_money_cost_basis_version": SMART_MONEY_COST_BASIS_VERSION,
    }


def enrich_dashboard_scores(radar: pd.DataFrame, frames: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    frames = frames or {}
    # Runtime compatibility installers may replace the public legacy wrapper.
    # Decision calculation always uses the stable on-disk implementation.
    local = _legacy._canonical_enrich_dashboard_scores(radar, frames)
    if local.empty:
        return local
    rows = []
    for _, row in local.iterrows():
        ticker = str(row.get("ticker") or "")
        frame = frames.get(ticker, pd.DataFrame())
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            payload = _ohlcv_cost_basis(frame)
            direct = _row_fallback_cost_basis(row)
            if direct.get("smart_money_cost_evidence_type") == "DIRECT_BROKER_EVIDENCE":
                payload = direct
        else:
            payload = _row_fallback_cost_basis(row)
        rows.append(payload)
    extra = pd.DataFrame(rows, index=local.index)
    for column in extra.columns:
        local[column] = extra[column]
    return local


# Stable implementation handle used by the explicit final-decision pipeline.
# Runtime compatibility patches may replace ``enrich_dashboard_scores`` but do
# not alter this canonical calculation function.
_canonical_enrich_dashboard_scores = enrich_dashboard_scores


def _fmt_rupiah(value: Any) -> str:
    parsed = _num(value)
    if not np.isfinite(parsed) or parsed <= 0:
        return "—"
    return "Rp" + f"{parsed:,.0f}".replace(",", ".")


def _fmt_pct(value: Any) -> str:
    parsed = _num(value)
    return "—" if not np.isfinite(parsed) else f"{parsed:+.1f}%"


def _cost_block(row: Mapping[str, Any]) -> str:
    low = _fmt_rupiah(row.get("estimated_smart_money_cost_low"))
    high = _fmt_rupiah(row.get("estimated_smart_money_cost_high"))
    mid = _fmt_rupiah(row.get("estimated_smart_money_cost"))
    state = escape(str(row.get("smart_money_cost_state") or "COST_UNAVAILABLE"))
    evidence = escape(str(row.get("smart_money_cost_evidence_type") or "COST_UNAVAILABLE"))
    distance = _fmt_pct(row.get("smart_money_cost_distance_pct"))
    confidence = _num(row.get("smart_money_cost_confidence_pct"), 0.0)
    return (
        '<div class="es-cost-basis">'
        '<span>EST. SMART MONEY COST</span>'
        f'<strong>{low} – {high}</strong>'
        f'<small>Mid {mid} • Price {distance} • {state} • conf {confidence:.0f}%</small>'
        f'<em>{evidence} — bukan beneficial-owner cost basis</em>'
        '</div>'
    )


def render_top3_dashboard_html(
    top3: pd.DataFrame,
    *,
    scan_id: str = "",
    as_of: Any = "",
    market_regime: str = "",
) -> str:
    html = _legacy.render_top3_dashboard_html(top3, scan_id=scan_id, as_of=as_of, market_regime=market_regime)
    css = """
    .es-cost-basis{margin-top:8px;padding:8px;border:1px solid #2b6f84;border-radius:8px;background:#092433;text-align:left}
    .es-cost-basis span{display:block;color:#8eb8c8;font-size:8px;font-weight:800;letter-spacing:.5px}
    .es-cost-basis strong{display:block;color:#77f0ba;font-size:13px;margin:3px 0}
    .es-cost-basis small,.es-cost-basis em{display:block;color:#a6c4d1;font-size:8px;font-style:normal;line-height:1.35}
    """
    html = html.replace("</style>", css + "</style>", 1)
    for _, row in top3.iterrows():
        evidence_type = str(row.get("broker_inventory_evidence_type") or "OHLCV_PROXY")
        flow_note = "DIRECT BROKER EVIDENCE" if "DIRECT" in evidence_type else "OHLCV PROXY — BUKAN IDENTITAS BROKER"
        marker = f"</div><p>{escape(flow_note)}</p>"
        replacement = "</div>" + _cost_block(row) + f"<p>{escape(flow_note)}</p>"
        html = html.replace(marker, replacement, 1)
    return html


__all__ = list(getattr(_legacy, "__all__", []))
for _name in (
    "SMART_MONEY_COST_BASIS_VERSION",
    "SCANNER_VERSION",
    "enrich_dashboard_scores",
    "render_top3_dashboard_html",
):
    if _name not in __all__:
        __all__.append(_name)
