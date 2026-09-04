from __future__ import annotations

from html import escape
from typing import Any, Mapping

import numpy as np
import pandas as pd
import top3_dashboard_legacy as _legacy

from release_contract import SCANNER_RELEASE_VERSION
from top3_dashboard_legacy import *  # noqa: F401,F403

SMART_MONEY_COST_BASIS_VERSION = "1.0.0"
TOP3_UI_VERSION = "1.0.0-institutional-ui"
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


_INSTITUTIONAL_CSS = """
/* Presentation-only institutional skin. Calculation authority remains in legacy. */
.es-wrap{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important;color:#e7eef6!important;background:#071019!important;border:1px solid #203244!important;border-radius:14px!important;padding:16px!important;box-shadow:none!important}
.es-title{text-align:left!important;margin:2px 0 0!important;font-size:clamp(23px,3vw,36px)!important;font-weight:720!important;letter-spacing:-.035em!important;color:#f3f7fb!important}.es-title b{color:#58c8b8!important;font-weight:720!important}
.es-method{max-width:none!important;text-align:left!important;margin:10px 0 16px!important;padding:10px 12px!important;background:#0b1621!important;border:1px solid #1d3042!important;border-radius:9px!important;color:#8fa3b6!important;font-size:11px!important;line-height:1.5!important}.es-method strong{color:#c7d6e4!important}.es-method p{margin:5px 0 0!important;color:#7890a4!important;font-size:10px!important}
.es-card{position:relative!important;margin:12px 0!important;padding:13px!important;background:#0a1520!important;border:1px solid #203244!important;border-radius:12px!important;box-shadow:0 8px 24px rgba(0,0,0,.16)!important;overflow:hidden!important}.es-card:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:#51687b}.es-card.rank1:before{background:#18b89f}.es-card.rank2:before{background:#b99a4b}.es-card.rank3:before{background:#5c87ad}.es-card.rank1,.es-card.rank2,.es-card.rank3{border-color:#203244!important}
.es-card-head{display:grid!important;grid-template-columns:54px minmax(190px,2fr) minmax(115px,.72fr) minmax(155px,1fr) minmax(140px,.9fr)!important;gap:9px!important;align-items:stretch!important}
.es-rank{border-radius:9px!important;background:#11231f!important;border:1px solid #245347!important;color:#d9f5ef!important}.rank2 .es-rank{background:#211e13!important;border-color:#584c25!important;color:#f1e3b0!important}.rank3 .es-rank{background:#101d28!important;border-color:#2c4d68!important;color:#d5e7f5!important}.es-rank small{font-size:9px!important;letter-spacing:.08em!important;opacity:.7}.es-rank strong{font-size:30px!important;font-variant-numeric:tabular-nums!important}
.es-identity,.es-score,.es-rec,.es-price{padding:10px 11px!important;border-radius:9px!important;background:#0d1925!important;border:1px solid #203244!important}.es-identity h2{font-size:29px!important;line-height:1!important;letter-spacing:-.025em!important;color:#f5f8fb!important}.es-identity p{margin:5px 0 2px!important;color:#9eb0c0!important;font-size:10px!important}.es-identity em{color:#58c8b8!important;font-size:10px!important;letter-spacing:.04em!important}.es-chip{margin:6px 4px 0 0!important;padding:2px 6px!important;background:#101f2d!important;border:1px solid #294055!important;border-radius:5px!important;color:#9fb3c4!important;font-size:8px!important;font-weight:650!important}
.es-score span,.es-rec span,.es-price span{font-size:8px!important;letter-spacing:.08em!important;color:#778da1!important}.es-score strong{font-size:36px!important;color:#69d7c6!important;font-variant-numeric:tabular-nums!important}.es-score small,.es-rec small{color:#8298aa!important;font-size:8px!important}.es-rec strong{font-size:17px!important;margin:5px 0!important}.es-rec.green strong{color:#63d6a8!important}.es-rec.gold strong{color:#d7bc68!important}.es-rec.orange strong{color:#d99b63!important}.es-rec.blue strong{color:#75a9d4!important}.es-price strong{font-size:20px!important;margin:5px 0!important;font-variant-numeric:tabular-nums!important}.es-price small.up{color:#63d6a8!important}.es-price small.down{color:#d87570!important}.es-price em{font-size:7px!important;margin-top:3px!important}.es-price em.current{color:#71899b!important}.es-price em.stale{color:#d99b63!important}
.es-grid-main{grid-template-columns:1fr 1.2fr .95fr!important;gap:9px!important;margin-top:9px!important}.es-grid-bottom{grid-template-columns:.9fr 1.15fr 1fr!important;gap:9px!important;margin-top:9px!important}.es-panel{background:#0c1823!important;border:1px solid #203244!important;border-radius:9px!important;padding:10px!important}.es-panel h3{text-align:left!important;color:#8fa6b8!important;font-size:9px!important;letter-spacing:.09em!important;margin:0 0 9px!important}
.es-plan-row{padding:5px 0!important;border-bottom:1px solid rgba(83,111,134,.18)!important;font-size:9px!important;color:#91a5b5!important}.es-plan-row b{color:#d8e2e9!important;font-variant-numeric:tabular-nums!important}.es-plan-row.stop b{color:#dc7b75!important}.es-plan-row.target b{color:#69d3a7!important}
.es-factor{grid-template-columns:100px 1fr 26px!important;gap:7px!important;font-size:9px!important;margin:6px 0!important;color:#90a5b6!important}.es-factor b{font-variant-numeric:tabular-nums!important;color:#cdd8e0!important}.es-bar{height:5px!important;background:#152637!important}.es-bar i{background:#28a995!important}
.es-gauge{width:98px!important;height:98px!important;background:conic-gradient(#2bb49e var(--pct),#5b2930 0)!important}.es-gauge:before{width:72px!important;height:72px!important;background:#0c1823!important}.es-gauge strong{font-size:20px!important}.es-gauge small{font-size:7px!important;color:#8399aa!important}.es-flow-stats{font-size:8px!important}.es-flow-stats span{padding:5px!important;background:#101e2b!important;border:1px solid #1c3042!important}.es-flow p{font-size:7px!important;color:#6f879a!important}
.es-report-row{font-size:9px!important;border-bottom:1px solid rgba(83,111,134,.16)!important}.es-report-row b{color:#cfb660!important}.es-reason-summary{margin:0 0 8px!important;padding:6px 8px!important;background:#101e2b!important;color:#819bad!important;font-size:8px!important}.es-reasons{font-size:9px!important}.es-reasons li{padding:4px 0!important}.es-reasons li.positive span{color:#63d6a8!important}.es-reasons li.developing span{color:#d7bc68!important}.es-reasons li.warning span{color:#d99b63!important}.es-highlight p{font-size:9px!important;line-height:1.55!important;color:#c7d3dc!important}.es-highlight small{color:#71889a!important;font-size:8px!important}.es-confidence{padding:6px!important;border-radius:6px!important;background:#101e2b!important;font-size:8px!important;letter-spacing:.05em!important}
.es-cost-basis{margin-top:8px!important;padding:8px 9px!important;border:1px solid #234154!important;border-radius:7px!important;background:#0f1d29!important;text-align:left!important}.es-cost-basis span{display:block;color:#7894a6!important;font-size:7px!important;font-weight:750;letter-spacing:.08em}.es-cost-basis strong{display:block;color:#69d7c6!important;font-size:12px!important;margin:3px 0}.es-cost-basis small,.es-cost-basis em{display:block;color:#8095a6!important;font-size:7px!important;font-style:normal!important;line-height:1.4}
.es-footer{margin-top:12px!important;padding:9px 0 0!important;border-top:1px solid #203244!important;color:#637b8e!important;font-size:8px!important}
@media(max-width:980px){.es-card-head{grid-template-columns:50px 1.5fr 100px!important}.es-rec,.es-price{grid-column:span 1}.es-grid-main,.es-grid-bottom{grid-template-columns:1fr!important}.es-identity h2{font-size:25px!important}}
@media(max-width:640px){.es-wrap{padding:8px!important;border-radius:10px!important}.es-title{font-size:22px!important}.es-method{font-size:9px!important;margin-bottom:10px!important}.es-card{padding:9px!important}.es-card-head{grid-template-columns:42px 1fr!important}.es-score,.es-rec,.es-price{grid-column:span 1!important}.es-score strong{font-size:31px!important}.es-rank strong{font-size:26px!important}.es-factor{grid-template-columns:90px 1fr 24px!important}}
"""


def render_top3_dashboard_html(
    top3: pd.DataFrame,
    *,
    scan_id: str = "",
    as_of: Any = "",
    market_regime: str = "",
) -> str:
    html = _legacy.render_top3_dashboard_html(top3, scan_id=scan_id, as_of=as_of, market_regime=market_regime)
    html = html.replace("</style>", _INSTITUTIONAL_CSS + "</style>", 1)
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
    "TOP3_UI_VERSION",
    "SCANNER_VERSION",
    "enrich_dashboard_scores",
    "render_top3_dashboard_html",
):
    if _name not in __all__:
        __all__.append(_name)
