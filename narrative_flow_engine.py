from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import math
import re

import numpy as np
import pandas as pd

from data_providers import bare_ticker, normalize_ticker


ENGINE_VERSION = "1.0.0-public-narrative-flow"


POSITIVE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "PROJECT_EXPANSION": (
        "project", "proyek", "expansion", "ekspansi", "capacity", "kapasitas", "plant", "pabrik",
        "smelter", "mine", "tambang", "contract", "kontrak", "order book", "data center", "hilirisasi",
        "commissioning", "commercial operation", "joint venture", "akuisisi", "acquisition",
    ),
    "EARNINGS_INFLECTION": (
        "profit", "laba", "revenue", "pendapatan", "margin", "ebitda", "guidance", "turnaround",
        "record high", "rekor", "cash flow", "free cash flow", "dividend", "dividen",
    ),
    "OWNERSHIP_CORPORATE_ACTION": (
        "buyback", "insider buy", "director buy", "pemegang saham", "tender offer", "strategic investor",
        "private placement", "rights issue", "merger", "spin off", "stock split",
    ),
    "POLICY_SECTOR_CATALYST": (
        "policy", "kebijakan", "quota", "kuota", "tariff", "tarif", "export ban", "larangan ekspor",
        "subsidy", "subsidi", "commodity", "komoditas", "government contract", "proyek pemerintah",
    ),
}

NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "default", "gagal bayar", "lawsuit", "gugatan", "fraud", "penipuan", "suspension", "suspensi",
    "bankruptcy", "pailit", "delisting", "investigation", "penyelidikan", "loss widens", "rugi membesar",
    "dilution", "dilusi", "going concern", "restatement", "revisi laporan", "debt restructuring",
)

HYPE_KEYWORDS: tuple[str, ...] = (
    "multibagger", "to the moon", "auto reject atas", "ara", "viral", "hot stock", "must buy",
    "target price naik", "buy now", "cuan besar", "saham gorengan",
)

OFFICIAL_HINTS: tuple[str, ...] = (
    "idx.co.id", "ojk.go.id", "company", "investor relation", "annual report", "financial statement",
    "keterbukaan informasi", "press release", "laporan keuangan",
)


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clip(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    number = _finite(value, low)
    return max(low, min(high, number))


def _safe_div(numerator: Any, denominator: Any, default: float = np.nan) -> float:
    a = _finite(numerator, np.nan)
    b = _finite(denominator, np.nan)
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return default
    return a / b


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain.div(loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame["Close"].shift(1)
    true_range = pd.concat(
        [frame["High"] - frame["Low"], (frame["High"] - previous).abs(), (frame["Low"] - previous).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _cmf(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    spread = (frame["High"] - frame["Low"]).replace(0, np.nan)
    multiplier = ((frame["Close"] - frame["Low"]) - (frame["High"] - frame["Close"])).div(spread)
    money_flow_volume = multiplier.fillna(0) * frame["Volume"]
    return money_flow_volume.rolling(period).sum().div(frame["Volume"].rolling(period).sum().replace(0, np.nan))


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume.fillna(0)).cumsum()


def idx_tick(price: float) -> float:
    if price < 200:
        return 1.0
    if price < 500:
        return 2.0
    if price < 2_000:
        return 5.0
    if price < 5_000:
        return 10.0
    return 25.0


def round_idx(price: float, mode: str = "nearest") -> float:
    if not np.isfinite(price) or price <= 0:
        return np.nan
    tick = idx_tick(price)
    scaled = price / tick
    if mode == "up":
        return float(math.ceil(scaled) * tick)
    if mode == "down":
        return float(math.floor(scaled) * tick)
    return float(round(scaled) * tick)


def _liquidity_score(adtv20: float) -> float:
    if adtv20 >= 100_000_000_000:
        return 95.0
    if adtv20 >= 25_000_000_000:
        return 85.0
    if adtv20 >= 7_500_000_000:
        return 72.0
    if adtv20 >= 2_000_000_000:
        return 55.0
    if adtv20 >= 500_000_000:
        return 38.0
    if adtv20 > 0:
        return 20.0
    return 0.0


def calculate_market_features(frame: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> dict[str, Any]:
    if frame is None or frame.empty or len(frame) < 60:
        return {"feature_state": "INSUFFICIENT_HISTORY", "history_bars": 0 if frame is None else len(frame)}
    local = frame.copy().sort_index()
    close = pd.to_numeric(local["Close"], errors="coerce")
    high = pd.to_numeric(local["High"], errors="coerce")
    low = pd.to_numeric(local["Low"], errors="coerce")
    volume = pd.to_numeric(local["Volume"], errors="coerce").fillna(0)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    rsi14 = _rsi(close)
    atr14 = _atr(local)
    cmf20 = _cmf(local)
    obv = _obv(close, volume)
    value = close * volume
    volume_ma20 = volume.rolling(20).mean()
    range_ = (high - low).replace(0, np.nan)
    clv = ((close - low) / range_).clip(0, 1)
    daily_return = close.pct_change()
    up_value = value.where(daily_return > 0, 0)
    down_value = value.where(daily_return < 0, 0)
    adtv20 = _finite(value.tail(20).mean(), 0.0)
    vol_ratio = _safe_div(volume.iloc[-1], volume_ma20.iloc[-1], 0.0)
    up_value_ratio = _safe_div(up_value.tail(20).sum(), up_value.tail(20).sum() + down_value.tail(20).sum(), 0.5)
    obv_slope = _safe_div(obv.iloc[-1] - obv.iloc[-21], abs(obv.iloc[-21]) + volume.tail(20).sum(), 0.0) if len(obv) >= 21 else 0.0
    close_acceptance = _finite(clv.tail(20).mul(volume.tail(20)).sum() / max(volume.tail(20).sum(), 1.0), 0.5)
    median_range = range_.rolling(20).median()
    high_volume = volume > volume_ma20 * 1.2
    absorption = high_volume & (range_ <= median_range * 1.05) & (clv >= 0.62)
    failed_absorption = high_volume & (clv <= 0.35)
    distribution = (daily_return < 0) & high_volume & (clv <= 0.45)
    accumulation = (daily_return > 0) & high_volume & (clv >= 0.60)
    pullback_mask = daily_return < 0
    pullback_volume_ratio = _safe_div(volume.tail(20)[pullback_mask.tail(20)].mean(), volume_ma20.iloc[-1], np.nan)
    high20 = _finite(high.tail(20).max(), np.nan)
    high55 = _finite(high.tail(55).max(), np.nan)
    low20 = _finite(low.tail(20).min(), np.nan)
    latest_close = _finite(close.iloc[-1], np.nan)
    latest_atr = _finite(atr14.iloc[-1], np.nan)
    latest_ema20 = _finite(ema20.iloc[-1], np.nan)
    latest_ema50 = _finite(ema50.iloc[-1], np.nan)
    latest_ema200 = _finite(ema200.iloc[-1], np.nan)
    latest_rsi = _finite(rsi14.iloc[-1], np.nan)
    latest_cmf = _finite(cmf20.iloc[-1], np.nan)
    momentum20 = 100 * _safe_div(latest_close, close.iloc[-21], np.nan) - 100 if len(close) >= 21 else np.nan
    momentum60 = 100 * _safe_div(latest_close, close.iloc[-61], np.nan) - 100 if len(close) >= 61 else np.nan
    rs60 = np.nan
    if benchmark is not None and not benchmark.empty and len(benchmark) >= 61:
        aligned = pd.concat([close.rename("stock"), benchmark["Close"].rename("bench")], axis=1).dropna()
        if len(aligned) >= 61:
            stock_return = aligned["stock"].iloc[-1] / aligned["stock"].iloc[-61] - 1
            bench_return = aligned["bench"].iloc[-1] / aligned["bench"].iloc[-61] - 1
            rs60 = 100 * (stock_return - bench_return)
    trend_score = np.mean([
        100.0 if latest_close > latest_ema20 else 20.0,
        100.0 if latest_ema20 > latest_ema50 else 25.0,
        100.0 if latest_ema50 > latest_ema200 else 30.0,
        _clip(50 + 1.2 * _finite(momentum20, 0)),
        _clip(50 + 1.0 * _finite(rs60, 0)),
    ])
    flow_score = (
        0.18 * _clip(100 * up_value_ratio)
        + 0.16 * _clip(50 + 220 * latest_cmf)
        + 0.12 * _clip(50 + 600 * obv_slope)
        + 0.14 * _clip(100 * close_acceptance)
        + 0.14 * _clip(50 + 8 * int(accumulation.tail(20).sum()) - 9 * int(distribution.tail(20).sum()))
        + 0.12 * _clip(50 + 10 * int(absorption.tail(20).sum()) - 14 * int(failed_absorption.tail(20).sum()))
        + 0.08 * _clip(70 if np.isfinite(pullback_volume_ratio) and pullback_volume_ratio < 0.85 else 45)
        + 0.06 * _liquidity_score(adtv20)
    )
    distribution_score = _clip(
        12 * int(distribution.tail(20).sum())
        + 15 * int(failed_absorption.tail(20).sum())
        + max(0.0, 55 - 100 * close_acceptance),
    )
    extension_atr = _safe_div(latest_close - latest_ema20, latest_atr, np.nan)
    breakout_distance_pct = 100 * _safe_div(latest_close - high55, high55, np.nan)
    crowding = _clip(
        35
        + 9 * max(0.0, _finite(extension_atr, 0) - 1.5)
        + 0.9 * max(0.0, _finite(latest_rsi, 50) - 65)
        + 0.8 * max(0.0, _finite(vol_ratio, 1) - 2.0) * 10,
    )
    if distribution_score >= 60:
        price_stage = "DISTRIBUTION"
    elif latest_close > latest_ema20 > latest_ema50 and momentum20 > 3:
        price_stage = "MARKUP"
    elif latest_close >= latest_ema50 and flow_score >= 58:
        price_stage = "ACCUMULATION"
    elif latest_close < latest_ema50 and latest_ema20 < latest_ema50:
        price_stage = "MARKDOWN"
    else:
        price_stage = "BASE_OR_TRANSITION"
    return {
        "feature_state": "OK",
        "history_bars": len(local),
        "last_date": local.index[-1].date().isoformat(),
        "last_price": round(latest_close, 4),
        "ema20": round(latest_ema20, 4),
        "ema50": round(latest_ema50, 4),
        "ema200": round(latest_ema200, 4),
        "atr14": round(latest_atr, 4),
        "rsi14": round(latest_rsi, 2),
        "cmf20": round(latest_cmf, 4) if np.isfinite(latest_cmf) else np.nan,
        "momentum20_pct": round(momentum20, 2) if np.isfinite(momentum20) else np.nan,
        "momentum60_pct": round(momentum60, 2) if np.isfinite(momentum60) else np.nan,
        "relative_strength60_pct": round(rs60, 2) if np.isfinite(rs60) else np.nan,
        "adtv20_idr": round(adtv20, 2),
        "liquidity_score": round(_liquidity_score(adtv20), 1),
        "volume_ratio20": round(vol_ratio, 3),
        "up_value_ratio20_pct": round(100 * up_value_ratio, 1),
        "close_acceptance20_pct": round(100 * close_acceptance, 1),
        "accumulation_days20": int(accumulation.tail(20).sum()),
        "distribution_days20": int(distribution.tail(20).sum()),
        "absorption_days20": int(absorption.tail(20).sum()),
        "failed_absorption_days20": int(failed_absorption.tail(20).sum()),
        "pullback_volume_ratio": round(pullback_volume_ratio, 3) if np.isfinite(pullback_volume_ratio) else np.nan,
        "trend_score": round(_clip(trend_score), 1),
        "smart_money_score": round(_clip(flow_score), 1),
        "smart_money_coverage_pct": 85.0 if len(local) >= 220 else round(85 * len(local) / 220, 1),
        "distribution_score": round(distribution_score, 1),
        "crowding_score": round(crowding, 1),
        "extension_atr": round(extension_atr, 2) if np.isfinite(extension_atr) else np.nan,
        "breakout_distance_55d_pct": round(breakout_distance_pct, 2) if np.isfinite(breakout_distance_pct) else np.nan,
        "price_stage": price_stage,
        "high20": round(high20, 4),
        "high55": round(high55, 4),
        "low20": round(low20, 4),
    }


def aggregate_broker_summary(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if frame is None or frame.empty:
        return {}
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    if "ticker" not in local.columns:
        return {}
    local["ticker"] = local["ticker"].map(normalize_ticker)
    for column in ("buy_value", "sell_value", "buy_volume", "sell_volume"):
        if column in local.columns:
            local[column] = pd.to_numeric(local[column], errors="coerce").fillna(0)
    output: dict[str, dict[str, Any]] = {}
    for ticker, group in local.groupby("ticker"):
        buy = group["buy_value"].sum() if "buy_value" in group else group.get("buy_volume", pd.Series(dtype=float)).sum()
        sell = group["sell_value"].sum() if "sell_value" in group else group.get("sell_volume", pd.Series(dtype=float)).sum()
        gross = buy + sell
        net_ratio = _safe_div(buy - sell, gross, np.nan)
        verified = bool(group.get("source_verified", pd.Series(False)).astype(str).str.lower().isin({"true", "1", "yes"}).all())
        provenance = "DIRECT_SOURCE_VERIFIED" if verified else "OBSERVED_UNVERIFIED"
        score = _clip(50 + 250 * _finite(net_ratio, 0)) if np.isfinite(net_ratio) else np.nan
        output[ticker] = {
            "broker_summary_score": round(score, 1) if np.isfinite(score) else np.nan,
            "broker_summary_coverage_pct": round(min(100.0, 10.0 * len(group)), 1),
            "broker_net_ratio": round(net_ratio, 4) if np.isfinite(net_ratio) else np.nan,
            "broker_summary_provenance_state": provenance,
        }
    return output


def _event_text(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in ("title", "summary", "category", "publisher", "source_tier", "url")).lower()


def _categorize(text: str) -> tuple[str, int]:
    best_category = "GENERAL_MARKET_NEWS"
    best_hits = 0
    for category, keywords in POSITIVE_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits > best_hits:
            best_hits = hits
            best_category = category
    return best_category, best_hits


def score_narrative_events(events: pd.DataFrame | None, as_of: Any = None) -> dict[str, Any]:
    if events is None or events.empty:
        return {
            "narrative_score": np.nan,
            "narrative_coverage_pct": 0.0,
            "narrative_state": "NO_ACTIVE_PUBLIC_NARRATIVE",
            "narrative_event_count": 0,
            "narrative_category": "NONE",
            "narrative_latest_title": "",
            "narrative_risk_flags": "NO_SOURCED_EVENT",
        }
    local = events.copy()
    local["published_at"] = pd.to_datetime(
        local.get("published_at", local.get("event_date")), errors="coerce", utc=True,
    )
    now = pd.Timestamp(as_of, tz="UTC") if as_of is not None and pd.Timestamp(as_of).tzinfo is None else pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    rows: list[dict[str, Any]] = []
    for _, row in local.iterrows():
        text = _event_text(row)
        category, hits = _categorize(text)
        negative_hits = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text)
        hype_hits = sum(1 for keyword in HYPE_KEYWORDS if keyword in text)
        published = row.get("published_at")
        age_days = max(0.0, (now - published).total_seconds() / 86400) if pd.notna(published) else 90.0
        freshness = 100 * math.exp(-age_days / 45.0)
        explicit_materiality = _finite(row.get("materiality_score"), np.nan)
        explicit_bridge = _finite(row.get("financial_bridge_score"), np.nan)
        official = any(hint in text for hint in OFFICIAL_HINTS) or str(row.get("source_tier") or "").upper() in {"OFFICIAL", "ISSUER", "REGULATOR"}
        materiality = explicit_materiality if np.isfinite(explicit_materiality) else _clip(38 + 12 * hits + 12 * int(official) - 18 * negative_hits)
        bridge = explicit_bridge if np.isfinite(explicit_bridge) else _clip(32 + 11 * hits + 8 * int(category in {"EARNINGS_INFLECTION", "PROJECT_EXPANSION"}) - 15 * negative_hits)
        quality = 90.0 if official else 62.0 if row.get("url") else 45.0
        event_score = _clip(0.30 * freshness + 0.30 * materiality + 0.25 * bridge + 0.15 * quality - 20 * negative_hits - 8 * hype_hits)
        rows.append({
            "event_score": event_score,
            "category": category,
            "negative_hits": negative_hits,
            "hype_hits": hype_hits,
            "published_at": published,
            "title": str(row.get("title") or ""),
            "publisher": str(row.get("publisher") or ""),
            "official": official,
        })
    scored = pd.DataFrame(rows).sort_values("event_score", ascending=False)
    top = scored.head(5)
    weighted = np.average(top["event_score"], weights=np.linspace(1.0, 0.5, len(top))) if len(top) else np.nan
    unique_publishers = max(1, scored["publisher"].replace("", np.nan).nunique(dropna=True))
    official_count = int(scored["official"].sum())
    coverage = _clip(18 * min(5, len(scored)) + 8 * min(3, unique_publishers) + 10 * min(2, official_count))
    negative_total = int(scored["negative_hits"].sum())
    hype_total = int(scored["hype_hits"].sum())
    narrative_score = 50 + (_clip(weighted) - 50) * coverage / 100 if np.isfinite(weighted) else np.nan
    top_category = str(top.iloc[0]["category"]) if len(top) else "NONE"
    if negative_total >= 2:
        state = "NEGATIVE_OR_CONTRADICTED_NARRATIVE"
    elif hype_total >= 2 and narrative_score >= 60:
        state = "CROWDED_PUBLIC_NARRATIVE"
    elif narrative_score >= 70 and coverage >= 55:
        state = "MATERIAL_NARRATIVE_CONFIRMED"
    elif narrative_score >= 58:
        state = "EARLY_OR_PARTIAL_NARRATIVE"
    else:
        state = "WEAK_OR_UNCONVERTED_NARRATIVE"
    risks = []
    if negative_total:
        risks.append("NEGATIVE_EVENT_PRESENT")
    if hype_total:
        risks.append("HYPE_LANGUAGE_PRESENT")
    if official_count == 0:
        risks.append("NO_OFFICIAL_SOURCE")
    return {
        "narrative_score": round(_clip(narrative_score), 1) if np.isfinite(narrative_score) else np.nan,
        "narrative_coverage_pct": round(coverage, 1),
        "narrative_state": state,
        "narrative_event_count": len(scored),
        "narrative_category": top_category,
        "narrative_latest_title": str(scored.sort_values("published_at", ascending=False, na_position="last").iloc[0]["title"]) if len(scored) else "",
        "narrative_risk_flags": " | ".join(risks) or "NO_MAJOR_NARRATIVE_RISK",
    }


def parse_ownership(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if frame is None or frame.empty:
        return {}
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    if "ticker" not in local.columns:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for _, row in local.iterrows():
        ticker = normalize_ticker(row.get("ticker"))
        free_float = _finite(row.get("free_float_pct"), np.nan)
        alignment = _finite(row.get("owner_alignment_score"), np.nan)
        insider = str(row.get("insider_buy_flag") or "").strip().lower() in {"true", "1", "yes"}
        controller_change = str(row.get("controller_change_flag") or "").strip().lower() in {"true", "1", "yes"}
        coverage = 0.0
        components = []
        if np.isfinite(alignment):
            components.append((alignment, 0.65))
            coverage += 60
        if np.isfinite(free_float):
            free_float_score = 80 if 10 <= free_float <= 35 else 62 if 5 <= free_float < 10 or 35 < free_float <= 50 else 35
            components.append((free_float_score, 0.20))
            coverage += 25
        if insider:
            components.append((80, 0.10))
            coverage += 10
        if controller_change:
            components.append((45, 0.05))
            coverage += 5
        weight = sum(item[1] for item in components)
        score = sum(value * item_weight for value, item_weight in components) / weight if weight else np.nan
        output[ticker] = {
            "ownership_score": round(_clip(score), 1) if np.isfinite(score) else np.nan,
            "ownership_coverage_pct": round(_clip(coverage), 1),
            "free_float_pct": round(free_float, 2) if np.isfinite(free_float) else np.nan,
            "ownership_note": str(row.get("ownership_note") or ""),
        }
    return output


def classify_lifecycle(features: Mapping[str, Any], narrative: Mapping[str, Any]) -> str:
    flow = _finite(features.get("smart_money_score"), np.nan)
    narrative_score = _finite(narrative.get("narrative_score"), np.nan)
    narrative_coverage = _finite(narrative.get("narrative_coverage_pct"), 0)
    distribution = _finite(features.get("distribution_score"), 0)
    crowding = _finite(features.get("crowding_score"), 0)
    price_stage = str(features.get("price_stage") or "")
    if distribution >= 60 or price_stage == "DISTRIBUTION":
        return "DISTRIBUTION_OR_BROKEN"
    if crowding >= 68 and np.isfinite(narrative_score) and narrative_score >= 60:
        return "CROWDED_HYPE"
    if np.isfinite(narrative_score) and narrative_score >= 58 and (not np.isfinite(flow) or flow < 52):
        return "STORY_AHEAD_OF_FLOW"
    if np.isfinite(flow) and flow >= 62 and narrative_coverage < 25:
        return "FLOW_AHEAD_OF_STORY"
    if np.isfinite(flow) and flow >= 62 and np.isfinite(narrative_score) and narrative_score >= 58:
        return "EXPANSION_CONFIRMED" if price_stage == "MARKUP" else "EARLY_NARRATIVE_FLOW_CONVERGENCE"
    if np.isfinite(flow) and flow >= 55 and price_stage in {"ACCUMULATION", "BASE_OR_TRANSITION"}:
        return "ACCUMULATION_BUILDING"
    return "NO_CONVERGENCE_YET"


def build_execution_plan(features: Mapping[str, Any], production_ready: bool) -> dict[str, Any]:
    close = _finite(features.get("last_price"), np.nan)
    atr = _finite(features.get("atr14"), np.nan)
    ema20 = _finite(features.get("ema20"), np.nan)
    high20 = _finite(features.get("high20"), np.nan)
    low20 = _finite(features.get("low20"), np.nan)
    stage = str(features.get("price_stage") or "")
    if not all(np.isfinite(value) for value in (close, atr, ema20, high20, low20)) or atr <= 0:
        return {"execution_state": "NO_VALID_PLAN"}
    if stage == "MARKUP":
        trigger = round_idx(max(close, high20), "up")
        entry_low = round_idx(trigger, "nearest")
        entry_high = round_idx(trigger + 0.35 * atr, "up")
    else:
        trigger = round_idx(high20, "up")
        entry_low = round_idx(max(low20, ema20 - 0.35 * atr), "down")
        entry_high = round_idx(min(trigger, max(close, ema20 + 0.25 * atr)), "up")
    if entry_high < entry_low:
        entry_low, entry_high = entry_high, entry_low
    entry_mid = (entry_low + entry_high) / 2
    structural_stop = min(low20, entry_low - 1.15 * atr)
    stop = round_idx(structural_stop, "down")
    if stop >= entry_low:
        stop = round_idx(entry_low - 1.2 * atr, "down")
    risk = entry_mid - stop
    if risk <= 0:
        return {"execution_state": "NO_VALID_PLAN"}
    tp1 = round_idx(entry_mid + 2.0 * risk, "up")
    tp2 = round_idx(entry_mid + 3.0 * risk, "up")
    if not (stop < entry_low <= entry_high < tp1 < tp2):
        return {"execution_state": "NO_VALID_PLAN"}
    return {
        "execution_state": "READY_WITH_TRIGGER" if production_ready else "RESEARCH_PLAN_ONLY",
        "entry_low": entry_low,
        "entry_high": entry_high,
        "trigger": trigger,
        "stop_loss": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr_tp1": round((tp1 - entry_mid) / risk, 2),
        "rr_tp2": round((tp2 - entry_mid) / risk, 2),
    }


def build_public_method_profile(
    *,
    ticker: str,
    features: Mapping[str, Any],
    narrative: Mapping[str, Any] | None = None,
    broker: Mapping[str, Any] | None = None,
    ownership: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    narrative = dict(narrative or {})
    broker = dict(broker or {})
    ownership = dict(ownership or {})
    flow = _finite(features.get("smart_money_score"), np.nan)
    flow_coverage = _finite(features.get("smart_money_coverage_pct"), 0)
    broker_score = _finite(broker.get("broker_summary_score"), np.nan)
    broker_coverage = _finite(broker.get("broker_summary_coverage_pct"), 0)
    if np.isfinite(broker_score):
        flow = 0.75 * flow + 0.25 * broker_score if np.isfinite(flow) else broker_score
        flow_coverage = min(100.0, 0.75 * flow_coverage + 0.25 * broker_coverage)
    narrative_score = _finite(narrative.get("narrative_score"), np.nan)
    narrative_coverage = _finite(narrative.get("narrative_coverage_pct"), 0)
    ownership_score = _finite(ownership.get("ownership_score"), np.nan)
    ownership_coverage = _finite(ownership.get("ownership_coverage_pct"), 0)
    trend = _finite(features.get("trend_score"), np.nan)
    liquidity = _finite(features.get("liquidity_score"), 0)
    alignment_score = np.nan
    alignment_coverage = 0.0
    if np.isfinite(narrative_score) and np.isfinite(flow):
        alignment_score = _clip(0.55 * narrative_score + 0.45 * flow)
        alignment_coverage = min(narrative_coverage, flow_coverage)
    components = [
        (flow, 0.38, flow_coverage),
        (narrative_score, 0.27, narrative_coverage),
        (trend, 0.12, 100.0 if np.isfinite(trend) else 0.0),
        (liquidity, 0.10, 100.0),
        (alignment_score, 0.08, alignment_coverage),
        (ownership_score, 0.05, ownership_coverage),
    ]
    available_weight = sum(weight for value, weight, _ in components if np.isfinite(value))
    raw = sum(value * weight for value, weight, _ in components if np.isfinite(value)) / available_weight if available_weight else np.nan
    coverage = sum(weight * coverage_value for value, weight, coverage_value in components if np.isfinite(value)) / max(available_weight, 1e-9)
    distribution = _finite(features.get("distribution_score"), 0)
    crowding = _finite(features.get("crowding_score"), 0)
    risks: list[str] = []
    if distribution >= 45:
        risks.append("DISTRIBUTION_RISK")
    if crowding >= 65:
        risks.append("CROWDED_LATE_ENTRY")
    if narrative_coverage < 30:
        risks.append("NARRATIVE_EVIDENCE_WEAK")
    if flow_coverage < 55:
        risks.append("FLOW_EVIDENCE_WEAK")
    if liquidity < 45:
        risks.append("LIQUIDITY_RISK")
    penalty = 0.22 * distribution + 0.12 * max(0.0, crowding - 55)
    conviction = _clip(raw - penalty) if np.isfinite(raw) else np.nan
    lifecycle = classify_lifecycle(features, narrative)
    production_ready = bool(
        np.isfinite(conviction)
        and conviction >= 65
        and coverage >= 58
        and narrative_coverage >= 30
        and flow_coverage >= 55
        and liquidity >= 45
        and distribution < 55
        and crowding < 68
        and lifecycle in {"EARLY_NARRATIVE_FLOW_CONVERGENCE", "EXPANSION_CONFIRMED"}
    )
    if production_ready and conviction >= 76:
        state = "PUBLIC_FRAMEWORK_READY"
    elif distribution >= 60 or lifecycle == "DISTRIBUTION_OR_BROKEN":
        state = "PUBLIC_FRAMEWORK_REJECT"
    elif coverage < 58 or narrative_coverage < 30:
        state = "PUBLIC_FRAMEWORK_EVIDENCE_PENDING"
    else:
        state = "PUBLIC_FRAMEWORK_WATCH"
    if state == "PUBLIC_FRAMEWORK_READY":
        action = "READY_WITH_TRIGGER"
    elif lifecycle == "ACCUMULATION_BUILDING":
        action = "WATCH_ACCUMULATION"
    elif lifecycle in {"STORY_AHEAD_OF_FLOW", "FLOW_AHEAD_OF_STORY", "NO_CONVERGENCE_YET"}:
        action = "WAIT_FOR_CONVERGENCE"
    elif lifecycle == "CROWDED_HYPE":
        action = "AVOID_CHASING"
    else:
        action = "REJECT_OR_EXIT_WATCH"
    if not production_ready:
        position_cap = 0.0
    elif conviction >= 82 and liquidity >= 70:
        position_cap = 12.0
    elif conviction >= 74:
        position_cap = 8.0
    else:
        position_cap = 5.0
    plan = build_execution_plan(features, production_ready)
    return {
        "ticker": normalize_ticker(ticker),
        "engine_version": ENGINE_VERSION,
        "framework_disclaimer": "PUBLIC_CLEAN_ROOM_RECONSTRUCTION_NOT_AFFILIATED_OR_PROPRIETARY",
        **dict(features),
        **narrative,
        **broker,
        **ownership,
        "narrative_flow_lifecycle": lifecycle,
        "narrative_flow_conviction_score": round(conviction, 1) if np.isfinite(conviction) else np.nan,
        "narrative_flow_coverage_pct": round(_clip(coverage), 1),
        "public_method_state": state,
        "production_ready": production_ready,
        "action": action,
        "risk_flags": " | ".join(risks) or "NO_MAJOR_PUBLIC_FRAMEWORK_RISK",
        "position_cap_pct": position_cap,
        **plan,
    }


def make_scan_id(as_of: Any, tickers: list[str]) -> str:
    payload = f"{pd.Timestamp(as_of).isoformat()}|{'|'.join(sorted(tickers))}|{ENGINE_VERSION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


__all__ = [
    "ENGINE_VERSION",
    "aggregate_broker_summary",
    "build_public_method_profile",
    "calculate_market_features",
    "make_scan_id",
    "parse_ownership",
    "score_narrative_events",
]
