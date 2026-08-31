from __future__ import annotations

"""Consumer for the Emir-owned official IDX participant-flow cache."""

from io import BytesIO
import gzip
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

PUBLIC_CACHE_URL = "https://raw.githubusercontent.com/rizanrizan93/Cuan-maksimal/main/data/public_broker_flow_30d.csv.gz"
VERSION = "1.0.1-emir-owned-public-idx-participant-cache"
SOURCE_NAME = "IDX_PUBLIC_TRADE_DETAIL_PUBLIK"

_BROKER_BASE_COLUMNS = frozenset({
    "broker_base_smart_money_score",
    "broker_base_emir_conviction_score",
    "broker_base_emir_final_score",
})
_BROKER_LEGACY_APPLIED_MARKERS = frozenset({
    "broker_emir_conviction_delta",
    "broker_confirmation_weight_pct",
    "broker_pre_confirmation_smart_money_score",
    "broker_post_confirmation_smart_money_score",
})


def _broker_row_has_value(row: Any, column: str) -> bool:
    value = row.get(column)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _classify_broker_enrichment_row(row: Any) -> str:
    """Resolve broker lineage from values carried by one decision row only."""
    explicit_value = row.get("broker_enrichment_compatibility_state")
    explicit = str(explicit_value).strip().upper() if _broker_row_has_value(row, "broker_enrichment_compatibility_state") else ""
    base_count = sum(_broker_row_has_value(row, column) for column in _BROKER_BASE_COLUMNS)
    marker_present = any(_broker_row_has_value(row, column) for column in _BROKER_LEGACY_APPLIED_MARKERS)
    if explicit.startswith("LEGACY_ALREADY_ENRICHED"):
        return "LEGACY_ALREADY_ENRICHED"
    if explicit.startswith("AMBIGUOUS_LEGACY"):
        return "AMBIGUOUS_LEGACY"
    if explicit.startswith("CANONICAL_ENRICHED") and base_count != len(_BROKER_BASE_COLUMNS):
        return "AMBIGUOUS_LEGACY"
    if explicit.startswith("PRISTINE_BASE"):
        return "AMBIGUOUS_LEGACY" if marker_present or 0 < base_count < len(_BROKER_BASE_COLUMNS) else "PRISTINE_BASE"
    if base_count == len(_BROKER_BASE_COLUMNS):
        return "CANONICAL_ENRICHED"
    if 0 < base_count < len(_BROKER_BASE_COLUMNS):
        return "AMBIGUOUS_LEGACY"
    if marker_present:
        return "LEGACY_ALREADY_ENRICHED"
    return "PRISTINE_BASE"


def classify_broker_enrichment_state(frame: pd.DataFrame) -> str:
    """Return one compatible scalar state; mixed frames aggregate fail-closed."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "PRISTINE_BASE"
    states = frame.apply(lambda row: _classify_broker_enrichment_row(row), axis=1)
    unique = states.dropna().unique().tolist()
    return unique[0] if len(unique) == 1 else "AMBIGUOUS_LEGACY"


def _canon(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _num(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def load_public_cache() -> pd.DataFrame:
    try:
        response = requests.get(PUBLIC_CACHE_URL, timeout=20, headers={"User-Agent": "Emir-Scanner/1.0"})
        response.raise_for_status()
        return _normalize(pd.read_csv(BytesIO(gzip.decompress(response.content))))
    except Exception:
        return _normalize(pd.DataFrame())


def _normalize(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["trade_date", "ticker", "broker_code", "buy_value", "sell_value", "buy_volume", "sell_volume", "buy_avg", "sell_avg", "net_value", "net_volume", "gross_value", "source", "source_verified", "provenance_state", "side", "net_rank"]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    out["ticker"] = out["ticker"].map(_canon)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    for column in ("buy_value", "sell_value", "buy_volume", "sell_volume", "buy_avg", "sell_avg", "net_value", "net_volume", "gross_value", "net_rank"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["ticker", "trade_date"]).drop_duplicates(["trade_date", "ticker", "broker_code", "side"], keep="last")


def _cross_section(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()
    score = pd.Series(np.nan, index=values.index, dtype=float)
    count = int(valid.sum())
    if count >= 3:
        score.loc[valid] = values.loc[valid].rank(pct=True) * 100.0
    elif count > 0:
        score.loc[valid] = 50.0
    return score


def score_broker_history(history: pd.DataFrame, universe: Iterable[Any]) -> pd.DataFrame:
    frame = _normalize(history)
    names = sorted({_canon(value) for value in universe if _canon(value)})
    rows: list[dict[str, Any]] = []
    if "ticker" in frame.columns and not frame.empty:
        frame = frame[frame["ticker"].isin(names)]
    for ticker in names:
        local = frame[frame["ticker"].eq(ticker)].copy() if "ticker" in frame.columns else pd.DataFrame()
        if local.empty:
            rows.append({"ticker": ticker, "broker_flow_coverage_pct": 0.0, "broker_accumulation_state": "NO_DATA", "broker_flow_version": VERSION})
            continue
        dates = sorted(local["trade_date"].dropna().unique())[-20:]
        recent = local[local["trade_date"].isin(dates)]
        buyers = recent[recent["side"].eq("TOP_NET_BUYER")]
        sellers = recent[recent["side"].eq("TOP_NET_SELLER")]
        top3 = buyers.sort_values(["trade_date", "net_rank"]).groupby("trade_date").head(3)
        counts = top3["broker_code"].value_counts() if not top3.empty else pd.Series(dtype=float)
        top_broker = str(counts.index[0]) if not counts.empty else ""
        persistence = float(counts.iloc[0]) / max(1, len(dates)) * 100.0 if not counts.empty else np.nan
        broker_net = buyers.groupby("broker_code")["net_value"].sum().sort_values(ascending=False) if not buyers.empty else pd.Series(dtype=float)
        top_net = float(broker_net.iloc[0]) if len(broker_net) else np.nan
        positive_total = float(buyers["net_value"].clip(lower=0).sum()) if not buyers.empty else 0.0
        concentration = top_net / positive_total * 100.0 if top_net > 0 and positive_total > 0 else np.nan
        positive = float(buyers["net_value"].sum()) if not buyers.empty else 0.0
        negative = float(sellers["net_value"].abs().sum()) if not sellers.empty else 0.0
        dominance_score = float(np.clip(50.0 + 15.0 * (positive / negative if negative > 0 else 5.0), 0, 100)) if positive > 0 else 30.0
        latest_date = max(dates) if dates else pd.NaT
        latest = local[local["trade_date"].eq(latest_date)] if pd.notna(latest_date) else pd.DataFrame()
        latest_buy = latest[latest["side"].eq("TOP_NET_BUYER")].sort_values("net_rank")
        latest_broker = str(latest_buy.iloc[0]["broker_code"]) if not latest_buy.empty else ""
        latest_avg = _num(latest_buy.iloc[0].get("buy_avg")) if not latest_buy.empty else np.nan
        rows.append({"ticker": ticker, "broker_flow_observed_days": int(recent["trade_date"].nunique()), "broker_flow_latest_date": latest_date, "broker_top_buyer_code": top_broker, "broker_latest_top_buyer_code": latest_broker, "broker_top3_buyer_persistence_20d_pct": persistence, "broker_top_buyer_net_value_20d": top_net, "broker_buyer_concentration_pct": concentration, "broker_buy_sell_dominance_score": dominance_score, "broker_latest_top_buyer_buy_avg": latest_avg, "broker_flow_coverage_pct": min(100.0, 100.0 * len(dates) / 20.0), "broker_flow_source": SOURCE_NAME, "broker_flow_provenance": "OFFICIAL_IDX_PUBLIC_EOD_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER", "broker_flow_version": VERSION})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "broker_top_buyer_net_value_20d" not in out.columns:
        out["broker_net_score"] = np.nan
        out["broker_accumulation_score"] = np.nan
        out["broker_smart_money_confirmation_score"] = np.nan
        return out
    out["broker_net_score"] = _cross_section(out["broker_top_buyer_net_value_20d"])
    out["broker_accumulation_score"] = (0.35 * out["broker_net_score"].fillna(50.0) + 0.25 * pd.to_numeric(out["broker_top3_buyer_persistence_20d_pct"], errors="coerce").fillna(0) + 0.20 * pd.to_numeric(out["broker_buyer_concentration_pct"], errors="coerce").fillna(0) + 0.20 * pd.to_numeric(out["broker_buy_sell_dominance_score"], errors="coerce").fillna(50)).clip(0, 100).round(1)
    out["broker_smart_money_confirmation_score"] = (0.70 * out["broker_accumulation_score"] + 0.30 * out["broker_net_score"].fillna(50)).clip(0, 100).round(1)
    out["broker_accumulation_state"] = np.select([out["broker_accumulation_score"].ge(70), out["broker_accumulation_score"].le(35)], ["PARTICIPANT_ACCUMULATION", "PARTICIPANT_DISTRIBUTION"], default="PARTICIPANT_MIXED")
    return out


def _merge(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_key"] = out["ticker"].map(_canon)
    right = features.copy()
    right["_key"] = right["ticker"].map(_canon)
    right = right.drop(columns=["ticker"]).drop_duplicates("_key")
    duplicates = [c for c in right.columns if c != "_key" and c in out.columns]
    if duplicates:
        out = out.drop(columns=duplicates)
    return out.merge(right, on="_key", how="left").drop(columns=["_key"])


def enrich_emir_broker(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    working = frame.copy(deep=True)
    order_column = "__broker_enrichment_row_order__"
    while order_column in working.columns:
        order_column += "_"
    working[order_column] = np.arange(len(working))
    states = working.apply(lambda row: _classify_broker_enrichment_row(row), axis=1)
    enrichable = states.eq("PRISTINE_BASE")
    protected = working.loc[~enrichable].copy()
    if not protected.empty:
        protected["broker_enrichment_compatibility_state"] = states.loc[~enrichable].map({
            "CANONICAL_ENRICHED": "CANONICAL_ENRICHED",
            "LEGACY_ALREADY_ENRICHED": "LEGACY_ALREADY_ENRICHED_PRESERVED_NO_REAPPLY",
            "AMBIGUOUS_LEGACY": "AMBIGUOUS_LEGACY_PRESERVED_NO_REAPPLY",
        })
    if not enrichable.any():
        return protected.sort_values(order_column, kind="stable").drop(columns=[order_column]).reset_index(drop=True)

    enrichable_rows = working.loc[enrichable].copy()
    features = score_broker_history(load_public_cache(), enrichable_rows["ticker"].tolist())
    if features.empty:
        combined = pd.concat([enrichable_rows, protected], ignore_index=True, sort=False)
        return combined.sort_values(order_column, kind="stable").drop(columns=[order_column]).reset_index(drop=True)
    out = _merge(enrichable_rows, features)
    for base_column, source_column in {
        "broker_base_smart_money_score": "smart_money_score",
        "broker_base_emir_conviction_score": "emir_conviction_score",
        "broker_base_emir_final_score": "emir_final_score",
    }.items():
        source = out.get(source_column, pd.Series(np.nan, index=out.index))
        if base_column not in out.columns:
            out[base_column] = source
        else:
            present = out.apply(lambda row: _broker_row_has_value(row, base_column), axis=1)
            out.loc[~present, base_column] = source.loc[~present]
    base = pd.to_numeric(out["broker_base_smart_money_score"], errors="coerce")
    broker = pd.to_numeric(out.get("broker_smart_money_confirmation_score"), errors="coerce")
    coverage = pd.to_numeric(out.get("broker_flow_coverage_pct"), errors="coerce").clip(0, 100)
    weight = 0.20 * coverage.fillna(0) / 100.0
    blended = ((1 - weight) * base + weight * broker).where(base.notna() & broker.notna(), base).clip(0, 100)
    out["broker_confirmation_weight_pct"] = (100 * weight).round(1)
    out["broker_pre_confirmation_smart_money_score"] = base
    out["broker_post_confirmation_smart_money_score"] = blended.round(1)
    out["smart_money_score"] = blended.round(1)
    delta = (1.5 * ((broker.fillna(50) - 50) / 50).clip(-1, 1) * coverage.fillna(0) / 100).clip(-1.5, 1.5)
    if "emir_conviction_score" in out.columns:
        out["broker_emir_conviction_delta"] = delta.round(3)
        out["emir_conviction_score"] = (pd.to_numeric(out["broker_base_emir_conviction_score"], errors="coerce") + delta).clip(0, 100).round(3)
    if "emir_final_score" in out.columns:
        out["emir_final_score"] = (pd.to_numeric(out["broker_base_emir_final_score"], errors="coerce") + delta).clip(0, 100).round(3)
    out["broker_enrichment_compatibility_state"] = "CANONICAL_ENRICHED"
    combined = pd.concat([out, protected], ignore_index=True, sort=False)
    return combined.sort_values(order_column, kind="stable").drop(columns=[order_column]).reset_index(drop=True)


__all__ = ["PUBLIC_CACHE_URL", "load_public_cache", "score_broker_history", "classify_broker_enrichment_state", "enrich_emir_broker"]
