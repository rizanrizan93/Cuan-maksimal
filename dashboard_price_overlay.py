from __future__ import annotations

"""Render-only current factual price overlay for the Emir dashboard.

This module is deliberately cache-only. It reads already-persisted shared
STOCK_SUMMARY facts and never calls ZAPI or another market-data provider.
It does not mutate scanner technical truth, scoring, ranking, or authorization.
"""

from typing import Any

import numpy as np
import pandas as pd

from idx_trading_calendar import latest_expected_completed_session
from shared_stock_summary_evidence import PROVIDER, TABLE, SharedStockSummaryEvidence


def _number(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) and np.isfinite(parsed) else float("nan")


def _normal_ticker(value: Any) -> str:
    ticker = str(value or "").strip().upper()
    return ticker[:-3] if ticker.endswith(".JK") else ticker


def _technical_date(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).tz_localize(None).normalize()


def apply_current_market_price_overlay(
    frame: pd.DataFrame,
    *,
    now: Any = None,
    evidence: SharedStockSummaryEvidence | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach display-only current-price fields to a detached copy.

    Priority:
    1. CURRENT + VALID shared stock-summary row for the expected completed session.
    2. Technical last_price only when its own last_date is current.
    3. Otherwise display price is unavailable; stale technical price is never
       labelled as current.

    The original last_price and all decision fields are left untouched.
    """
    out = frame.copy(deep=True)
    expected = pd.Timestamp(latest_expected_completed_session(now)).tz_localize(None).normalize()
    expected_date = expected.date().isoformat()
    meta: dict[str, Any] = {
        "state": "NO_ROWS",
        "expected_trade_date": expected_date,
        "shared_rows": 0,
        "overlaid_rows": 0,
        "technical_stale_rows": 0,
        "provider_calls": 0,
        "request_avoided": True,
    }
    if out.empty:
        return out, meta

    shared_rows: list[dict[str, Any]] = []
    source = evidence or SharedStockSummaryEvidence("EMIR_DASHBOARD_CACHE_ONLY")
    if getattr(source, "ready", False) and getattr(source, "backend", None) is not None:
        try:
            # IMPORTANT: direct backend read only. Do not call source.get_day(),
            # because that path may refresh from ZAPI on a cache miss.
            shared_rows = [
                dict(row)
                for row in source.backend.read_rows(
                    TABLE,
                    {"provider": PROVIDER, "trade_date": expected_date},
                    limit=5000,
                )
            ]
            meta["state"] = "SHARED_CACHE_READ"
        except Exception as exc:
            meta["state"] = "SHARED_CACHE_READ_FAILED"
            meta["detail"] = type(exc).__name__
    else:
        meta["state"] = "SHARED_HUB_UNAVAILABLE"

    valid: dict[str, dict[str, Any]] = {}
    for row in shared_rows:
        ticker = _normal_ticker(row.get("ticker"))
        close = _number(row.get("close"))
        if (
            ticker
            and np.isfinite(close)
            and close > 0
            and str(row.get("validation_state") or "").upper() == "VALID"
            and str(row.get("freshness_state") or "").upper() == "CURRENT"
            and str(row.get("trade_date") or "")[:10] == expected_date
        ):
            valid[ticker] = row
    meta["shared_rows"] = len(valid)

    display_prices: list[float] = []
    display_changes: list[float] = []
    display_asofs: list[str] = []
    display_sources: list[str] = []
    display_states: list[str] = []
    technical_states: list[str] = []
    notes: list[str] = []

    for _, row in out.iterrows():
        ticker = _normal_ticker(row.get("ticker"))
        tech_date = _technical_date(row.get("last_date"))
        tech_current = bool(tech_date is not None and tech_date >= expected)
        factual = valid.get(ticker)

        if factual is not None:
            close = _number(factual.get("close"))
            previous = _number(factual.get("previous"))
            change = (
                100.0 * (close / previous - 1.0)
                if np.isfinite(previous) and previous > 0
                else float("nan")
            )
            display_prices.append(close)
            display_changes.append(change)
            display_asofs.append(expected_date)
            display_sources.append("SHARED_STOCK_SUMMARY_ZAPI_CACHE")
            display_states.append("CURRENT_FACTUAL")
            meta["overlaid_rows"] += 1
            if tech_current:
                technical_states.append("TECHNICAL_CURRENT")
                notes.append(f"Factual close {expected_date} · technical current")
            else:
                technical_states.append("TECHNICAL_STALE_RESCAN_REQUIRED")
                notes.append(f"Factual close {expected_date} · TECHNICAL STALE — RESCAN REQUIRED")
                meta["technical_stale_rows"] += 1
        elif tech_current:
            price = _number(row.get("last_price"))
            display_prices.append(price)
            display_changes.append(_number(row.get("dashboard_price_change_pct")))
            display_asofs.append(expected_date)
            display_sources.append("CURRENT_TECHNICAL_OHLCV")
            display_states.append("CURRENT_TECHNICAL")
            technical_states.append("TECHNICAL_CURRENT")
            notes.append(f"Technical close {expected_date}")
        else:
            display_prices.append(float("nan"))
            display_changes.append(float("nan"))
            display_asofs.append(expected_date)
            display_sources.append("UNAVAILABLE")
            display_states.append("CURRENT_PRICE_UNAVAILABLE")
            technical_states.append("TECHNICAL_STALE_RESCAN_REQUIRED")
            notes.append(f"Harga current unavailable · technical stale vs {expected_date}")
            meta["technical_stale_rows"] += 1

    out["display_last_price"] = display_prices
    out["display_price_change_pct"] = display_changes
    out["display_price_asof"] = display_asofs
    out["display_price_source"] = display_sources
    out["display_price_state"] = display_states
    out["display_technical_freshness_state"] = technical_states
    out["display_price_note"] = notes
    return out, meta


__all__ = ["apply_current_market_price_overlay"]
