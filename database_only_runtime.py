from __future__ import annotations

"""Read-only EMIR runtime backed exclusively by the compact Block IDX database."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from persistence import DatabaseConfig, _headers


DATABASE_ONLY_RUNTIME_VERSION = "EMIR_BLOCK_IDX_DATABASE_ONLY_RUNTIME_V2"


@dataclass(frozen=True)
class DatabaseOnlySnapshot:
    health: dict[str, Any]
    ranking: pd.DataFrame
    top3: pd.DataFrame
    state: str


def _json_response(response: requests.Response, label: str) -> Any:
    if response.status_code >= 400:
        raise RuntimeError(f"{label} HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if not isinstance(payload, (dict, list)):
        raise RuntimeError(f"{label} returned a non-JSON-object payload")
    return payload


def _rpc(config: DatabaseConfig, function_name: str, payload: dict[str, Any]) -> Any:
    if not config.ready:
        raise RuntimeError("EMIR database-only runtime is not configured")
    response = requests.post(
        f"{config.url}/rest/v1/rpc/{function_name}",
        headers=_headers(config),
        json=payload,
        timeout=35,
    )
    return _json_response(response, function_name)


def load_database_only_snapshot(
    config: DatabaseConfig,
    *,
    rank_limit: int = 1000,
) -> DatabaseOnlySnapshot:
    """Load a frozen server-side decision snapshot without any provider fallback."""
    health = _rpc(config, "cak_idx_database_health_v2", {})
    ranking_payload = _rpc(
        config,
        "cak_load_latest_idx_rank_v2",
        {"p_limit": min(1000, max(1, int(rank_limit)))},
    )
    top3_payload = _rpc(config, "cak_load_latest_idx_top3_v3", {})
    ranking = pd.DataFrame(ranking_payload if isinstance(ranking_payload, list) else [])
    top3 = pd.DataFrame(top3_payload if isinstance(top3_payload, list) else [])
    if not top3.empty and "ownership" in top3.columns:
        ownership = top3["ownership"].apply(lambda value: value if isinstance(value, dict) else {})
        for source, target in (
            ("controller_pct", "controller_pct"),
            ("public_pct", "public_pct"),
            ("treasury_pct", "treasury_pct"),
            ("holder_count", "holder_count"),
        ):
            top3[target] = ownership.apply(lambda value, key=source: value.get(key))
    if not top3.empty and "recent_events" in top3.columns:
        top3["recent_event_count"] = top3["recent_events"].apply(
            lambda value: len(value) if isinstance(value, list) else 0
        )

    latest_market = str(health.get("latest_market_date") or "")
    latest_rank = str(health.get("latest_rank_date") or "")
    source_ready = bool(latest_market and latest_rank and latest_market == latest_rank)
    top3_ready = len(top3.index) == 3
    state = "DATABASE_ONLY_READY" if source_ready and top3_ready else "DATABASE_SOURCE_NOT_READY"
    return DatabaseOnlySnapshot(health=health, ranking=ranking, top3=top3, state=state)


def load_database_market_panel(
    config: DatabaseConfig,
    tickers: list[str] | tuple[str, ...],
    *,
    sessions: int = 130,
) -> dict[str, pd.DataFrame]:
    """Load normalized OHLCV frames for charts; never calls Yahoo/KSEI/other providers."""
    clean = sorted({str(value).strip().upper().removesuffix(".JK") for value in tickers if str(value).strip()})
    if not clean:
        return {}
    payload = _rpc(
        config,
        "cak_load_idx_market_panel_v1",
        {"p_tickers": clean, "p_sessions": min(160, max(20, int(sessions)))},
    )
    rows = pd.DataFrame(payload if isinstance(payload, list) else [])
    if rows.empty:
        return {}
    rows["trade_date"] = pd.to_datetime(rows["trade_date"], errors="coerce", utc=True)
    for column in ("open", "high", "low", "close", "volume", "traded_value", "frequency", "foreign_net"):
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    frames: dict[str, pd.DataFrame] = {}
    for ticker, local in rows.groupby("ticker", sort=True):
        frame = local.sort_values("trade_date").set_index("trade_date")
        frame = frame.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        frames[str(ticker)] = frame
    return frames


__all__ = [
    "DATABASE_ONLY_RUNTIME_VERSION",
    "DatabaseOnlySnapshot",
    "load_database_market_panel",
    "load_database_only_snapshot",
]
