from __future__ import annotations

"""Read-only EMIR Block IDX database consumer.

No function in this module imports or calls an internet market-data provider.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import os

import pandas as pd

from persistence import DatabaseConfig, _request


@dataclass(frozen=True)
class DatabaseOnlySnapshot:
    state: str
    rank_date: str | None
    ranked_rows: int
    execution_ready_rows: int
    top3_rows: int
    detail: str = ""


def database_only_enabled() -> bool:
    return str(os.getenv("CAK_SCAN_DATABASE_ONLY", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _get(config: DatabaseConfig, table: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not config.ready:
        raise RuntimeError("EMIR_DATABASE_NOT_READY")
    response = _request(config, "GET", table, params=dict(params), timeout=10)
    payload = response.json()
    return payload if isinstance(payload, list) else []


def load_latest_database_ranking(
    config: DatabaseConfig,
    *,
    limit: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, DatabaseOnlySnapshot]:
    """Load one immutable EOD rank snapshot and its Top-3 in two small reads."""
    if not database_only_enabled():
        return pd.DataFrame(), pd.DataFrame(), DatabaseOnlySnapshot(
            "DATABASE_ONLY_DISABLED", None, 0, 0, 0
        )
    try:
        latest = _get(
            config,
            "cak_idx_rank_daily",
            {
                "select": "rank_date",
                "order": "rank_date.desc",
                "limit": "1",
            },
        )
        if not latest:
            return pd.DataFrame(), pd.DataFrame(), DatabaseOnlySnapshot(
                "SOURCE_NOT_READY", None, 0, 0, 0, "No official EOD ranking has been materialized."
            )
        rank_date = str(latest[0].get("rank_date") or "")
        ranks = _get(
            config,
            "cak_idx_rank_daily",
            {
                "select": "*",
                "rank_date": f"eq.{rank_date}",
                "order": "overall_rank.asc",
                "limit": str(max(3, min(int(limit), 200))),
            },
        )
        top3 = _get(
            config,
            "cak_idx_top3_execution",
            {
                "select": "*",
                "rank_date": f"eq.{rank_date}",
                "order": "execution_rank.asc",
                "limit": "3",
            },
        )
        rank_frame = pd.DataFrame(ranks)
        top_frame = pd.DataFrame(top3)
        eligible = int(rank_frame.get("execution_eligible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        state = "READY" if len(top_frame) == 3 else "INSUFFICIENT_EXECUTION_READY"
        return rank_frame, top_frame, DatabaseOnlySnapshot(
            state, rank_date, len(rank_frame), eligible, len(top_frame)
        )
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), DatabaseOnlySnapshot(
            "DATABASE_UNAVAILABLE", None, 0, 0, 0, f"{type(exc).__name__}: {exc}"
        )


def load_market_panel(
    config: DatabaseConfig,
    tickers: Iterable[str],
    *,
    sessions: int = 130,
) -> pd.DataFrame:
    """Read a bounded official market panel. Missing DB data never triggers a web fallback."""
    names = [
        str(t).strip().upper().replace(".JK", "")
        for t in dict.fromkeys(tickers)
        if str(t).strip()
    ]
    if not names or not database_only_enabled():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(names), 50):
        chunk = names[start:start + 50]
        encoded = ",".join(f'"{name.replace(chr(34), "")}"' for name in chunk)
        rows.extend(_get(
            config,
            "cak_idx_market_daily",
            {
                "select": "ticker,trade_date,open,high,low,close,volume,traded_value,frequency,foreign_buy,foreign_sell,foreign_net,bid,offer",
                "ticker": f"in.({encoded})",
                "order": "ticker.asc,trade_date.desc",
                "limit": str(min(10000, max(1000, len(chunk) * min(max(int(sessions), 20), 160)))),
            },
        ))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    return frame.dropna(subset=["ticker", "trade_date"]).sort_values(["ticker", "trade_date"]).reset_index(drop=True)


__all__ = [
    "DatabaseOnlySnapshot", "database_only_enabled",
    "load_latest_database_ranking", "load_market_panel",
]
