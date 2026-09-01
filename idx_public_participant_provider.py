from __future__ import annotations

"""Emir-owned adapter for IDX public Trade Detail participant evidence."""

from datetime import date
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Iterator
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

from idx_trade_detail_discovery import (
    DiscoveryAttempt,
    discover_trade_detail_url as _discover_trade_detail_url,
    download_trade_detail as _download_trade_detail,
)


PUBLIC_INDEX_URL = (
    "https://www.idxdata3.co.id/INET_Specification/Market_Summary/Market_Indices/"
    "IX200720.TXT?directory=.%2FIDX+Reporting+PSPP%2FRevitalisasi%2FPUBLIK%2F"
)
SOURCE_NAME = "IDX_PUBLIC_TRADE_DETAIL_PUBLIK"
PROVENANCE = "VERIFIED_IDX_PUBLIC_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER"


def _canon(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; Emir-IDX-Participant-Collector/1.0)",
        "Accept": "text/csv,text/plain,*/*",
    }


def discover_trade_detail_url(
    trade_date: date,
    timeout: int = 20,
    diagnostics: list[DiscoveryAttempt] | None = None,
) -> str:
    return _discover_trade_detail_url(trade_date, timeout=timeout, diagnostics=diagnostics)


def download_trade_detail(
    trade_date: date,
    timeout: int = 45,
    diagnostics: list[DiscoveryAttempt] | None = None,
) -> tuple[Path, str]:
    return _download_trade_detail(trade_date, timeout=timeout, diagnostics=diagnostics)


def _read_trade_chunks(path: Path, universe: set[str] | None = None, chunksize: int = 250_000) -> Iterator[pd.DataFrame]:
    accepted = {"asset", "participant_buy", "participant_sell", "volume", "value", "tradingdate"}
    for separator in ("|", ","):
        try:
            iterator = pd.read_csv(
                path,
                sep=separator,
                usecols=lambda column: str(column).strip().lower() in accepted,
                chunksize=chunksize,
                low_memory=False,
                dtype=str,
            )
            yielded = False
            for chunk in iterator:
                chunk.columns = [str(column).strip().lower() for column in chunk.columns]
                renames = {
                    "seccode": "asset", "code": "asset", "ticker": "asset",
                    "brokersellid": "participant_sell", "brokerbuyid": "participant_buy",
                    "sellbrokerid": "participant_sell", "buybrokerid": "participant_buy",
                    "quantity": "volume", "tradedate": "tradingdate",
                }
                chunk = chunk.rename(columns={old: new for old, new in renames.items() if old in chunk.columns and new not in chunk.columns})
                required = {"asset", "participant_buy", "participant_sell", "volume", "value"}
                if not required.issubset(chunk.columns):
                    continue
                chunk["asset"] = chunk["asset"].map(_canon)
                if universe:
                    chunk = chunk[chunk["asset"].isin(universe)]
                if chunk.empty:
                    continue
                for column in ("participant_buy", "participant_sell"):
                    chunk[column] = chunk[column].astype(str).str.strip().str.upper()
                for column in ("volume", "value"):
                    chunk[column] = pd.to_numeric(chunk[column], errors="coerce").fillna(0.0)
                yielded = True
                yield chunk
            if yielded:
                return
        except Exception:
            continue
    raise RuntimeError("IDX_TRADE_DETAIL_PARSE_FAILED")


def aggregate_trade_detail(path: Path, trade_date: date, universe: Iterable[Any] | None = None) -> pd.DataFrame:
    names = {_canon(value) for value in (universe or []) if _canon(value)}
    buys: list[pd.DataFrame] = []
    sells: list[pd.DataFrame] = []
    for chunk in _read_trade_chunks(path, names or None):
        buys.append(
            chunk[chunk["participant_buy"].ne("")]
            .groupby(["asset", "participant_buy"], as_index=False)
            .agg(buy_value=("value", "sum"), buy_volume=("volume", "sum"))
            .rename(columns={"participant_buy": "broker_code"})
        )
        sells.append(
            chunk[chunk["participant_sell"].ne("")]
            .groupby(["asset", "participant_sell"], as_index=False)
            .agg(sell_value=("value", "sum"), sell_volume=("volume", "sum"))
            .rename(columns={"participant_sell": "broker_code"})
        )
    if not buys and not sells:
        return pd.DataFrame()
    buy = pd.concat(buys, ignore_index=True).groupby(["asset", "broker_code"], as_index=False).sum() if buys else pd.DataFrame(columns=["asset", "broker_code"])
    sell = pd.concat(sells, ignore_index=True).groupby(["asset", "broker_code"], as_index=False).sum() if sells else pd.DataFrame(columns=["asset", "broker_code"])
    out = buy.merge(sell, on=["asset", "broker_code"], how="outer").fillna(0.0)
    out = out.rename(columns={"asset": "ticker"})
    out["trade_date"] = pd.Timestamp(trade_date)
    out["buy_avg"] = out["buy_value"].div(out["buy_volume"].replace(0.0, np.nan))
    out["sell_avg"] = out["sell_value"].div(out["sell_volume"].replace(0.0, np.nan))
    out["net_value"] = out["buy_value"] - out["sell_value"]
    out["net_volume"] = out["buy_volume"] - out["sell_volume"]
    out["gross_value"] = out["buy_value"] + out["sell_value"]
    out["source"] = SOURCE_NAME
    out["source_verified"] = True
    out["provenance_state"] = PROVENANCE
    return out


def trim_daily_top_flow(frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    positive = frame[frame["net_value"].gt(0)].copy()
    negative = frame[frame["net_value"].lt(0)].copy()
    positive["side"], negative["side"] = "TOP_NET_BUYER", "TOP_NET_SELLER"
    positive["net_rank"] = positive.groupby(["trade_date", "ticker"])["net_value"].rank(method="first", ascending=False)
    negative["net_rank"] = negative.groupby(["trade_date", "ticker"])["net_value"].rank(method="first", ascending=True)
    return pd.concat(
        [positive[positive["net_rank"].le(top_n)], negative[negative["net_rank"].le(top_n)]],
        ignore_index=True,
    ).sort_values(["trade_date", "ticker", "side", "net_rank"], kind="stable").reset_index(drop=True)


__all__ = ["aggregate_trade_detail", "download_trade_detail", "trim_daily_top_flow"]
