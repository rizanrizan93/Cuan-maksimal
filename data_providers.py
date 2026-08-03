from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable
import time

import numpy as np
import pandas as pd
import requests
try:
    import yfinance as yf
except Exception:  # optional fallback; direct Yahoo chart remains available
    yf = None


USER_AGENT = "Mozilla/5.0 (compatible; IDX-Narrative-Flow-Scanner/1.0; research)"


@dataclass(frozen=True)
class FetchResult:
    ticker: str
    frame: pd.DataFrame
    provider: str
    status: str
    detail: str = ""


def normalize_ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.startswith("^") or text.endswith("=X"):
        return text
    return text if text.endswith(".JK") else f"{text}.JK"


def bare_ticker(value: Any) -> str:
    return normalize_ticker(value).removesuffix(".JK")


def parse_universe_csv(uploaded: Any) -> list[str]:
    frame = pd.read_csv(uploaded)
    if frame.empty:
        return []
    candidates = [column for column in frame.columns if str(column).strip().lower() in {"ticker", "symbol", "kode", "code"}]
    column = candidates[0] if candidates else frame.columns[0]
    values = [normalize_ticker(value) for value in frame[column].tolist()]
    return list(dict.fromkeys(value for value in values if value))


def _sanitize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    local = frame.copy()
    if isinstance(local.columns, pd.MultiIndex):
        local.columns = [column[0] if isinstance(column, tuple) else column for column in local.columns]
    rename = {str(column).lower(): column for column in local.columns}
    required = {}
    for name in ("Open", "High", "Low", "Close", "Volume"):
        source = rename.get(name.lower())
        if source is not None:
            required[source] = name
    local = local.rename(columns=required)
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(local.columns):
        return pd.DataFrame()
    if not isinstance(local.index, pd.DatetimeIndex):
        local.index = pd.to_datetime(local.index, errors="coerce")
    if local.index.tz is not None:
        local.index = local.index.tz_convert("Asia/Jakarta").tz_localize(None)
    local = local[~local.index.isna()].sort_index()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        local[column] = pd.to_numeric(local[column], errors="coerce")
    local = local.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
    local = local[(local[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    local["Volume"] = local["Volume"].fillna(0).clip(lower=0)
    return local[~local.index.duplicated(keep="last")]


def yahoo_chart_direct(ticker: str, period: str = "5y", timeout: int = 18) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{normalize_ticker(ticker)}"
    response = requests.get(
        url,
        params={"range": period, "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not result:
        return pd.DataFrame()
    timestamps = result.get("timestamp") or []
    quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])
    adj = (((result.get("indicators") or {}).get("adjclose") or [{}])[0]).get("adjclose") or []
    n = min(len(timestamps), *(len(quote.get(name.lower()) or []) for name in ("Open", "High", "Low", "Close", "Volume")))
    if n <= 0:
        return pd.DataFrame()
    index = pd.to_datetime(timestamps[:n], unit="s", utc=True).tz_convert("Asia/Jakarta").tz_localize(None).normalize()
    frame = pd.DataFrame(
        {
            "Open": quote.get("open")[:n],
            "High": quote.get("high")[:n],
            "Low": quote.get("low")[:n],
            "Close": quote.get("close")[:n],
            "Volume": quote.get("volume")[:n],
        },
        index=index,
    )
    if len(adj) >= n:
        raw_close = pd.to_numeric(frame["Close"], errors="coerce")
        adjusted = pd.to_numeric(pd.Series(adj[:n], index=index), errors="coerce")
        ratio = adjusted.div(raw_close.replace(0, np.nan)).where(lambda x: x.gt(0)).fillna(1.0)
        for column in ("Open", "High", "Low", "Close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * ratio
        frame["Close"] = adjusted.where(adjusted.gt(0), frame["Close"])
    return _sanitize_ohlcv(frame)


def fetch_ohlcv(ticker: str, period: str = "5y") -> FetchResult:
    symbol = normalize_ticker(ticker)
    try:
        frame = yahoo_chart_direct(symbol, period=period)
        if not frame.empty:
            return FetchResult(symbol, frame, "YAHOO_CHART_DIRECT", "OK")
    except Exception as exc:
        direct_error = f"{type(exc).__name__}: {exc}"
    else:
        direct_error = "NO_DATA"
    if yf is None:
        return FetchResult(symbol, pd.DataFrame(), "NONE", "ERROR", f"direct={direct_error}; yfinance unavailable")
    try:
        frame = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        frame = _sanitize_ohlcv(frame)
        if not frame.empty:
            return FetchResult(symbol, frame, "YFINANCE", "OK", direct_error)
        return FetchResult(symbol, pd.DataFrame(), "YFINANCE", "NO_DATA", direct_error)
    except Exception as exc:
        return FetchResult(symbol, pd.DataFrame(), "NONE", "ERROR", f"direct={direct_error}; fallback={type(exc).__name__}: {exc}")


def fetch_many_ohlcv(tickers: Iterable[str], period: str = "5y", max_workers: int = 8) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    frames: dict[str, pd.DataFrame] = {}
    audit: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 16))) as executor:
        futures = {executor.submit(fetch_ohlcv, ticker, period): ticker for ticker in symbols}
        for future in as_completed(futures):
            result = future.result()
            audit.append({
                "ticker": result.ticker,
                "provider": result.provider,
                "status": result.status,
                "bars": len(result.frame),
                "last_date": result.frame.index.max().date().isoformat() if not result.frame.empty else "",
                "detail": result.detail,
            })
            if not result.frame.empty:
                frames[result.ticker] = result.frame
    return frames, pd.DataFrame(audit).sort_values(["status", "ticker"]).reset_index(drop=True)


def _normalize_news_item(item: dict[str, Any], ticker: str) -> dict[str, Any] | None:
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = str(content.get("title") or "").strip()
    if not title:
        return None
    provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
    canonical = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
    click = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else {}
    published = content.get("pubDate") or content.get("providerPublishTime") or item.get("providerPublishTime")
    if isinstance(published, (int, float)):
        published_at = pd.to_datetime(published, unit="s", errors="coerce", utc=True)
    else:
        published_at = pd.to_datetime(published, errors="coerce", utc=True)
    return {
        "ticker": normalize_ticker(ticker),
        "published_at": published_at,
        "title": title,
        "summary": str(content.get("summary") or content.get("description") or "").strip(),
        "publisher": str(provider.get("displayName") or content.get("publisher") or item.get("publisher") or "").strip(),
        "url": str(canonical.get("url") or click.get("url") or content.get("link") or item.get("link") or "").strip(),
        "source_tier": "PUBLIC_NEWS",
    }


def fetch_yahoo_news(ticker: str, limit: int = 8) -> list[dict[str, Any]]:
    symbol = normalize_ticker(ticker)
    if yf is None:
        return []
    try:
        items = yf.Ticker(symbol).news or []
    except Exception:
        return []
    output: list[dict[str, Any]] = []
    for item in items[: max(0, int(limit))]:
        if isinstance(item, dict):
            normalized = _normalize_news_item(item, symbol)
            if normalized:
                output.append(normalized)
    return output


def fetch_many_news(tickers: Iterable[str], limit: int = 8, max_workers: int = 6) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 10))) as executor:
        futures = {executor.submit(fetch_yahoo_news, ticker, limit): ticker for ticker in symbols}
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=["ticker", "published_at", "title", "summary", "publisher", "url", "source_tier"])
    frame = pd.DataFrame(rows)
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    return frame.sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)
