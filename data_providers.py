from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any, Iterable
import contextlib
from urllib.parse import quote_plus
import random
import time
import threading
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
try:
    import yfinance as yf
except Exception:  # optional fallback
    yf = None


USER_AGENT = "Mozilla/5.0 (compatible; IDX-Emir-Autonomous-Scanner/1.5.2; research)"
SESSION_CLOSE_HOUR = 16
SESSION_CLOSE_MINUTE = 20
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
_MIN_REQUEST_INTERVAL_SECONDS = 0.12

KSEI_PROFILE_URL = "https://web.ksei.co.id/services/registered-securities/shares/lc/{ticker}?setLocale=en-US"


def parse_ksei_price_history_html(html: str) -> pd.DataFrame:
    """Parse KSEI public price history. KSEI volume is displayed in lots, so convert to shares."""
    soup = BeautifulSoup(html or "", "html.parser")
    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        table_rows = table.find_all("tr")
        if not table_rows:
            continue
        headers = [cell.get_text(" ", strip=True).lower() for cell in table_rows[0].find_all(["th", "td"])]
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(headers)):
            continue
        for row in table_rows[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            if len(cells) < len(headers):
                continue
            values = dict(zip(headers, cells))
            date = pd.to_datetime(values.get("date"), errors="coerce", dayfirst=True)
            if pd.isna(date):
                continue
            def number(key: str) -> float:
                text = str(values.get(key, "")).replace(",", "").strip()
                try:
                    return float(text)
                except (TypeError, ValueError):
                    return np.nan
            volume_lots = number("volume")
            rows.append({
                "Date": date,
                "Open": number("open"),
                "High": number("high"),
                "Low": number("low"),
                "Close": number("close"),
                "Volume": volume_lots * 100.0 if np.isfinite(volume_lots) else np.nan,
            })
        if rows:
            break
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("Date")
    return _sanitize_ohlcv(frame)


def ksei_price_history(ticker: str, timeout: int = 18, retries: int = 2) -> pd.DataFrame:
    symbol = normalize_ticker(ticker)
    if symbol.startswith("^") or symbol.endswith("=X"):
        return pd.DataFrame()
    url = KSEI_PROFILE_URL.format(ticker=bare_ticker(symbol))
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            _pace_request()
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"},
                timeout=timeout,
            )
            response.raise_for_status()
            return parse_ksei_price_history_html(response.text)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= retries or not _retryable_status(exc):
                raise
            time.sleep(min(4.0, 0.8 * (2 ** attempt) + random.uniform(0.1, 0.5)))
    if last_error:
        raise last_error
    return pd.DataFrame()


def _pace_request() -> None:
    """Avoid synchronized request bursts that trigger free-provider throttling."""
    global _LAST_REQUEST_AT
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _MIN_REQUEST_INTERVAL_SECONDS - (now - _LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()


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


def parse_universe_frame(uploaded: Any) -> pd.DataFrame:
    frame = pd.read_csv(uploaded)
    metadata_columns = ["ticker", "company_name", "sector", "theme", "macro_theme", "secular_trend", "catalyst", "universe_note"]
    if frame.empty:
        return pd.DataFrame(columns=metadata_columns)
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    candidates = [column for column in local.columns if column in {"ticker", "symbol", "kode", "code"}]
    ticker_column = candidates[0] if candidates else local.columns[0]
    local["ticker"] = local[ticker_column].map(normalize_ticker)
    aliases = {
        "company": "company_name", "issuer": "company_name", "nama": "company_name",
        "industry": "sector", "sektor": "sector",
        "narrative_theme": "theme", "story": "theme",
        "macro": "macro_theme", "top_down_theme": "macro_theme",
        "secular": "secular_trend", "secular_theme": "secular_trend",
        "trigger_catalyst": "catalyst",
        "note": "universe_note", "notes": "universe_note",
    }
    for source, target in aliases.items():
        if source in local.columns and target not in local.columns:
            local[target] = local[source]
    for column in ("company_name", "sector", "theme", "macro_theme", "secular_trend", "catalyst", "universe_note"):
        if column not in local.columns:
            local[column] = ""
        local[column] = local[column].fillna("").astype(str).str.strip()
    local = local[local["ticker"].ne("")].drop_duplicates("ticker", keep="first")
    return local[metadata_columns].reset_index(drop=True)


def parse_universe_csv(uploaded: Any) -> list[str]:
    return parse_universe_frame(uploaded)["ticker"].tolist()


def _coalesce_duplicate_columns(local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return local
    output: dict[str, pd.Series] = {}
    for column in local.columns:
        key = str(column).strip()
        series = local[column]
        if isinstance(series, pd.DataFrame):
            series = series.bfill(axis=1).iloc[:, 0]
        if key not in output:
            output[key] = series
        else:
            output[key] = output[key].where(output[key].notna(), series)
    return pd.DataFrame(output, index=local.index)


def _sanitize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    local = frame.copy()
    if isinstance(local.columns, pd.MultiIndex):
        ohlcv_names = {"open", "high", "low", "close", "adj close", "volume"}
        selected_level = None
        for level in range(local.columns.nlevels):
            values = {str(item).strip().lower() for item in local.columns.get_level_values(level)}
            if len(values & ohlcv_names) >= 4:
                selected_level = level
                break
        if selected_level is not None:
            local.columns = [str(column[selected_level]).strip() for column in local.columns]
        else:
            local.columns = ["_".join(str(part) for part in column if str(part)) for column in local.columns]
    local = _coalesce_duplicate_columns(local)
    canonical: dict[str, str] = {}
    for column in local.columns:
        normalized = str(column).strip().lower().replace("_", " ")
        if normalized in {"open", "high", "low", "close", "volume"}:
            canonical[column] = normalized.title()
        elif normalized == "adj close" and "Close" not in canonical.values():
            canonical[column] = "Close"
    local = local.rename(columns=canonical)
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(local.columns):
        return pd.DataFrame()
    local = local[["Open", "High", "Low", "Close", "Volume"]]
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


def completed_session_frame(frame: pd.DataFrame, *, now: Any = None, completed_only: bool = True) -> pd.DataFrame:
    local = _sanitize_ohlcv(frame)
    if local.empty or not completed_only:
        return local
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    if current.tzinfo is None:
        current = current.tz_localize("Asia/Jakarta")
    else:
        current = current.tz_convert("Asia/Jakarta")
    today = current.tz_localize(None).normalize()
    session_complete = current.weekday() < 5 and (current.hour, current.minute) >= (SESSION_CLOSE_HOUR, SESSION_CLOSE_MINUTE)
    if not session_complete and len(local) and local.index.max().normalize() >= today:
        local = local[local.index.normalize() < today]
    return local


def _retryable_status(exc: Exception) -> bool:
    text = str(exc)
    return any(code in text for code in ("429", "500", "502", "503", "504", "timed out", "Timeout"))


def yahoo_chart_direct(ticker: str, period: str = "5y", timeout: int = 18, retries: int = 3) -> pd.DataFrame:
    symbol = normalize_ticker(ticker)
    hosts = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            _pace_request()
            host = hosts[attempt % len(hosts)]
            url = f"https://{host}/v8/finance/chart/{symbol}"
            response = requests.get(
                url,
                params={"range": period, "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"},
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=timeout,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(15.0, max(0.0, float(retry_after))))
                    except (TypeError, ValueError):
                        pass
            response.raise_for_status()
            payload = response.json()
            result = ((payload.get("chart") or {}).get("result") or [None])[0]
            if not result:
                return pd.DataFrame()
            timestamps = result.get("timestamp") or []
            quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])
            lengths = [len(quote.get(name) or []) for name in ("open", "high", "low", "close", "volume")]
            n = min([len(timestamps), *lengths]) if lengths else 0
            if n <= 0:
                return pd.DataFrame()
            index = pd.to_datetime(timestamps[:n], unit="s", utc=True).tz_convert("Asia/Jakarta").tz_localize(None).normalize()
            frame = pd.DataFrame({
                "Open": (quote.get("open") or [])[:n],
                "High": (quote.get("high") or [])[:n],
                "Low": (quote.get("low") or [])[:n],
                "Close": (quote.get("close") or [])[:n],
                "Volume": (quote.get("volume") or [])[:n],
            }, index=index)
            adj = (((result.get("indicators") or {}).get("adjclose") or [{}])[0]).get("adjclose") or []
            if len(adj) >= n:
                raw_close = pd.to_numeric(frame["Close"], errors="coerce")
                adjusted = pd.to_numeric(pd.Series(adj[:n], index=index), errors="coerce")
                ratio = adjusted.div(raw_close.replace(0, np.nan)).where(lambda value: value.gt(0)).fillna(1.0)
                for column in ("Open", "High", "Low", "Close"):
                    frame[column] = pd.to_numeric(frame[column], errors="coerce") * ratio
                frame["Close"] = adjusted.where(adjusted.gt(0), frame["Close"])
            return _sanitize_ohlcv(frame)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= retries or not _retryable_status(exc):
                raise
            time.sleep(min(6.0, (1.2 * (2 ** attempt)) + random.uniform(0.1, 0.7)))
    if last_error:
        raise last_error
    return pd.DataFrame()



def yahoo_chart_window(
    ticker: str,
    *,
    start: Any,
    end: Any = None,
    timeout: int = 18,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch a bounded daily window for incremental cache refresh."""
    symbol = normalize_ticker(ticker)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    hosts = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            _pace_request()
            host = hosts[attempt % len(hosts)]
            response = requests.get(
                f"https://{host}/v8/finance/chart/{symbol}",
                params={
                    "period1": int(start_ts.timestamp()),
                    "period2": int(end_ts.timestamp()),
                    "interval": "1d",
                    "events": "div,splits",
                    "includeAdjustedClose": "true",
                },
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=timeout,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(15.0, max(0.0, float(retry_after))))
                    except (TypeError, ValueError):
                        pass
            response.raise_for_status()
            payload = response.json()
            result = ((payload.get("chart") or {}).get("result") or [None])[0]
            if not result:
                return pd.DataFrame()
            timestamps = result.get("timestamp") or []
            quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])
            lengths = [len(quote.get(name) or []) for name in ("open", "high", "low", "close", "volume")]
            n = min([len(timestamps), *lengths]) if lengths else 0
            if n <= 0:
                return pd.DataFrame()
            index = pd.to_datetime(timestamps[:n], unit="s", utc=True).tz_convert("Asia/Jakarta").tz_localize(None).normalize()
            frame = pd.DataFrame({
                "Open": (quote.get("open") or [])[:n],
                "High": (quote.get("high") or [])[:n],
                "Low": (quote.get("low") or [])[:n],
                "Close": (quote.get("close") or [])[:n],
                "Volume": (quote.get("volume") or [])[:n],
            }, index=index)
            adj = (((result.get("indicators") or {}).get("adjclose") or [{}])[0]).get("adjclose") or []
            if len(adj) >= n:
                raw_close = pd.to_numeric(frame["Close"], errors="coerce")
                adjusted = pd.to_numeric(pd.Series(adj[:n], index=index), errors="coerce")
                ratio = adjusted.div(raw_close.replace(0, np.nan)).where(lambda value: value.gt(0)).fillna(1.0)
                for column in ("Open", "High", "Low", "Close"):
                    frame[column] = pd.to_numeric(frame[column], errors="coerce") * ratio
                frame["Close"] = adjusted.where(adjusted.gt(0), frame["Close"])
            return _sanitize_ohlcv(frame)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= retries or not _retryable_status(exc):
                raise
            time.sleep(min(6.0, (1.2 * (2 ** attempt)) + random.uniform(0.1, 0.7)))
    if last_error:
        raise last_error
    return pd.DataFrame()


def fetch_ohlcv_window(
    ticker: str,
    *,
    start: Any,
    end: Any = None,
    completed_only: bool = True,
    now: Any = None,
    direct_retries: int = 2,
) -> FetchResult:
    """Incremental OHLCV fetch. It returns only the requested tail window."""
    symbol = normalize_ticker(ticker)
    try:
        frame = completed_session_frame(
            yahoo_chart_window(symbol, start=start, end=end, retries=direct_retries),
            now=now,
            completed_only=completed_only,
        )
        if not frame.empty:
            return FetchResult(symbol, frame, "YAHOO_CHART_INCREMENTAL", "OK")
    except Exception as exc:
        direct_error = f"{type(exc).__name__}: {exc}"
    else:
        direct_error = "NO_DATA"
    yf_error = "yfinance unavailable"
    if yf is not None:
        try:
            captured = StringIO()
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
            if start_ts.tzinfo is not None:
                start_ts = start_ts.tz_convert("UTC").tz_localize(None)
            if end_ts.tzinfo is not None:
                end_ts = end_ts.tz_convert("UTC").tz_localize(None)
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                downloaded = yf.download(
                    symbol,
                    start=start_ts.date().isoformat(),
                    end=end_ts.date().isoformat(),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by="column",
                )
            frame = completed_session_frame(downloaded, now=now, completed_only=completed_only)
            if not frame.empty:
                return FetchResult(symbol, frame, "YFINANCE_INCREMENTAL", "OK", direct_error)
            message = captured.getvalue().strip().replace("\n", " | ")
            yf_error = f"NO_DATA; {message}" if message else "NO_DATA"
        except Exception as exc:
            yf_error = f"{type(exc).__name__}: {exc}"
    return FetchResult(symbol, pd.DataFrame(), "NONE", "ERROR", f"direct={direct_error}; yfinance={yf_error}")

def fetch_ohlcv(ticker: str, period: str = "5y", *, completed_only: bool = True, now: Any = None, direct_retries: int = 3) -> FetchResult:
    symbol = normalize_ticker(ticker)
    try:
        frame = completed_session_frame(yahoo_chart_direct(symbol, period=period, retries=direct_retries), now=now, completed_only=completed_only)
        if not frame.empty:
            return FetchResult(symbol, frame, "YAHOO_CHART_DIRECT", "OK")
    except Exception as exc:
        direct_error = f"{type(exc).__name__}: {exc}"
    else:
        direct_error = "NO_DATA"
    yf_error = "yfinance unavailable"
    if yf is not None:
        try:
            captured = StringIO()
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                downloaded = yf.download(
                    symbol, period=period, interval="1d", auto_adjust=True,
                    progress=False, threads=False, group_by="column",
                )
            frame = completed_session_frame(downloaded, now=now, completed_only=completed_only)
            if not frame.empty:
                return FetchResult(symbol, frame, "YFINANCE", "OK", direct_error)
            message = captured.getvalue().strip().replace("\n", " | ")
            yf_error = f"NO_DATA; {message}" if message else "NO_DATA"
        except Exception as exc:
            yf_error = f"{type(exc).__name__}: {exc}"
    try:
        frame = completed_session_frame(ksei_price_history(symbol), now=now, completed_only=completed_only)
        if not frame.empty:
            return FetchResult(symbol, frame, "KSEI_PRICE_HISTORY", "OK", f"direct={direct_error}; yfinance={yf_error}")
        ksei_error = "NO_DATA"
    except Exception as exc:
        ksei_error = f"{type(exc).__name__}: {exc}"
    return FetchResult(symbol, pd.DataFrame(), "NONE", "ERROR", f"direct={direct_error}; yfinance={yf_error}; ksei={ksei_error}")


def fetch_many_ohlcv(
    tickers: Iterable[str], period: str = "5y", max_workers: int = 4, *, completed_only: bool = True, now: Any = None,
    serial_recovery_limit: int = 40,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    frames: dict[str, pd.DataFrame] = {}
    audit: list[dict[str, Any]] = []
    worker_count = max(1, min(int(max_workers), 4))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(fetch_ohlcv, ticker, period, completed_only=completed_only, now=now): ticker for ticker in symbols}
        for future in as_completed(futures):
            result = future.result()
            audit.append({
                "ticker": result.ticker,
                "provider": result.provider,
                "status": result.status,
                "bars": len(result.frame),
                "last_date": result.frame.index.max().date().isoformat() if not result.frame.empty else "",
                "detail": result.detail,
                "recovery_pass": False,
            })
            if not result.frame.empty:
                frames[result.ticker] = result.frame
    failed_symbols = [row["ticker"] for row in audit if row.get("status") != "OK"]
    recovery_limit = max(0, min(int(serial_recovery_limit), len(failed_symbols)))
    if recovery_limit:
        for symbol in failed_symbols[:recovery_limit]:
            time.sleep(0.18 + random.uniform(0.0, 0.12))
            result = fetch_ohlcv(symbol, period, completed_only=completed_only, now=now, direct_retries=1)
            if not result.frame.empty:
                frames[result.ticker] = result.frame
            for row in audit:
                if row.get("ticker") == symbol:
                    prior_detail = str(row.get("detail") or "")
                    row.update({
                        "provider": result.provider,
                        "status": result.status,
                        "bars": len(result.frame),
                        "last_date": result.frame.index.max().date().isoformat() if not result.frame.empty else "",
                        "detail": f"SERIAL_RECOVERY_AFTER={prior_detail}; {result.detail}",
                        "recovery_pass": True,
                    })
                    break

    audit_frame = pd.DataFrame(audit)
    if audit_frame.empty:
        audit_frame = pd.DataFrame(columns=["ticker", "provider", "status", "bars", "last_date", "detail"])
    else:
        today = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Jakarta")
        if today.tzinfo is None:
            today = today.tz_localize("Asia/Jakarta")
        else:
            today = today.tz_convert("Asia/Jakarta")
        parsed = pd.to_datetime(audit_frame["last_date"], errors="coerce")
        audit_frame["data_age_days"] = (today.tz_localize(None).normalize() - parsed).dt.days
        audit_frame["completed_session_state"] = np.where(
            audit_frame["status"].eq("OK"),
            np.where(audit_frame["data_age_days"].fillna(999).le(4), "CURRENT_COMPLETED_SESSION", "STALE_COMPLETED_SESSION"),
            "NO_COMPLETED_SESSION",
        )
        audit_frame["quality_state"] = np.where(
            audit_frame["status"].ne("OK"), "PROVIDER_FAILED",
            np.where(pd.to_numeric(audit_frame["bars"], errors="coerce").fillna(0).lt(220), "INSUFFICIENT_HISTORY",
                     np.where(audit_frame["data_age_days"].fillna(999).gt(7), "STALE_DATA", "VALID")),
        )
    return frames, audit_frame.sort_values(["status", "ticker"]).reset_index(drop=True)



def assess_benchmark_freshness(benchmark: pd.DataFrame, universe_frames: dict[str, pd.DataFrame], min_universe_count: int = 20) -> dict[str, Any]:
    """Compare benchmark completion against the modal completed date of the universe.

    Calendar age alone is insufficient around weekends/holidays. A benchmark that is older
    than the dominant universe session is stale relative to the scan and must not drive
    market-regime or relative-strength calculations as fully current evidence.
    """
    universe_dates: list[pd.Timestamp] = []
    for frame in (universe_frames or {}).values():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            value = pd.to_datetime(frame.index.max(), errors="coerce")
            if pd.notna(value):
                universe_dates.append(pd.Timestamp(value).tz_localize(None).normalize())
    benchmark_last = pd.NaT
    if isinstance(benchmark, pd.DataFrame) and not benchmark.empty:
        value = pd.to_datetime(benchmark.index.max(), errors="coerce")
        if pd.notna(value):
            benchmark_last = pd.Timestamp(value).tz_localize(None).normalize()
    if len(universe_dates) < max(1, int(min_universe_count)):
        return {
            "benchmark_freshness_state": "UNIVERSE_REFERENCE_INSUFFICIENT",
            "benchmark_last_date": benchmark_last.date().isoformat() if pd.notna(benchmark_last) else "",
            "universe_reference_date": "",
            "benchmark_business_lag_days": None,
            "benchmark_usable": False,
            "universe_reference_count": len(universe_dates),
        }
    counts = pd.Series(universe_dates).value_counts()
    reference = pd.Timestamp(counts.index[0]).normalize()
    if pd.isna(benchmark_last):
        state, usable, lag = "BENCHMARK_MISSING", False, None
    else:
        lag = int(np.busday_count(benchmark_last.date(), reference.date())) if benchmark_last < reference else 0
        usable = bool(benchmark_last >= reference)
        state = "CURRENT_RELATIVE_TO_UNIVERSE" if usable else "STALE_RELATIVE_TO_UNIVERSE"
    return {
        "benchmark_freshness_state": state,
        "benchmark_last_date": benchmark_last.date().isoformat() if pd.notna(benchmark_last) else "",
        "universe_reference_date": reference.date().isoformat(),
        "benchmark_business_lag_days": lag,
        "benchmark_usable": usable,
        "universe_reference_count": len(universe_dates),
    }

def _normalize_news_item(item: dict[str, Any], ticker: str, source: str = "YAHOO_NEWS") -> dict[str, Any] | None:
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
        "collection_provider": source,
        "source_verified": False,
    }


def fetch_yahoo_news(ticker: str, limit: int = 8) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symbol = normalize_ticker(ticker)
    if yf is None:
        return [], {"ticker": symbol, "provider": "YAHOO_NEWS", "status": "UNAVAILABLE", "items": 0, "detail": "yfinance unavailable"}
    try:
        items = yf.Ticker(symbol).news or []
        output = [normalized for item in items[: max(1, limit)] if (normalized := _normalize_news_item(item, symbol)) is not None]
        return output, {"ticker": symbol, "provider": "YAHOO_NEWS", "status": "OK" if output else "NO_ITEMS", "items": len(output), "detail": ""}
    except Exception as exc:
        return [], {"ticker": symbol, "provider": "YAHOO_NEWS", "status": "ERROR", "items": 0, "detail": f"{type(exc).__name__}: {exc}"}


def fetch_google_news_rss(ticker: str, company_name: str = "", limit: int = 8, timeout: int = 15) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symbol = normalize_ticker(ticker)
    base = bare_ticker(symbol)
    query = f'"{company_name}" saham' if company_name else f'"{base}" saham IDX'
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=id&gl=ID&ceid=ID:id"
    try:
        _pace_request()
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        output: list[dict[str, Any]] = []
        for item in root.findall(".//item")[: max(1, limit)]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            published = pd.to_datetime(item.findtext("pubDate"), errors="coerce", utc=True)
            source_node = item.find("source")
            publisher = (source_node.text or "").strip() if source_node is not None else "Google News"
            if title:
                output.append({
                    "ticker": symbol,
                    "published_at": published,
                    "title": title,
                    "summary": "",
                    "publisher": publisher,
                    "url": link,
                    "source_tier": "PUBLIC_NEWS",
                    "collection_provider": "GOOGLE_NEWS_RSS",
                    "source_verified": False,
                })
        return output, {"ticker": symbol, "provider": "GOOGLE_NEWS_RSS", "status": "OK" if output else "NO_ITEMS", "items": len(output), "detail": ""}
    except Exception as exc:
        return [], {"ticker": symbol, "provider": "GOOGLE_NEWS_RSS", "status": "ERROR", "items": 0, "detail": f"{type(exc).__name__}: {exc}"}


def fetch_many_news(
    universe: pd.DataFrame | Iterable[str], limit: int = 8, max_workers: int = 4, *, use_yahoo: bool = True, use_google: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(universe, pd.DataFrame):
        metadata = universe.copy()
        metadata["ticker"] = metadata["ticker"].map(normalize_ticker)
        company_map = metadata.set_index("ticker").get("company_name", pd.Series(dtype=str)).to_dict()
        symbols = metadata["ticker"].tolist()
    else:
        symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in universe if normalize_ticker(ticker)))
        company_map = {}
    events: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    def fetch_symbol(symbol: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        local_events: list[dict[str, Any]] = []
        local_audits: list[dict[str, Any]] = []
        if use_yahoo:
            items, audit = fetch_yahoo_news(symbol, limit=limit)
            local_events.extend(items)
            local_audits.append(audit)
        if use_google:
            items, audit = fetch_google_news_rss(symbol, str(company_map.get(symbol) or ""), limit=limit)
            local_events.extend(items)
            local_audits.append(audit)
        return local_events, local_audits

    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 4))) as executor:
        futures = {executor.submit(fetch_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            local_events, local_audits = future.result()
            events.extend(local_events)
            audits.extend(local_audits)
    event_frame = pd.DataFrame(events)
    if not event_frame.empty:
        event_frame["published_at"] = pd.to_datetime(event_frame["published_at"], errors="coerce", utc=True)
        event_frame = event_frame.sort_values("published_at", ascending=False, na_position="last")
        event_frame = event_frame.drop_duplicates(["ticker", "title", "url"], keep="first").reset_index(drop=True)
    audit_frame = pd.DataFrame(audits)
    if audit_frame.empty:
        audit_frame = pd.DataFrame(columns=["ticker", "provider", "status", "items", "detail"])
    return event_frame, audit_frame.sort_values(["status", "ticker", "provider"]).reset_index(drop=True)


__all__ = [
    "FetchResult", "bare_ticker", "completed_session_frame", "fetch_many_news", "fetch_many_ohlcv", "fetch_ohlcv_window", "yahoo_chart_window",
    "ksei_price_history", "normalize_ticker", "parse_ksei_price_history_html", "parse_universe_csv", "parse_universe_frame", "_sanitize_ohlcv",
]
