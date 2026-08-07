from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping
import json

import numpy as np
import pandas as pd

from persistence import DatabaseConfig, _json_value, _post_payload_in_chunks, _request, database_status
from data_providers import (
    FetchResult,
    completed_session_frame,
    fetch_many_ohlcv,
    fetch_ohlcv_window,
    normalize_ticker,
    _sanitize_ohlcv,
)
from autonomous_enrichment import fetch_many_fundamentals, fetch_many_ksei_profiles
from data_providers import fetch_many_news

CACHE_VERSION = "1.6.4"
OHLCV_CACHE_TTL_HOURS = 12.0
KSEI_CACHE_TTL_HOURS = 24.0
FUNDAMENTAL_CACHE_TTL_HOURS = 24.0 * 7
NEWS_CACHE_TTL_HOURS = 2.0
STALE_OHLCV_FALLBACK_DAYS = 10
STALE_SOURCE_FALLBACK_DAYS = 30


@dataclass(frozen=True)
class CacheRead:
    payload: Any
    checked_at: pd.Timestamp | pd.NaT
    valid_until: pd.Timestamp | pd.NaT
    provider: str
    status: str
    content_sha256: str


def _utc_timestamp(value: Any = None) -> pd.Timestamp:
    ts = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _canonical(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_canonical(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _canonical(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return ts.isoformat() if pd.notna(ts) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value if isinstance(value, (str, int, dict, list)) or value is None else str(value)


def _hash_payload(value: Any) -> str:
    """Streaming canonical hash tolerant of JSONB int/float numeric round-trips."""
    digest = sha256()

    def feed(item: Any) -> None:
        if isinstance(item, Mapping):
            digest.update(b"{")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                feed(str(key))
                feed(item[key])
            digest.update(b"}")
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                feed(child)
            digest.update(b"]")
            return
        if isinstance(item, (bool, np.bool_)):
            digest.update(b"T" if bool(item) else b"F")
            return
        if isinstance(item, (int, float, np.integer, np.floating)):
            number = float(item)
            digest.update(b"N")
            digest.update((format(number, ".15g") if np.isfinite(number) else "NULL").encode("ascii"))
            digest.update(b";")
            return
        if isinstance(item, (pd.Timestamp, np.datetime64)):
            timestamp = pd.Timestamp(item)
            feed(timestamp.isoformat() if pd.notna(timestamp) else None)
            return
        if item is None:
            digest.update(b"Z")
            return
        try:
            if pd.isna(item):
                digest.update(b"Z")
                return
        except Exception:
            pass
        text = item if isinstance(item, str) else str(item)
        encoded = text.encode("utf-8")
        digest.update(b"S")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)

    feed(value)
    return digest.hexdigest()


def frame_to_payload(frame: pd.DataFrame) -> list[list[Any]]:
    local = _sanitize_ohlcv(frame)
    rows: list[list[Any]] = []
    for index, row in local.iterrows():
        rows.append([
            pd.Timestamp(index).date().isoformat(),
            _json_value(row.get("Open")),
            _json_value(row.get("High")),
            _json_value(row.get("Low")),
            _json_value(row.get("Close")),
            _json_value(row.get("Volume")),
        ])
    return rows


def payload_to_frame(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, list) or len(item) < 6:
            continue
        rows.append({"Date": item[0], "Open": item[1], "High": item[2], "Low": item[3], "Close": item[4], "Volume": item[5]})
    if not rows:
        return pd.DataFrame()
    local = pd.DataFrame(rows)
    local["Date"] = pd.to_datetime(local["Date"], errors="coerce")
    return _sanitize_ohlcv(local.dropna(subset=["Date"]).set_index("Date"))


def _period_cutoff(period: str, now: Any = None) -> pd.Timestamp:
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    if current.tzinfo is not None:
        current = current.tz_convert("Asia/Jakarta").tz_localize(None)
    years = 5 if str(period).lower() == "5y" else 3
    return current.normalize() - pd.DateOffset(years=years, days=10)


def trim_period(frame: pd.DataFrame, period: str, now: Any = None) -> pd.DataFrame:
    local = _sanitize_ohlcv(frame)
    return local[local.index >= _period_cutoff(period, now)] if not local.empty else local


def build_ohlcv_cache_row(
    ticker: str,
    frame: pd.DataFrame,
    *,
    period: str,
    provider: str,
    quality_state: str = "VALID",
    checked_at: Any = None,
    last_scan_id: str = "",
) -> dict[str, Any]:
    symbol = normalize_ticker(ticker)
    local = _sanitize_ohlcv(frame)
    checked = _utc_timestamp(checked_at)
    payload = frame_to_payload(local)
    return {
        "ticker": symbol,
        "period": str(period),
        "first_session_date": local.index.min().date().isoformat() if not local.empty else None,
        "last_session_date": local.index.max().date().isoformat() if not local.empty else None,
        "bars": int(len(local)),
        "provider": str(provider or ""),
        "quality_state": str(quality_state or ""),
        "checked_at": checked.isoformat(),
        "last_scan_id": str(last_scan_id or ""),
        "content_sha256": _hash_payload(payload),
        "payload": payload,
    }


def build_source_cache_row(
    ticker: str,
    family: str,
    payload: Any,
    *,
    provider: str,
    status: str,
    checked_at: Any = None,
    ttl_hours: float,
    latest_observed_at: Any = None,
    last_scan_id: str = "",
) -> dict[str, Any]:
    symbol = normalize_ticker(ticker)
    checked = _utc_timestamp(checked_at)
    normalized_payload = _canonical(payload)
    return {
        "cache_key": f"{str(family).upper()}:{symbol}",
        "ticker": symbol,
        "family": str(family).upper(),
        "provider": str(provider or ""),
        "status": str(status or ""),
        "checked_at": checked.isoformat(),
        "valid_until": (checked + pd.Timedelta(hours=float(ttl_hours))).isoformat(),
        "latest_observed_at": _utc_timestamp(latest_observed_at).isoformat() if latest_observed_at is not None and pd.notna(latest_observed_at) else None,
        "last_scan_id": str(last_scan_id or ""),
        "content_sha256": _hash_payload(normalized_payload),
        "payload": normalized_payload,
    }


def _get_in_chunks(config: DatabaseConfig, table: str, key: str, values: Iterable[str], select: str = "*") -> list[dict[str, Any]]:
    unique = list(dict.fromkeys(str(value) for value in values if str(value)))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(unique), 20):
        chunk = unique[start:start + 20]
        if not chunk:
            continue
        response = _request(config, "GET", table, params={"select": select, key: "in.(" + ",".join(f'"{value.replace(chr(34), "")}"' for value in chunk) + ")"}, timeout=8)
        payload = response.json()
        if isinstance(payload, list):
            rows.extend(payload)
    return rows


def read_ohlcv_cache(config: DatabaseConfig, tickers: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Best-effort cache read. Any database error becomes a cache miss, not a scan failure."""
    if not config.ready:
        return {}
    symbols = [normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)]
    try:
        rows = _get_in_chunks(config, "cak_ohlcv_cache", "ticker", symbols)
    except Exception:
        return {}
    return {str(row.get("ticker") or ""): row for row in rows if row.get("ticker")}


def read_source_cache(config: DatabaseConfig, tickers: Iterable[str], family: str) -> dict[str, dict[str, Any]]:
    """Best-effort source-cache read. Missing/unreadable rows are fetched from providers."""
    if not config.ready:
        return {}
    symbols = [normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)]
    keys = [f"{str(family).upper()}:{symbol}" for symbol in symbols]
    try:
        rows = _get_in_chunks(config, "cak_source_cache", "cache_key", keys)
    except Exception:
        return {}
    return {str(row.get("ticker") or ""): row for row in rows if row.get("ticker")}



def _row_hash_valid(row: Mapping[str, Any]) -> bool:
    expected = str(row.get("content_sha256") or "")
    return bool(expected and expected == _hash_payload(row.get("payload")))

def _cache_age_hours(row: Mapping[str, Any], now: Any = None) -> float:
    checked = pd.to_datetime(row.get("checked_at"), errors="coerce", utc=True)
    if pd.isna(checked):
        return float("inf")
    return max(0.0, (_utc_timestamp(now) - checked).total_seconds() / 3600.0)


def _expected_completed_weekday(now: Any = None) -> pd.Timestamp:
    current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    if current.tzinfo is None:
        current = current.tz_localize("Asia/Jakarta")
    else:
        current = current.tz_convert("Asia/Jakarta")
    date = current.tz_localize(None).normalize()
    after_close = (current.hour, current.minute) >= (16, 20)
    if current.weekday() < 5 and after_close:
        target = date
    else:
        target = date - pd.Timedelta(days=1)
    while target.weekday() >= 5:
        target -= pd.Timedelta(days=1)
    return target


def _ohlcv_cache_fresh(row: Mapping[str, Any], frame: pd.DataFrame, *, ttl_hours: float, now: Any = None) -> bool:
    if frame.empty:
        return False
    if _cache_age_hours(row, now) <= float(ttl_hours):
        return True
    expected = _expected_completed_weekday(now)
    return bool(frame.index.max().normalize() >= expected)


def _audit_row(ticker: str, provider: str, status: str, frame: pd.DataFrame | None = None, detail: str = "", **extra: Any) -> dict[str, Any]:
    local = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    return {
        "ticker": normalize_ticker(ticker),
        "provider": provider,
        "status": status,
        "bars": int(len(local)),
        "last_date": local.index.max().date().isoformat() if not local.empty else "",
        "detail": detail,
        "recovery_pass": False,
        "data_age_days": (pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize() - local.index.max().normalize()).days if not local.empty else np.nan,
        "completed_session_state": "CURRENT_COMPLETED_SESSION" if status in {"CACHE_HIT", "INCREMENTAL_REFRESH", "COLD_REFRESH", "STALE_CACHE_FALLBACK"} and not local.empty else "PROVIDER_FAILED",
        "quality_state": "VALID" if status in {"CACHE_HIT", "INCREMENTAL_REFRESH", "COLD_REFRESH"} else "STALE_CACHE" if status == "STALE_CACHE_FALLBACK" else "PROVIDER_FAILED",
        "cache_state": status,
        **extra,
    }


def fetch_ohlcv_cache_first(
    config: DatabaseConfig,
    tickers: Iterable[str],
    *,
    period: str = "5y",
    max_workers: int = 4,
    completed_only: bool = True,
    now: Any = None,
    force_refresh: bool = False,
    cache_ttl_hours: float = OHLCV_CACHE_TTL_HOURS,
    last_scan_id: str = "",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[dict[str, Any]]]:
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    cached_rows = read_ohlcv_cache(config, symbols)
    frames: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    cold: list[str] = []
    stale: list[tuple[str, pd.DataFrame, Mapping[str, Any]]] = []

    for symbol in symbols:
        row = cached_rows.get(symbol)
        if row and not _row_hash_valid(row):
            row = None
        cached = payload_to_frame(row.get("payload")) if row else pd.DataFrame()
        cached = trim_period(cached, period, now)
        cache_period = str(row.get("period") or "") if row else ""
        period_sufficient = cache_period == "5y" or cache_period == period or (period == "3y" and not cached.empty)
        if row and period_sufficient and not force_refresh and _ohlcv_cache_fresh(row, cached, ttl_hours=cache_ttl_hours, now=now):
            frames[symbol] = completed_session_frame(cached, now=now, completed_only=completed_only)
            audits.append(_audit_row(symbol, "SUPABASE_OHLCV_CACHE", "CACHE_HIT", frames[symbol], f"age_hours={_cache_age_hours(row, now):.2f}"))
        elif row and period_sufficient and not cached.empty:
            stale.append((symbol, cached, row))
        else:
            cold.append(symbol)

    if cold:
        fetched, fetched_audit = fetch_many_ohlcv(cold, period=period, max_workers=max_workers, completed_only=completed_only, now=now)
        fetched_map = {str(row.get("ticker")): row for row in fetched_audit.to_dict(orient="records")} if not fetched_audit.empty else {}
        for symbol in cold:
            frame = fetched.get(symbol, pd.DataFrame())
            source = fetched_map.get(symbol, {})
            if not frame.empty:
                frames[symbol] = frame
                audits.append(_audit_row(symbol, str(source.get("provider") or "LIVE_PROVIDER"), "COLD_REFRESH", frame, str(source.get("detail") or "")))
                writes.append(build_ohlcv_cache_row(symbol, frame, period=period, provider=str(source.get("provider") or "LIVE_PROVIDER"), checked_at=now, last_scan_id=last_scan_id))
            else:
                audits.append(_audit_row(symbol, str(source.get("provider") or "NONE"), "CACHE_MISS_PROVIDER_FAILED", frame, str(source.get("detail") or "")))

    def refresh_one(item: tuple[str, pd.DataFrame, Mapping[str, Any]]) -> tuple[str, FetchResult, pd.DataFrame, Mapping[str, Any]]:
        symbol, cached, row = item
        start = cached.index.max() - pd.Timedelta(days=14)
        result = fetch_ohlcv_window(symbol, start=start, end=_utc_timestamp(now) + pd.Timedelta(days=2), completed_only=completed_only, now=now)
        return symbol, result, cached, row

    if stale:
        with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 4))) as executor:
            futures = {executor.submit(refresh_one, item): item[0] for item in stale}
            for future in as_completed(futures):
                symbol, result, cached, row = future.result()
                if not result.frame.empty:
                    merged = _sanitize_ohlcv(pd.concat([cached, result.frame], axis=0))
                    merged = trim_period(merged, period, now)
                    frames[symbol] = merged
                    audits.append(_audit_row(symbol, result.provider, "INCREMENTAL_REFRESH", merged, f"tail_bars={len(result.frame)}; {result.detail}"))
                    writes.append(build_ohlcv_cache_row(symbol, merged, period="5y" if str(row.get("period")) == "5y" or period == "5y" else period, provider=result.provider, checked_at=now, last_scan_id=last_scan_id))
                else:
                    age_days = max(0, int((_utc_timestamp(now).tz_convert("Asia/Jakarta").tz_localize(None).normalize() - cached.index.max().normalize()).days))
                    if age_days <= STALE_OHLCV_FALLBACK_DAYS:
                        frames[symbol] = completed_session_frame(cached, now=now, completed_only=completed_only)
                        audits.append(_audit_row(symbol, "SUPABASE_OHLCV_CACHE", "STALE_CACHE_FALLBACK", frames[symbol], f"age_days={age_days}; refresh={result.detail}"))
                    else:
                        audits.append(_audit_row(symbol, result.provider, "CACHE_STALE_PROVIDER_FAILED", pd.DataFrame(), result.detail))

    audit = pd.DataFrame(audits)
    if not audit.empty:
        audit = audit.sort_values(["status", "ticker"]).reset_index(drop=True)
    return frames, audit, writes


def _source_row_fresh(row: Mapping[str, Any], now: Any = None) -> bool:
    valid_until = pd.to_datetime(row.get("valid_until"), errors="coerce", utc=True)
    return bool(pd.notna(valid_until) and valid_until >= _utc_timestamp(now))


def _payload_records(row: Mapping[str, Any], key: str | None = None) -> list[dict[str, Any]]:
    payload = row.get("payload")
    if key and isinstance(payload, dict):
        payload = payload.get(key)
    return payload if isinstance(payload, list) else []


def fetch_ksei_cache_first(
    config: DatabaseConfig,
    tickers: Iterable[str],
    *,
    max_workers: int = 4,
    now: Any = None,
    force_refresh: bool = False,
    last_scan_id: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    cached = read_source_cache(config, symbols, "KSEI")
    profiles: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    refresh: list[str] = []
    for symbol in symbols:
        row = cached.get(symbol)
        if row and not _row_hash_valid(row):
            cached.pop(symbol, None)
            row = None
        if row and not force_refresh and _source_row_fresh(row, now):
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            profiles.extend(payload.get("profiles") or [])
            actions.extend(payload.get("actions") or [])
            audits.append({"ticker": symbol, "provider": "SUPABASE_KSEI_CACHE", "status": "CACHE_HIT", "items": len(payload.get("profiles") or []) + len(payload.get("actions") or []), "detail": f"age_hours={_cache_age_hours(row, now):.2f}", "cache_state": "CACHE_HIT"})
        else:
            refresh.append(symbol)
    writes: list[dict[str, Any]] = []
    if refresh:
        fresh_profiles, fresh_actions, fresh_audit = fetch_many_ksei_profiles(refresh, max_workers=max_workers)
        pmap = {ticker: group.to_dict(orient="records") for ticker, group in fresh_profiles.groupby("ticker")} if not fresh_profiles.empty else {}
        amap = {ticker: group.to_dict(orient="records") for ticker, group in fresh_actions.groupby("ticker")} if not fresh_actions.empty else {}
        audit_map = {str(row.get("ticker")): row for row in fresh_audit.to_dict(orient="records")} if not fresh_audit.empty else {}
        for symbol in refresh:
            pitems, aitems = pmap.get(symbol, []), amap.get(symbol, [])
            audit = audit_map.get(symbol, {})
            if pitems or aitems:
                profiles.extend(pitems); actions.extend(aitems)
                audits.append({**audit, "ticker": symbol, "status": "COLD_REFRESH" if symbol not in cached else "REFRESHED", "cache_state": "COLD_REFRESH" if symbol not in cached else "REFRESHED"})
                writes.append(build_source_cache_row(symbol, "KSEI", {"profiles": pitems, "actions": aitems}, provider=str(audit.get("provider") or "KSEI_SECURITY_PROFILE"), status="OK", checked_at=now, ttl_hours=KSEI_CACHE_TTL_HOURS, last_scan_id=last_scan_id))
            elif symbol in cached:
                row = cached[symbol]
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                age_days = _cache_age_hours(row, now) / 24.0
                if age_days <= STALE_SOURCE_FALLBACK_DAYS:
                    profiles.extend(payload.get("profiles") or []); actions.extend(payload.get("actions") or [])
                    audits.append({**audit, "ticker": symbol, "provider": "SUPABASE_KSEI_CACHE", "status": "STALE_CACHE_FALLBACK", "items": len(payload.get("profiles") or []) + len(payload.get("actions") or []), "detail": f"age_days={age_days:.1f}; refresh={audit.get('detail','')}", "cache_state": "STALE_CACHE_FALLBACK"})
                else:
                    audits.append({**audit, "ticker": symbol, "status": "CACHE_STALE_PROVIDER_FAILED", "cache_state": "CACHE_STALE_PROVIDER_FAILED"})
            else:
                audits.append({**audit, "ticker": symbol, "status": str(audit.get("status") or "ERROR"), "cache_state": "CACHE_MISS_PROVIDER_FAILED"})
    return pd.DataFrame(profiles), pd.DataFrame(actions), pd.DataFrame(audits), writes


def fetch_fundamental_cache_first(
    config: DatabaseConfig,
    tickers: Iterable[str],
    *,
    max_workers: int = 3,
    now: Any = None,
    force_refresh: bool = False,
    last_scan_id: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    cached = read_source_cache(config, symbols, "FUNDAMENTAL")
    snapshots: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    refresh: list[str] = []
    for symbol in symbols:
        row = cached.get(symbol)
        if row and not _row_hash_valid(row):
            cached.pop(symbol, None)
            row = None
        if row and not force_refresh and _source_row_fresh(row, now):
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if payload:
                snapshots.append(payload)
            audits.append({"ticker": symbol, "provider": "SUPABASE_FUNDAMENTAL_CACHE", "status": "CACHE_HIT", "items": int(bool(payload)), "detail": f"age_hours={_cache_age_hours(row, now):.2f}", "cache_state": "CACHE_HIT"})
        else:
            refresh.append(symbol)
    writes: list[dict[str, Any]] = []
    if refresh:
        fresh, fresh_audit = fetch_many_fundamentals(refresh, max_workers=max_workers)
        fmap = fresh.set_index("ticker").to_dict(orient="index") if not fresh.empty else {}
        amap = {str(row.get("ticker")): row for row in fresh_audit.to_dict(orient="records")} if not fresh_audit.empty else {}
        for symbol in refresh:
            payload, audit = fmap.get(symbol, {}), amap.get(symbol, {})
            if payload and str(payload.get("fundamental_provenance_state")) != "PROVIDER_FAILED":
                snapshots.append({"ticker": symbol, **payload})
                audits.append({**audit, "ticker": symbol, "status": "COLD_REFRESH" if symbol not in cached else "REFRESHED", "cache_state": "COLD_REFRESH" if symbol not in cached else "REFRESHED"})
                writes.append(build_source_cache_row(symbol, "FUNDAMENTAL", {"ticker": symbol, **payload}, provider=str(audit.get("provider") or "YFINANCE_FUNDAMENTALS"), status=str(audit.get("status") or "OK"), checked_at=now, ttl_hours=FUNDAMENTAL_CACHE_TTL_HOURS, last_scan_id=last_scan_id))
            elif symbol in cached:
                row = cached[symbol]; old = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                if _cache_age_hours(row, now) / 24.0 <= STALE_SOURCE_FALLBACK_DAYS and old:
                    snapshots.append(old)
                    audits.append({**audit, "ticker": symbol, "provider": "SUPABASE_FUNDAMENTAL_CACHE", "status": "STALE_CACHE_FALLBACK", "items": 1, "detail": f"refresh={audit.get('detail','')}", "cache_state": "STALE_CACHE_FALLBACK"})
                else:
                    audits.append({**audit, "ticker": symbol, "status": "CACHE_STALE_PROVIDER_FAILED", "cache_state": "CACHE_STALE_PROVIDER_FAILED"})
            else:
                audits.append({**audit, "ticker": symbol, "status": str(audit.get("status") or "ERROR"), "cache_state": "CACHE_MISS_PROVIDER_FAILED"})
    return pd.DataFrame(snapshots), pd.DataFrame(audits), writes


def fetch_news_cache_first(
    config: DatabaseConfig,
    universe: pd.DataFrame,
    *,
    limit: int = 8,
    max_workers: int = 4,
    use_yahoo: bool = True,
    use_google: bool = True,
    now: Any = None,
    force_refresh: bool = False,
    last_scan_id: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    local_universe = universe.copy()
    local_universe["ticker"] = local_universe["ticker"].map(normalize_ticker)
    symbols = local_universe["ticker"].drop_duplicates().tolist()
    cached = read_source_cache(config, symbols, "NEWS")
    events: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    refresh: list[str] = []
    for symbol in symbols:
        row = cached.get(symbol)
        if row and not _row_hash_valid(row):
            cached.pop(symbol, None)
            row = None
        if row and not force_refresh and _source_row_fresh(row, now):
            payload = row.get("payload") if isinstance(row.get("payload"), list) else []
            events.extend(payload)
            audits.append({"ticker": symbol, "provider": "SUPABASE_NEWS_CACHE", "status": "CACHE_HIT", "items": len(payload), "detail": f"age_hours={_cache_age_hours(row, now):.2f}", "cache_state": "CACHE_HIT"})
        else:
            refresh.append(symbol)
    writes: list[dict[str, Any]] = []
    if refresh:
        fresh_universe = local_universe[local_universe["ticker"].isin(refresh)][["ticker", "company_name"]]
        fresh_events, fresh_audit = fetch_many_news(fresh_universe, limit=limit, max_workers=max_workers, use_yahoo=use_yahoo, use_google=use_google)
        emap = {ticker: group.to_dict(orient="records") for ticker, group in fresh_events.groupby("ticker")} if not fresh_events.empty else {}
        audit_groups = {ticker: group.to_dict(orient="records") for ticker, group in fresh_audit.groupby("ticker")} if not fresh_audit.empty else {}
        for symbol in refresh:
            new_items = emap.get(symbol, [])
            old_items = cached.get(symbol, {}).get("payload") if isinstance(cached.get(symbol, {}).get("payload"), list) else []
            merged = pd.DataFrame([*new_items, *old_items])
            if not merged.empty:
                merged["ticker"] = symbol
                merged["published_at"] = pd.to_datetime(merged.get("published_at"), errors="coerce", utc=True)
                merged = merged.sort_values("published_at", ascending=False, na_position="last").drop_duplicates(["ticker", "title", "url"], keep="first").head(max(20, limit * 4))
                items = [_canonical(record) for record in merged.to_dict(orient="records")]
            else:
                items = []
            provider_audits = audit_groups.get(symbol, [])
            provider_ok = any(str(row.get("status")) in {"OK", "NO_ITEMS"} for row in provider_audits)
            if provider_ok or items:
                events.extend(items)
                audits.extend([{**row, "cache_state": "COLD_REFRESH" if symbol not in cached else "REFRESHED"} for row in provider_audits] or [{"ticker": symbol, "provider": "NEWS", "status": "NO_ITEMS", "items": 0, "detail": "", "cache_state": "COLD_REFRESH"}])
                latest = pd.to_datetime(pd.Series([item.get("published_at") for item in items]), errors="coerce", utc=True).max() if items else None
                writes.append(build_source_cache_row(symbol, "NEWS", items, provider="YAHOO_AND_GOOGLE_PUBLIC_NEWS", status="OK" if items else "NO_ITEMS", checked_at=now, ttl_hours=NEWS_CACHE_TTL_HOURS, latest_observed_at=latest, last_scan_id=last_scan_id))
            elif old_items and _cache_age_hours(cached[symbol], now) / 24.0 <= STALE_SOURCE_FALLBACK_DAYS:
                events.extend(old_items)
                audits.append({"ticker": symbol, "provider": "SUPABASE_NEWS_CACHE", "status": "STALE_CACHE_FALLBACK", "items": len(old_items), "detail": "Live news providers failed.", "cache_state": "STALE_CACHE_FALLBACK"})
            else:
                audits.extend([{**row, "cache_state": "CACHE_MISS_PROVIDER_FAILED"} for row in provider_audits])
    event_frame = pd.DataFrame(events)
    if not event_frame.empty:
        event_frame["published_at"] = pd.to_datetime(event_frame["published_at"], errors="coerce", utc=True)
        event_frame = event_frame.sort_values("published_at", ascending=False, na_position="last").drop_duplicates(["ticker", "title", "url"], keep="first").reset_index(drop=True)
    return event_frame, pd.DataFrame(audits), writes



def load_cached_ohlcv_frames(
    config: DatabaseConfig,
    tickers: Iterable[str],
    *,
    period: str = "5y",
    now: Any = None,
    completed_only: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Read-only cache load used by resumable finalisation. It never calls an external provider."""
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    rows = read_ohlcv_cache(config, symbols)
    frames: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for symbol in symbols:
        row = rows.get(symbol)
        if not row:
            audits.append(_audit_row(symbol, "SUPABASE_OHLCV_CACHE", "CACHE_MISS_NO_PROVIDER_CALL", pd.DataFrame(), "Resumable read-only load."))
            continue
        if not _row_hash_valid(row):
            audits.append(_audit_row(symbol, "SUPABASE_OHLCV_CACHE", "CACHE_HASH_INVALID", pd.DataFrame(), "SHA-256 mismatch."))
            continue
        frame = completed_session_frame(trim_period(payload_to_frame(row.get("payload")), period, now), now=now, completed_only=completed_only)
        if frame.empty:
            audits.append(_audit_row(symbol, "SUPABASE_OHLCV_CACHE", "CACHE_EMPTY", frame, "No valid cached bars."))
            continue
        frames[symbol] = frame
        audits.append(_audit_row(symbol, "SUPABASE_OHLCV_CACHE", "CACHE_LOAD", frame, f"age_hours={_cache_age_hours(row, now):.2f}"))
    return frames, pd.DataFrame(audits)


def load_cached_ksei(
    config: DatabaseConfig, tickers: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read KSEI profile/action payloads without triggering network refresh."""
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    rows = read_source_cache(config, symbols, "KSEI")
    profiles: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for symbol in symbols:
        row = rows.get(symbol)
        if not row or not _row_hash_valid(row):
            audits.append({"ticker": symbol, "provider": "SUPABASE_KSEI_CACHE", "status": "CACHE_MISS_OR_HASH_INVALID", "items": 0, "detail": "Read-only resumable load.", "cache_state": "CACHE_MISS"})
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        pitems = payload.get("profiles") or []
        aitems = payload.get("actions") or []
        profiles.extend(pitems); actions.extend(aitems)
        audits.append({"ticker": symbol, "provider": "SUPABASE_KSEI_CACHE", "status": "CACHE_LOAD", "items": len(pitems) + len(aitems), "detail": "Read-only resumable load.", "cache_state": "CACHE_LOAD"})
    return pd.DataFrame(profiles), pd.DataFrame(actions), pd.DataFrame(audits)


def load_cached_fundamentals(
    config: DatabaseConfig, tickers: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    rows = read_source_cache(config, symbols, "FUNDAMENTAL")
    snapshots: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for symbol in symbols:
        row = rows.get(symbol)
        if not row or not _row_hash_valid(row):
            audits.append({"ticker": symbol, "provider": "SUPABASE_FUNDAMENTAL_CACHE", "status": "CACHE_MISS_OR_HASH_INVALID", "items": 0, "detail": "Read-only resumable load.", "cache_state": "CACHE_MISS"})
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if payload:
            snapshots.append({"ticker": symbol, **payload})
        audits.append({"ticker": symbol, "provider": "SUPABASE_FUNDAMENTAL_CACHE", "status": "CACHE_LOAD", "items": int(bool(payload)), "detail": "Read-only resumable load.", "cache_state": "CACHE_LOAD"})
    return pd.DataFrame(snapshots), pd.DataFrame(audits)


def load_cached_news(
    config: DatabaseConfig, tickers: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)))
    rows = read_source_cache(config, symbols, "NEWS")
    events: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for symbol in symbols:
        row = rows.get(symbol)
        if not row or not _row_hash_valid(row):
            audits.append({"ticker": symbol, "provider": "SUPABASE_NEWS_CACHE", "status": "CACHE_MISS_OR_HASH_INVALID", "items": 0, "detail": "Read-only resumable load.", "cache_state": "CACHE_MISS"})
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), list) else []
        events.extend(payload)
        audits.append({"ticker": symbol, "provider": "SUPABASE_NEWS_CACHE", "status": "CACHE_LOAD", "items": len(payload), "detail": "Read-only resumable load.", "cache_state": "CACHE_LOAD"})
    frame = pd.DataFrame(events)
    if not frame.empty and "published_at" in frame.columns:
        frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
        dedupe = [column for column in ("ticker", "title", "url") if column in frame.columns]
        if dedupe:
            frame = frame.sort_values("published_at", ascending=False, na_position="last").drop_duplicates(dedupe, keep="first").reset_index(drop=True)
    return frame, pd.DataFrame(audits)

def persist_verify_cache_bundle(
    config: DatabaseConfig,
    *,
    scan_id: str,
    ohlcv_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    chunk_size: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = database_status(config)
    if not config.ready:
        write = pd.DataFrame([{
            **base, "scan_id": scan_id, "table": "__SUMMARY__",
            "rows_attempted": len(ohlcv_rows) + len(source_rows), "rows_written": 0,
            "state": "CACHE_DATABASE_DISABLED",
            "detail": "Cache database unavailable; scanner continues with live providers and in-memory results.",
        }])
        verify = pd.DataFrame([{
            **base, "scan_id": scan_id, "table": "__SUMMARY__",
            "rows_expected": len(ohlcv_rows) + len(source_rows), "rows_verified": 0,
            "state": "CACHE_PERSISTENCE_SKIPPED",
            "detail": "No cache readback attempted because database is unavailable.",
        }])
        return write, verify
    write_reports: list[dict[str, Any]] = []
    specs = [
        ("cak_ohlcv_cache", "ticker", ohlcv_rows),
        ("cak_source_cache", "cache_key", source_rows),
    ]
    for table, conflict, rows in specs:
        try:
            written = _post_payload_in_chunks(config, table=table, conflict=conflict, payload=rows, chunk_size=chunk_size, return_rows=False)
            write_reports.append({**base, "scan_id": scan_id, "table": table, "rows_attempted": len(rows), "rows_written": written, "state": "CACHE_WRITTEN" if rows else "CACHE_EMPTY_EXPECTED", "detail": ""})
        except Exception as exc:
            write_reports.append({**base, "scan_id": scan_id, "table": table, "rows_attempted": len(rows), "rows_written": 0, "state": "CACHE_WRITE_FAILED", "detail": str(exc)})
    write_exact = all(row["rows_attempted"] == row["rows_written"] and row["state"] in {"CACHE_WRITTEN", "CACHE_EMPTY_EXPECTED"} for row in write_reports)
    write_reports.insert(0, {**base, "scan_id": scan_id, "table": "__SUMMARY__", "rows_attempted": sum(r["rows_attempted"] for r in write_reports), "rows_written": sum(r["rows_written"] for r in write_reports), "state": "CACHE_WRITE_ALL" if write_exact else "CACHE_WRITE_PARTIAL", "detail": "Persistent source cache write."})

    verify_reports: list[dict[str, Any]] = []
    if write_exact:
        for table, key, rows in specs:
            expected = {str(row[key]): str(row.get("content_sha256") or "") for row in rows}
            try:
                observed_rows = _get_in_chunks(config, table, key, expected.keys(), select=f"{key},content_sha256") if expected else []
                observed = {str(row.get(key)): str(row.get("content_sha256") or "") for row in observed_rows}
                exact = expected == observed
                verify_reports.append({**base, "scan_id": scan_id, "table": table, "rows_expected": len(expected), "rows_verified": len(observed), "state": "CACHE_VERIFIED_EXACT" if exact else "CACHE_HASH_MISMATCH", "detail": "Key and SHA-256 readback matched." if exact else "Cache readback key/hash mismatch."})
            except Exception as exc:
                verify_reports.append({**base, "scan_id": scan_id, "table": table, "rows_expected": len(expected), "rows_verified": 0, "state": "CACHE_READBACK_FAILED", "detail": str(exc)})
    else:
        verify_reports.append({**base, "scan_id": scan_id, "table": "__SUMMARY__", "rows_expected": 0, "rows_verified": 0, "state": "CACHE_READBACK_SKIPPED", "detail": "Write incomplete."})
    verify_exact = write_exact and all(row["state"] == "CACHE_VERIFIED_EXACT" for row in verify_reports)
    verify_reports.insert(0, {**base, "scan_id": scan_id, "table": "__SUMMARY__", "rows_expected": sum(r.get("rows_expected", 0) for r in verify_reports), "rows_verified": sum(r.get("rows_verified", 0) for r in verify_reports), "state": "CACHE_DATABASE_COMMITTED" if verify_exact else "CACHE_DATABASE_NOT_COMMITTED", "detail": "Persistent cache exact key/hash verification."})
    return pd.DataFrame(write_reports), pd.DataFrame(verify_reports)


def cache_commit_succeeded(verification: pd.DataFrame | None) -> bool:
    return bool(isinstance(verification, pd.DataFrame) and not verification.empty and str(verification.iloc[0].get("state")) == "CACHE_DATABASE_COMMITTED")


def cache_persistence_state(verification: pd.DataFrame | None) -> str:
    if not isinstance(verification, pd.DataFrame) or verification.empty:
        return "CACHE_PERSISTENCE_UNKNOWN"
    state = str(verification.iloc[0].get("state") or "")
    if state == "CACHE_DATABASE_COMMITTED":
        return "CACHE_FULLY_PERSISTED"
    if state in {"CACHE_DATABASE_NOT_COMMITTED", "CACHE_READBACK_SKIPPED"}:
        return "CACHE_PARTIAL_OR_UNVERIFIED"
    if state in {"CACHE_PERSISTENCE_SKIPPED", "CACHE_DATABASE_DISABLED"}:
        return "CACHE_MEMORY_ONLY"
    return state or "CACHE_PERSISTENCE_UNKNOWN"


def cache_summary(audit_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in audit_frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["cache_state", "count"])
    local = pd.concat(frames, ignore_index=True, sort=False)
    if "cache_state" not in local.columns:
        return pd.DataFrame(columns=["cache_state", "count"])
    return local["cache_state"].fillna("NO_CACHE_STATE").value_counts().rename_axis("cache_state").reset_index(name="count")


__all__ = [
    "CACHE_VERSION", "OHLCV_CACHE_TTL_HOURS", "KSEI_CACHE_TTL_HOURS", "FUNDAMENTAL_CACHE_TTL_HOURS", "NEWS_CACHE_TTL_HOURS",
    "frame_to_payload", "payload_to_frame", "build_ohlcv_cache_row", "build_source_cache_row", "read_ohlcv_cache", "read_source_cache",
    "fetch_ohlcv_cache_first", "fetch_ksei_cache_first", "fetch_fundamental_cache_first", "fetch_news_cache_first",
    "load_cached_ohlcv_frames", "load_cached_ksei", "load_cached_fundamentals", "load_cached_news",
    "persist_verify_cache_bundle", "cache_commit_succeeded", "cache_persistence_state", "cache_summary", "trim_period",
]
