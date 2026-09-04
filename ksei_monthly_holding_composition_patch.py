from __future__ import annotations

"""Official KSEI monthly holding-composition fallback.

KSEI publishes a monthly ZIP named ``BalanceposEfekYYYYMMDD.zip`` containing a
pipe-delimited balance-position file.  The file is an official slow-moving
snapshot of scripless holdings by local/foreign investor class.  It is useful
ownership context, but it is *not* regulatory free float and does not identify
beneficial owners.

The patch supplements failed/missing per-security KSEI profile requests with one
monthly archive download.  Existing verified per-security profiles always win.
"""

from functools import wraps
from io import BytesIO, StringIO
from typing import Any, Iterable
import csv
import re
import time
from urllib.parse import urljoin
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

PATCH_VERSION = "1.0.0-ksei-monthly-holding-composition"
ARCHIVE_PAGE = "https://web.ksei.co.id/archive_download/holding_composition"
USER_AGENT = "Mozilla/5.0 (compatible; IDX-Emir-Scanner/1.9; research; official-public-data)"
_CACHE: dict[str, Any] = {"fetched_at": 0.0, "profiles": pd.DataFrame(), "source_url": "", "observed_on": ""}
CACHE_TTL_SECONDS = 12 * 3600


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _number(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text or text in {"-", "--"}:
        return np.nan
    try:
        number = float(text)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _discover_archive_url(session: requests.Session, *, now: Any = None, timeout: float = 12.0) -> tuple[str, str]:
    """Prefer the archive page href; fall back to recent month-end candidates."""
    try:
        response = session.get(ARCHIVE_PAGE, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if response.ok:
            matches = re.findall(r'href=["\']([^"\']*BalanceposEfek(\d{8})\.zip[^"\']*)["\']', response.text, flags=re.I)
            if matches:
                href, stamp = sorted(matches, key=lambda item: item[1])[-1]
                return urljoin(ARCHIVE_PAGE, href), stamp
    except Exception:
        pass

    current = pd.Timestamp.now(tz="Asia/Jakarta") if now is None else pd.Timestamp(now)
    if current.tzinfo is not None:
        current = current.tz_convert("Asia/Jakarta").tz_localize(None)
    # Published files are month-end snapshots, sometimes using the last business
    # day. Search current and previous month, walking seven days backward.
    month_ends = [current.normalize(), (current.replace(day=1) - pd.Timedelta(days=1)).normalize()]
    for anchor in month_ends:
        for offset in range(0, 8):
            stamp = (anchor - pd.Timedelta(days=offset)).strftime("%Y%m%d")
            url = f"https://web.ksei.co.id/Download/BalanceposEfek{stamp}.zip"
            try:
                response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=True)
                content_type = str(response.headers.get("content-type") or "").lower()
                if response.status_code == 200 and ("zip" in content_type or url.lower().endswith(".zip")):
                    response.close()
                    return url, stamp
                response.close()
            except Exception:
                continue
    return "", ""


def parse_balancepos_zip(content: bytes, *, source_url: str = "", observed_on: str = "") -> pd.DataFrame:
    """Parse official KSEI balance-position archive into profile-compatible rows."""
    if not content:
        return pd.DataFrame()
    try:
        archive = ZipFile(BytesIO(content))
    except Exception:
        return pd.DataFrame()
    names = [name for name in archive.namelist() if "balancepos" in name.lower()]
    if not names:
        names = [name for name in archive.namelist() if name.lower().endswith((".txt", ".csv"))]
    if not names:
        return pd.DataFrame()
    raw = archive.read(names[0])
    text = ""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        return pd.DataFrame()

    reader = csv.reader(StringIO(text), delimiter="|")
    rows = list(reader)
    if len(rows) < 2:
        return pd.DataFrame()
    header = [str(item).strip().lower() for item in rows[0]]
    if len(header) < 25 or "code" not in header[1] or "type" not in header[2]:
        return pd.DataFrame()

    profiles: list[dict[str, Any]] = []
    for values in rows[1:]:
        if len(values) < 25:
            continue
        security_type = str(values[2] or "").strip().upper()
        if security_type != "EQUITY":
            continue
        ticker = _ticker(values[1])
        if not ticker:
            continue
        sec_num = _number(values[3])
        local_categories = [_number(value) for value in values[5:14]]
        foreign_categories = [_number(value) for value in values[15:24]]
        local_total = _number(values[14])
        foreign_total = _number(values[24])
        if not np.isfinite(local_total):
            local_total = float(np.nansum(local_categories))
        if not np.isfinite(foreign_total):
            foreign_total = float(np.nansum(foreign_categories))
        scripless = local_total + foreign_total if np.isfinite(local_total) and np.isfinite(foreign_total) else np.nan
        local_pct = 100.0 * local_total / scripless if np.isfinite(scripless) and scripless > 0 else np.nan
        foreign_pct = 100.0 * foreign_total / scripless if np.isfinite(scripless) and scripless > 0 else np.nan
        scripless_pct = 100.0 * scripless / sec_num if np.isfinite(sec_num) and sec_num > 0 and np.isfinite(scripless) else np.nan
        local_id = local_categories[4] if len(local_categories) > 4 else np.nan
        foreign_id = foreign_categories[4] if len(foreign_categories) > 4 else np.nan
        institutional = (
            scripless - (0.0 if not np.isfinite(local_id) else local_id) - (0.0 if not np.isfinite(foreign_id) else foreign_id)
            if np.isfinite(scripless) else np.nan
        )
        profiles.append({
            "ticker": ticker,
            "company_name": "",
            "sector": "",
            "listing_date": "",
            "security_status": "",
            "total_shares": sec_num,
            "registered_amount": np.nan,
            "scripless_pct": scripless_pct,
            "local_pct": local_pct,
            "foreign_pct": foreign_pct,
            "ksei_source_url": source_url,
            "ksei_source_verified": True,
            "ksei_observed_on": observed_on or str(values[0] or "").strip(),
            "ksei_monthly_local_total_shares": local_total,
            "ksei_monthly_foreign_total_shares": foreign_total,
            "ksei_monthly_local_individual_shares": local_id,
            "ksei_monthly_foreign_individual_shares": foreign_id,
            "ksei_monthly_institutional_shares": institutional,
            "ksei_monthly_holding_composition_state": "OFFICIAL_KSEI_SCRIPLESS_COMPOSITION_NOT_REGULATORY_FREE_FLOAT",
        })
    return pd.DataFrame(profiles)


def fetch_monthly_profiles(tickers: Iterable[str], *, now: Any = None, timeout: float = 15.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted = {_ticker(value) for value in tickers if _ticker(value)}
    if not wanted:
        return pd.DataFrame(), pd.DataFrame()
    age = time.monotonic() - float(_CACHE.get("fetched_at") or 0.0)
    cached = _CACHE.get("profiles")
    if age <= CACHE_TTL_SECONDS and isinstance(cached, pd.DataFrame) and not cached.empty:
        local = cached.loc[cached["ticker"].isin(wanted)].copy()
        return local.reset_index(drop=True), pd.DataFrame([{
            "provider": "KSEI_MONTHLY_HOLDING_COMPOSITION",
            "status": "PROCESS_CACHE_HIT",
            "items": len(local),
            "detail": f"source={_CACHE.get('source_url')}; observed_on={_CACHE.get('observed_on')}",
        }])

    session = requests.Session()
    source_url, stamp = _discover_archive_url(session, now=now, timeout=min(timeout, 12.0))
    if not source_url:
        return pd.DataFrame(), pd.DataFrame([{
            "provider": "KSEI_MONTHLY_HOLDING_COMPOSITION",
            "status": "ARCHIVE_NOT_FOUND_FAIL_SOFT",
            "items": 0,
            "detail": "Official monthly archive unavailable; no ownership value inferred.",
        }])
    try:
        response = session.get(source_url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
        profiles = parse_balancepos_zip(response.content, source_url=source_url, observed_on=stamp)
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame([{
            "provider": "KSEI_MONTHLY_HOLDING_COMPOSITION",
            "status": "DOWNLOAD_OR_PARSE_FAIL_SOFT",
            "items": 0,
            "detail": f"{type(exc).__name__}: {str(exc)[:240]}",
        }])
    if profiles.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "provider": "KSEI_MONTHLY_HOLDING_COMPOSITION",
            "status": "EMPTY_FAIL_SOFT",
            "items": 0,
            "detail": source_url,
        }])
    _CACHE.update({"fetched_at": time.monotonic(), "profiles": profiles, "source_url": source_url, "observed_on": stamp})
    local = profiles.loc[profiles["ticker"].isin(wanted)].copy()
    return local.reset_index(drop=True), pd.DataFrame([{
        "provider": "KSEI_MONTHLY_HOLDING_COMPOSITION",
        "status": "OFFICIAL_ARCHIVE_CURRENT",
        "items": len(local),
        "detail": f"source={source_url}; observed_on={stamp}; semantics=scripless composition not regulatory free float",
    }])


def _wrap_fetch_many(module: Any) -> None:
    original = getattr(module, "fetch_many_ksei_profiles", None)
    if not callable(original) or getattr(original, "__ksei_monthly_holding_fallback_v1__", False):
        return

    @wraps(original)
    def wrapped(tickers: Iterable[str], max_workers: int = 2):
        requested = [_ticker(value) for value in tickers if _ticker(value)]
        profiles, actions, audit = original(requested, max_workers=max_workers)
        profiles = profiles.copy() if isinstance(profiles, pd.DataFrame) else pd.DataFrame()
        verified_tickers: set[str] = set()
        if not profiles.empty and "ticker" in profiles.columns:
            verified = profiles.get("ksei_source_verified", pd.Series(False, index=profiles.index)).fillna(False).astype(bool)
            verified_tickers = set(profiles.loc[verified, "ticker"].map(_ticker))
        missing = [ticker for ticker in requested if ticker not in verified_tickers]
        monthly = monthly_audit = pd.DataFrame()
        if missing:
            monthly, monthly_audit = fetch_monthly_profiles(missing)
        if not monthly.empty:
            profiles = pd.concat([profiles, monthly], ignore_index=True, sort=False)
            # Keep higher-resolution verified per-security profile when present.
            profiles["ticker"] = profiles["ticker"].map(_ticker)
            profiles["_verified"] = profiles.get("ksei_source_verified", False).fillna(False).astype(bool)
            profiles["_monthly"] = profiles.get("ksei_monthly_holding_composition_state", pd.Series("", index=profiles.index)).fillna("").astype(str).str.len().gt(0)
            profiles = profiles.sort_values(["ticker", "_verified", "_monthly"], ascending=[True, False, True]).drop_duplicates("ticker", keep="first")
            profiles = profiles.drop(columns=["_verified", "_monthly"], errors="ignore").reset_index(drop=True)
        audits = [frame for frame in (audit, monthly_audit) if isinstance(frame, pd.DataFrame) and not frame.empty]
        audit_out = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
        return profiles, actions, audit_out

    wrapped.__ksei_monthly_holding_fallback_v1__ = True
    setattr(module, "fetch_many_ksei_profiles", wrapped)


def install() -> dict[str, str]:
    import autonomous_enrichment
    import persistent_cache

    # persistent_cache imported the function by name, so both bindings must be
    # patched to make the fallback effective on cache misses.
    _wrap_fetch_many(autonomous_enrichment)
    _wrap_fetch_many(persistent_cache)
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "source": "OFFICIAL_KSEI_MONTHLY_HOLDING_COMPOSITION",
        "semantics": "SCRIPLESS_LOCAL_FOREIGN_COMPOSITION_NOT_REGULATORY_FREE_FLOAT",
    }


__all__ = ["PATCH_VERSION", "fetch_monthly_profiles", "install", "parse_balancepos_zip"]
