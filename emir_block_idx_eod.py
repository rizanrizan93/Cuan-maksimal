from __future__ import annotations

"""Official Block IDX producer for EMIR.

This module is intentionally the only EMIR runtime allowed to call Block IDX primary APIs.
Official XBRL attachment bytes may use IDX-owned static hosts. The interactive
scanner consumes normalized/cache tables through Supabase only.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from typing import Any, Iterable, Mapping
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo
import json
import math
import os
import time

import pandas as pd
from curl_cffi import requests as curl_requests
import requests

BLOCK_IDX_BASE = "https://block.idx.id"
WIB = ZoneInfo("Asia/Jakarta")
PRODUCER_VERSION = "EMIR_BLOCK_IDX_EOD_V1"
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class EndpointSpec:
    key: str
    family: str
    path: str
    acquisition_class: str
    cadence: str
    history: bool
    scoring_role: str
    verified: bool = True


# Direct routes proven by the existing official IDX/IDX Flow contracts. Routes
# that are not proven are catalogued in SQL as CATALOG_ONLY and are never guessed.
ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec("stock_summary", "MARKET_DAILY", "/primary/TradingSummary/GetStockSummary", "A", "EOD", True, "PRICE_LIQUIDITY_FOREIGN"),
    EndpointSpec("index_summary", "INDEX_DAILY", "/primary/TradingSummary/GetIndexSummary", "A", "EOD", True, "MARKET_SECTOR_REGIME"),
    EndpointSpec("broker_summary", "BROKER_MARKET_DAILY", "/primary/TradingSummary/GetBrokerSummary", "D", "EOD", True, "MARKET_ONLY_NOT_TICKER_FLOW"),
    EndpointSpec("companies", "COMPANY_REFERENCE", "/primary/ListedCompany/GetCompanyProfiles", "C", "MONTHLY", False, "UNIVERSE_IDENTITY"),
    EndpointSpec("company_profile", "COMPANY_DETAIL", "/primary/ListedCompany/GetCompanyProfilesDetail", "C", "ROTATING_WEEKLY", False, "OWNERSHIP_CONTROLLER"),
    EndpointSpec("financial_report", "FINANCIAL_REPORT", "/primary/ListedCompany/GetFinancialReport", "B", "EVENT", True, "OFFICIAL_FUNDAMENTAL"),
    EndpointSpec("announcements", "ANNOUNCEMENT", "/primary/NewsAnnouncement/GetAllAnnouncement", "B", "EOD_DELTA", True, "CATALYST_RISK"),
    EndpointSpec("company_announcements", "ANNOUNCEMENT", "/primary/ListedCompany/GetProfileAnnouncement", "D", "FINALIST", True, "CATALYST_CONFIRMATION"),
    EndpointSpec("uma", "RISK_EVENT", "/primary/NewsAnnouncement/GetUma", "B", "EOD_DELTA", True, "EXECUTION_GUARD"),
    EndpointSpec("suspension", "RISK_EVENT", "/primary/NewsAnnouncement/GetSuspension", "B", "EOD_DELTA", True, "HARD_BLOCK"),
    EndpointSpec("issued_history", "CAPITAL_ACTION", "/primary/ListingActivity/GetIssuedHistory", "B", "EOD_DELTA", True, "DILUTION_CAPITAL_ACTION"),
    EndpointSpec("trading_info", "TRADING_DETAIL", "/primary/ListedCompany/GetTradingInfoSS", "D", "FINALIST", True, "FINALIST_DIAGNOSTIC"),
)


def endpoint_catalog() -> list[dict[str, Any]]:
    return [spec.__dict__.copy() for spec in ENDPOINTS]


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _ticker(value: Any) -> str:
    value = _clean(value).upper().replace(".JK", "")
    return value if 2 <= len(value) <= 12 and value.replace("-", "").replace(".", "").isalnum() else ""


def _iso_date(value: Any) -> str | None:
    try:
        parsed = pd.Timestamp(value)
    except Exception:
        return None
    return parsed.date().isoformat() if pd.notna(parsed) else None


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "Results", "results", "Data", "items", "Items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def _payload_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return sha256(raw.encode("utf-8")).hexdigest()


class BlockIdxClient:
    def __init__(self, *, timeout: float = 35.0, retries: int = 2, pace_seconds: float = 0.35):
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.pace_seconds = max(0.0, float(pace_seconds))
        self.session = curl_requests.Session(impersonate="chrome")
        self.headers = {
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "id-ID,id;q=0.9,en;q=0.7",
            "Referer": f"{BLOCK_IDX_BASE}/id",
            "User-Agent": "Mozilla/5.0",
        }

    def get_json(self, spec: EndpointSpec, params: Mapping[str, Any]) -> tuple[Any, str, int]:
        if not spec.verified:
            raise ValueError(f"unverified route rejected: {spec.key}")
        url = urljoin(BLOCK_IDX_BASE, spec.path)
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "block.idx.id":
            raise ValueError("non-Block-Idx route rejected")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    url,
                    params={k: v for k, v in params.items() if v is not None},
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
                status = int(response.status_code)
                if 300 <= status < 400:
                    raise RuntimeError(f"redirect rejected HTTP {status}")
                if status == 200:
                    payload = response.json()
                    final_url = str(response.url)
                    if urlparse(final_url).netloc != "block.idx.id":
                        raise RuntimeError("response host mismatch")
                    time.sleep(self.pace_seconds)
                    return payload, final_url, status
                if status not in TRANSIENT_HTTP or attempt >= self.retries:
                    raise RuntimeError(f"Block IDX HTTP {status} for {spec.key}")
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 1.2 * (2 ** attempt)
                time.sleep(min(12.0, delay))
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(12.0, 1.2 * (2 ** attempt)))
        raise RuntimeError(f"{spec.key} unavailable: {type(last_error).__name__}: {last_error}") from last_error


class SupabaseSink:
    def __init__(self, url: str, secret: str, *, timeout: float = 45.0):
        self.url = str(url).rstrip("/")
        self.secret = str(secret).strip()
        self.timeout = float(timeout)
        if not self.url.startswith("https://") or not self.secret:
            raise ValueError("dedicated EMIR Supabase URL and backend secret are required")
        if self.secret.startswith(("eyJ", "sb_publishable_")):
            raise ValueError("publishable/anon key rejected; backend secret/service-role required")
        self.headers = {
            "apikey": self.secret,
            "Authorization": f"Bearer {self.secret}",
            "Content-Type": "application/json",
        }

    @classmethod
    def from_env(cls) -> "SupabaseSink":
        return cls(
            os.getenv("SUPABASE_URL", ""),
            os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        )

    def upsert(self, table: str, rows: Iterable[Mapping[str, Any]], conflict: str, *, chunk: int = 400) -> int:
        records = [dict(row) for row in rows]
        total = 0
        for start in range(0, len(records), max(1, int(chunk))):
            batch = records[start:start + chunk]
            response = requests.post(
                f"{self.url}/rest/v1/{table}",
                params={"on_conflict": conflict},
                headers={**self.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
                data=json.dumps(batch, separators=(",", ":"), ensure_ascii=False, default=str),
                timeout=self.timeout,
            )
            response.raise_for_status()
            total += len(batch)
        return total

    def select(self, table: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        response = requests.get(
            f"{self.url}/rest/v1/{table}",
            headers=self.headers,
            params=dict(params),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def rpc(self, name: str, payload: Mapping[str, Any] | None = None) -> Any:
        response = requests.post(
            f"{self.url}/rest/v1/rpc/{name}",
            headers=self.headers,
            json=dict(payload or {}),
            timeout=max(self.timeout, 120.0),
        )
        response.raise_for_status()
        return response.json() if response.content else None

    def ingestion_run(self, run_key: str, **values: Any) -> None:
        self.upsert(
            "cak_idx_ingestion_runs",
            [{"run_key": run_key, **values}],
            "run_key",
            chunk=1,
        )


def normalize_stock_summary(payload: Any, requested: date, source_url: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _rows(payload):
        ticker = _ticker(item.get("StockCode"))
        trade_date = _iso_date(item.get("Date"))
        if not ticker or trade_date != requested.isoformat():
            continue
        volume = _number(item.get("Volume")) or 0.0
        value = _number(item.get("Value")) or 0.0
        frequency = _number(item.get("Frequency")) or 0.0
        if min(volume, value, frequency) < 0:
            continue
        foreign_buy = _number(item.get("ForeignBuy")) or 0.0
        foreign_sell = _number(item.get("ForeignSell")) or 0.0
        out.append({
            "trade_date": trade_date,
            "ticker": ticker,
            "stock_name": _clean(item.get("StockName")) or None,
            "previous": _number(item.get("Previous")),
            "open": _number(item.get("OpenPrice")),
            "high": _number(item.get("High")),
            "low": _number(item.get("Low")),
            "close": _number(item.get("Close")),
            "change": _number(item.get("Change")),
            "volume": volume,
            "traded_value": value,
            "frequency": frequency,
            "foreign_buy": foreign_buy,
            "foreign_sell": foreign_sell,
            "foreign_net": foreign_buy - foreign_sell,
            "listed_shares": _number(item.get("ListedShares")),
            "tradable_shares": _number(item.get("TradebleShares")),
            "bid": _number(item.get("Bid")),
            "offer": _number(item.get("Offer")),
            "bid_volume": _number(item.get("BidVolume")),
            "offer_volume": _number(item.get("OfferVolume")),
            "non_regular_volume": _number(item.get("NonRegularVolume")),
            "non_regular_value": _number(item.get("NonRegularValue")),
            "source_url": source_url,
            "payload_hash": _payload_hash(item),
            "source_verified": True,
            "provenance_state": "VERIFIED_OFFICIAL_BLOCK_IDX_STOCK_SUMMARY",
        })
    return out


def normalize_index_summary(payload: Any, requested: date, source_url: str) -> list[dict[str, Any]]:
    out = []
    for item in _rows(payload):
        code = _clean(item.get("IndexCode")).upper()
        trade_date = _iso_date(item.get("Date"))
        if not code or trade_date != requested.isoformat():
            continue
        out.append({
            "trade_date": trade_date, "index_code": code,
            "previous": _number(item.get("Previous")), "highest": _number(item.get("Highest")),
            "lowest": _number(item.get("Lowest")), "close": _number(item.get("Close")),
            "number_of_stock": _number(item.get("NumberOfStock")), "change": _number(item.get("Change")),
            "volume": _number(item.get("Volume")), "traded_value": _number(item.get("Value")),
            "frequency": _number(item.get("Frequency")), "market_capital": _number(item.get("MarketCapital")),
            "source_url": source_url, "payload_hash": _payload_hash(item), "source_verified": True,
        })
    return out


def normalize_broker_summary(payload: Any, requested: date, source_url: str) -> list[dict[str, Any]]:
    out = []
    for item in _rows(payload):
        code = _clean(item.get("IDFirm")).upper()
        trade_date = _iso_date(item.get("Date"))
        if not code or trade_date != requested.isoformat():
            continue
        out.append({
            "trade_date": trade_date, "broker_code": code, "broker_name": _clean(item.get("FirmName")) or None,
            "traded_value": _number(item.get("Value")) or 0.0, "volume": _number(item.get("Volume")) or 0.0,
            "frequency": _number(item.get("Frequency")) or 0.0, "source_url": source_url,
            "payload_hash": _payload_hash(item), "source_verified": True,
            "semantic_scope": "MARKET_WIDE_NO_TICKER_BUY_SELL_SPLIT",
        })
    return out


def generic_events(payload: Any, family: str, source_url: str, observed_on: date) -> list[dict[str, Any]]:
    rows = []
    for item in _rows(payload):
        ticker = _ticker(item.get("CompanyID") or item.get("Kode") or item.get("KodeEmiten") or item.get("StockCode"))
        event_date = _iso_date(
            item.get("UMADate") or item.get("Date") or item.get("TanggalPencatatan")
            or item.get("PublishDate") or item.get("PublishedDate") or observed_on
        )
        source_ref = _clean(
            item.get("UMAID") or item.get("AnnouncementNo") or item.get("Data_Download")
            or item.get("Attachment") or item.get("ID") or item.get("Id") or item.get("id") or _payload_hash(item)
        )
        if not event_date:
            continue
        if family == "SUSPENSION":
            text = (_clean(item.get("Info_Type")) + " " + _clean(item.get("Judul"))).lower()
            event_type = "UNSUSPEND" if ("upt" in text or "pembukaan" in text or "unsuspend" in text) else "SUSPEND"
        elif family == "UMA":
            event_type = "UMA"
        elif family == "ISSUED_HISTORY":
            event_type = _clean(item.get("JenisTindakan")).upper().replace(" ", "_") or "CAPITAL_ACTION"
        else:
            event_type = family
        rows.append({
            "event_family": family, "event_type": event_type, "ticker": ticker or None,
            "event_date": event_date, "publication_date": _iso_date(item.get("PublishDate") or item.get("PublishedDate")),
            "source_ref": source_ref[:512], "title": _clean(item.get("Judul") or item.get("Title")) or None,
            "source_url": source_url, "payload_hash": _payload_hash(item), "source_verified": True,
            "raw_payload": item,
        })
    return rows


def normalize_companies(payload: Any, observed_on: date, source_url: str) -> list[dict[str, Any]]:
    out = []
    for item in _rows(payload):
        ticker = _ticker(item.get("KodeEmiten") or item.get("Code") or item.get("StockCode"))
        if not ticker:
            continue
        out.append({
            "ticker": ticker, "observed_on": observed_on.isoformat(),
            "company_name": _clean(item.get("NamaEmiten") or item.get("NamaPerusahaan") or item.get("Name")) or None,
            "sector": _clean(item.get("Sektor") or item.get("Sector")) or None,
            "subsector": _clean(item.get("SubSektor") or item.get("SubSector")) or None,
            "listing_date": _iso_date(item.get("TanggalPencatatan") or item.get("ListingDate")),
            "source_url": source_url, "payload_hash": _payload_hash(item), "source_verified": True,
            "raw_payload": item,
        })
    return out


def _spec(key: str) -> EndpointSpec:
    return next(item for item in ENDPOINTS if item.key == key)


class EmirBlockIdxProducer:
    def __init__(self, client: BlockIdxClient, sink: SupabaseSink):
        self.client = client
        self.sink = sink

    def _manifest(self, spec: EndpointSpec, target_date: date | None, source_url: str, payload: Any, accepted: int, state: str = "OK") -> None:
        self.sink.upsert("cak_idx_payload_manifest", [{
            "provider": "BLOCK_IDX_OFFICIAL", "endpoint_key": spec.key,
            "target_date": target_date.isoformat() if target_date else date.today().isoformat(),
            "source_url": source_url, "payload_hash": _payload_hash(payload),
            "rows_received": len(_rows(payload)), "rows_accepted": int(accepted),
            "validation_state": state, "producer_version": PRODUCER_VERSION,
        }], "provider,endpoint_key,target_date,payload_hash", chunk=1)

    def collect_market_day(self, target: date) -> dict[str, int]:
        ymd = target.strftime("%Y%m%d")
        counts: dict[str, int] = {}
        jobs = (
            ("stock_summary", {"length": 2000, "start": 0, "date": ymd}, normalize_stock_summary, "cak_idx_market_daily", "trade_date,ticker"),
            ("index_summary", {"length": 1000, "start": 0, "date": ymd}, normalize_index_summary, "cak_idx_index_daily", "trade_date,index_code"),
            ("broker_summary", {"length": 200, "start": 0, "date": ymd}, normalize_broker_summary, "cak_idx_broker_market_daily", "trade_date,broker_code"),
        )
        for key, params, normalizer, table, conflict in jobs:
            spec = _spec(key)
            payload, url, _ = self.client.get_json(spec, params)
            rows = normalizer(payload, target, url)
            if key == "stock_summary":
                total = int(payload.get("recordsTotal") or 0) if isinstance(payload, dict) else 0
                if not rows or (total and len(rows) != total):
                    raise RuntimeError(f"incomplete core StockSummary {target}: accepted={len(rows)} total={total}")
            persisted = self.sink.upsert(table, rows, conflict) if rows else 0
            self._manifest(spec, target, url, payload, persisted, "VALID" if rows else "NO_DATA")
            counts[key] = persisted
        return counts

    def collect_event_window(self, start: date, end: date) -> dict[str, int]:
        dfrom, dto = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        requests_to_make = (
            ("uma", {"dateFrom": dfrom, "dateTo": dto, "indexfrom": 0, "pagesize": 5000}, "UMA"),
            ("suspension", {"dateFrom": dfrom, "dateTo": dto, "indexfrom": 0, "pagesize": 5000}, "SUSPENSION"),
            ("issued_history", {"caType": "", "dateFrom": dfrom, "dateTo": dto, "start": 0, "length": 5000}, "ISSUED_HISTORY"),
        )
        counts: dict[str, int] = {}
        for key, params, family in requests_to_make:
            spec = _spec(key)
            try:
                payload, url, _ = self.client.get_json(spec, params)
                rows = generic_events(payload, family, url, end)
                persisted = self.sink.upsert(
                    "cak_idx_events", rows,
                    "event_family,event_type,event_date,source_ref,payload_hash",
                ) if rows else 0
                self._manifest(spec, end, url, payload, persisted, "VALID_EMPTY" if not rows else "VALID")
                counts[key] = persisted
            except Exception as exc:
                counts[key] = 0
                self.sink.upsert("cak_idx_ingestion_failures", [{
                    "endpoint_key": key, "target_date": end.isoformat(),
                    "failure_class": type(exc).__name__, "message": str(exc)[:1000],
                    "retryable": any(token in str(exc) for token in ("429", "500", "502", "503", "504", "timeout")),
                }], "endpoint_key,target_date,failure_class,message", chunk=1)

        spec = _spec("announcements")
        announcement_total = 0
        try:
            page = 1
            while page <= 100:
                payload, url, _ = self.client.get_json(spec, {
                    "keywords": "", "dateFrom": dfrom, "dateTo": dto,
                    "pageNumber": page, "pageSize": 1000, "lang": "id",
                })
                rows = generic_events(payload, "ANNOUNCEMENT", url, end)
                announcement_total += self.sink.upsert(
                    "cak_idx_events", rows,
                    "event_family,event_type,event_date,source_ref,payload_hash",
                ) if rows else 0
                self._manifest(spec, end, url, payload, len(rows), "VALID_EMPTY" if not rows else "VALID")
                page_count = max(1, int(payload.get("PageCount") or 1)) if isinstance(payload, dict) else 1
                if page >= page_count:
                    break
                page += 1
            counts["announcements"] = announcement_total
        except Exception as exc:
            counts["announcements"] = announcement_total
            self.sink.upsert("cak_idx_ingestion_failures", [{
                "endpoint_key": "announcements", "target_date": end.isoformat(),
                "failure_class": type(exc).__name__, "message": str(exc)[:1000],
                "retryable": any(token in str(exc) for token in ("429", "500", "502", "503", "504", "timeout")),
            }], "endpoint_key,target_date,failure_class,message", chunk=1)
        return counts

    def collect_company_reference(self, observed_on: date) -> int:
        spec = _spec("companies")
        payload, url, _ = self.client.get_json(spec, {"emitenType": "s", "start": 0, "length": 2000})
        rows = normalize_companies(payload, observed_on, url)
        if len(rows) < 800:
            raise RuntimeError(f"company directory unexpectedly small: {len(rows)}")
        persisted = self.sink.upsert("cak_idx_company_snapshot", rows, "ticker,observed_on")
        self._manifest(spec, observed_on, url, payload, persisted)
        return persisted

    def collect_financial_index(self, observed_on: date) -> int:
        spec = _spec("financial_report")
        periods = ((observed_on.year, "TW1"), (observed_on.year, "TW2"), (observed_on.year - 1, "audit"))
        total = 0
        for year, period in periods:
            payload, url, _ = self.client.get_json(spec, {
                "periode": period, "year": year, "indexFrom": 0, "pageSize": 2000,
                "reportType": "rdf", "kodeEmiten": "",
            })
            rows = generic_events(payload, "FINANCIAL_REPORT", url, observed_on)
            total += self.sink.upsert(
                "cak_idx_events", rows,
                "event_family,event_type,event_date,source_ref,payload_hash",
            ) if rows else 0
            self._manifest(spec, observed_on, url, payload, len(rows), "VALID_EMPTY" if not rows else "VALID")
        return total

    def collect_official_fundamentals(self, tickers: Iterable[str], observed_on: date) -> int:
        from idx_official_fundamentals import fetch_many_idx_official_fundamentals

        names = list(dict.fromkeys(_ticker(t) for t in tickers if _ticker(t)))
        if not names:
            return 0
        frame, _audit = fetch_many_idx_official_fundamentals(names, now=pd.Timestamp(observed_on), max_workers=2)
        if frame.empty:
            return 0
        rows = []
        for item in frame.to_dict(orient="records"):
            if not bool(item.get("idx_official_source_verified")):
                continue
            ticker = _ticker(item.get("ticker"))
            equity = _number(item.get("idx_official_equity"))
            assets = _number(item.get("idx_official_assets"))
            income = _number(item.get("idx_official_net_income"))
            rows.append({
                "ticker": ticker, "period_end": _iso_date(item.get("idx_official_period_end")),
                "observed_on": observed_on.isoformat(), "period_type": _clean(item.get("idx_official_period")),
                "revenue": _number(item.get("idx_official_revenue")),
                "revenue_growth_yoy_pct": _number(item.get("idx_official_revenue_growth_yoy_pct")),
                "net_income": income, "earnings_growth_yoy_pct": _number(item.get("idx_official_earnings_growth_yoy_pct")),
                "net_margin_pct": _number(item.get("idx_official_net_margin_pct")),
                "roe_pct": (100.0 * income / equity) if income is not None and equity and equity > 0 else None,
                "roa_pct": (100.0 * income / assets) if income is not None and assets and assets > 0 else None,
                "ocf": _number(item.get("idx_official_ocf")), "fcf": _number(item.get("idx_official_fcf_proxy")),
                "cash": _number(item.get("idx_official_cash")),
                "debt": _number(item.get("idx_official_interest_bearing_debt_proxy")),
                "debt_to_equity": _number(item.get("idx_official_interest_bearing_debt_to_equity")),
                "current_ratio": _number(item.get("idx_official_current_ratio")),
                "cash_to_debt_ratio": _number(item.get("idx_official_cash_to_debt_ratio")),
                "source_url": _clean(item.get("idx_official_source_url")),
                "coverage_pct": _number(item.get("idx_official_coverage_pct")),
                "payload_hash": _payload_hash(item), "source_verified": True, "raw_payload": item,
            })
        return self.sink.upsert(
            "cak_idx_fundamental_snapshot", rows,
            "ticker,period_end,observed_on,payload_hash",
        ) if rows else 0

    def refresh_ranking(self, as_of: date) -> Any:
        return self.sink.rpc("cak_refresh_idx_ranking_v1", {"p_as_of": as_of.isoformat()})


def weekdays(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            yield cursor
        cursor += timedelta(days=1)


__all__ = [
    "BLOCK_IDX_BASE", "ENDPOINTS", "EndpointSpec", "PRODUCER_VERSION",
    "BlockIdxClient", "SupabaseSink", "EmirBlockIdxProducer", "endpoint_catalog",
    "normalize_stock_summary", "normalize_index_summary", "normalize_broker_summary",
    "generic_events", "normalize_companies", "weekdays",
]
