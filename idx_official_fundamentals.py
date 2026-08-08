from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable, Mapping
from urllib.parse import quote
from zipfile import ZipFile, BadZipFile
import math
import random
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

from data_providers import USER_AGENT, bare_ticker, normalize_ticker

IDX_STATIC_BASE = (
    "https://www.idx.co.id/Portals/0/StaticData/ListedCompanies/Corporate_Actions/"
    "New_Info_JSX/Jenis_Informasi/01_Laporan_Keuangan/02_Soft_Copy_Laporan_Keuangan"
)
IDX_FINANCIAL_PORTAL = "https://idx.id/en/listed-companies/financial-statements-and-annual-report"

_PERIOD_END_MONTH = {"TW1": 3, "TW2": 6, "TW3": 9, "AUDIT": 12}


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    number = _finite(value, low)
    return float(min(high, max(low, number)))


def idx_instance_url(ticker: str, year: int, period: str) -> str:
    symbol = bare_ticker(ticker)
    label = str(period).upper()
    return (
        f"{IDX_STATIC_BASE}//Laporan%20Keuangan%20Tahun%20{int(year)}/"
        f"{label}/{quote(symbol)}/instance.zip"
    )


def candidate_reporting_periods(now: Any = None, *, max_candidates: int = 5) -> list[tuple[int, str]]:
    current = pd.Timestamp.now(tz="Asia/Jakarta") if now is None else pd.Timestamp(now)
    if current.tzinfo is not None:
        current = current.tz_convert("Asia/Jakarta")
    year, month = int(current.year), int(current.month)
    candidates: list[tuple[int, str]] = []
    # Indonesian quarterly reports are generally available after the quarter closes.
    # Try newest plausible period first, then fall back. 404/503 is treated as evidence
    # unavailable, never as a zero-valued financial statement.
    if month >= 11:
        candidates.append((year, "TW3"))
    if month >= 8:
        candidates.append((year, "TW2"))
    if month >= 5:
        candidates.append((year, "TW1"))
    candidates.extend([(year - 1, "AUDIT"), (year - 1, "TW3")])
    return list(dict.fromkeys(candidates))[: max(1, int(max_candidates))]


def _local_name(tag: str) -> str:
    return str(tag).split("}")[-1]


@dataclass(frozen=True)
class ContextPeriod:
    context_id: str
    start: pd.Timestamp | pd.NaT
    end: pd.Timestamp | pd.NaT
    instant: pd.Timestamp | pd.NaT

    @property
    def effective_end(self) -> pd.Timestamp | pd.NaT:
        return self.instant if pd.notna(self.instant) else self.end

    @property
    def duration_days(self) -> float:
        if pd.isna(self.start) or pd.isna(self.end):
            return np.nan
        return float((self.end - self.start).days + 1)


def _parse_date(text: Any) -> pd.Timestamp | pd.NaT:
    value = pd.to_datetime(text, errors="coerce")
    if isinstance(value, pd.DatetimeIndex):
        return value[0] if len(value) else pd.NaT
    return value if pd.notna(value) else pd.NaT


def _parse_numeric(text: Any) -> float:
    if text is None:
        return np.nan
    cleaned = str(text).strip().replace(",", "")
    if cleaned in {"", "-", "—"}:
        return np.nan
    try:
        number = float(cleaned)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def _extract_instance_bytes(content: bytes) -> bytes:
    try:
        with ZipFile(BytesIO(content)) as archive:
            names = archive.namelist()
            preferred = next((name for name in names if name.lower().endswith("instance.xbrl")), None)
            preferred = preferred or next((name for name in names if name.lower().endswith((".xbrl", ".xml"))), None)
            if not preferred:
                raise ValueError("IDX instance.zip contains no XBRL/XML instance")
            return archive.read(preferred)
    except BadZipFile as exc:
        raise ValueError("IDX response is not a valid instance.zip") from exc


def parse_idx_xbrl_instance(ticker: str, xml_bytes: bytes, *, source_url: str = "", period_label: str = "") -> dict[str, Any]:
    symbol = normalize_ticker(ticker)
    root = ET.fromstring(xml_bytes)
    contexts: dict[str, ContextPeriod] = {}
    for elem in root.iter():
        if _local_name(elem.tag) != "context":
            continue
        cid = str(elem.attrib.get("id") or "")
        if not cid:
            continue
        start = end = instant = pd.NaT
        for child in elem.iter():
            name = _local_name(child.tag)
            if name == "startDate":
                start = _parse_date(child.text)
            elif name == "endDate":
                end = _parse_date(child.text)
            elif name == "instant":
                instant = _parse_date(child.text)
        contexts[cid] = ContextPeriod(cid, start, end, instant)

    facts: list[dict[str, Any]] = []
    for elem in root.iter():
        context_ref = elem.attrib.get("contextRef")
        if not context_ref or context_ref not in contexts:
            continue
        value = _parse_numeric(elem.text)
        if not np.isfinite(value):
            continue
        facts.append({
            "name": _local_name(elem.tag),
            "context": str(context_ref),
            "value": value,
            "unit": str(elem.attrib.get("unitRef") or ""),
        })
    if not facts:
        raise ValueError("IDX XBRL contains no numeric facts")

    duration_contexts = [c for c in contexts.values() if pd.notna(c.end) and pd.notna(c.start)]
    instant_contexts = [c for c in contexts.values() if pd.notna(c.instant)]
    current_end = max((c.end for c in duration_contexts if pd.notna(c.end)), default=pd.NaT)
    if pd.isna(current_end):
        current_end = max((c.instant for c in instant_contexts if pd.notna(c.instant)), default=pd.NaT)
    if pd.isna(current_end):
        raise ValueError("IDX XBRL reporting period cannot be resolved")

    # Prefer consolidated, non-segmented contexts. IDX commonly uses CurrentYearDuration,
    # PriorYearDuration, CurrentYearInstant. Member/segment contexts are deliberately
    # de-prioritised to avoid accidentally reading one business segment as the issuer total.
    def context_score(cid: str, *, duration: bool, target_end: pd.Timestamp) -> tuple[int, int, int]:
        c = contexts.get(cid)
        if not c:
            return (-999, -999, -999)
        end = c.end if duration else c.instant
        if pd.isna(end):
            return (-999, -999, -999)
        distance = abs(int((pd.Timestamp(end) - pd.Timestamp(target_end)).days))
        lower = cid.lower()
        preferred = int(("currentyearduration" in lower) if duration else ("currentyearinstant" in lower))
        segmented = int(any(token in lower for token in ("member", "segment", "product", "service", "party")))
        return (preferred * 100 - segmented * 20 - distance, -segmented, -len(cid))

    def value_for(names: Iterable[str], *, duration: bool, target_end: pd.Timestamp) -> float:
        allowed = set(names)
        candidates = [f for f in facts if f["name"] in allowed]
        if not candidates:
            return np.nan
        ranked = sorted(candidates, key=lambda f: context_score(f["context"], duration=duration, target_end=target_end), reverse=True)
        best = ranked[0]
        c = contexts.get(best["context"])
        actual_end = c.end if duration and c else c.instant if c else pd.NaT
        if pd.isna(actual_end) or abs((pd.Timestamp(actual_end) - pd.Timestamp(target_end)).days) > 45:
            return np.nan
        return _finite(best["value"], np.nan)

    # Prior comparable period: same duration length, end ~1 year earlier.
    current_duration_candidates = [c for c in duration_contexts if abs((c.end - current_end).days) <= 7]
    current_duration = max(current_duration_candidates, key=lambda c: (0 if np.isnan(c.duration_days) else c.duration_days), default=None)
    target_prior_end = pd.Timestamp(current_end) - pd.DateOffset(years=1)
    prior_candidates = [
        c for c in duration_contexts
        if abs((c.end - target_prior_end).days) <= 45
        and (current_duration is None or not np.isfinite(current_duration.duration_days) or not np.isfinite(c.duration_days) or abs(c.duration_days - current_duration.duration_days) <= 35)
    ]
    prior_end = max((c.end for c in prior_candidates), default=pd.NaT)

    revenue_names = ("SalesAndRevenue", "Revenues", "Revenue", "OperatingRevenues")
    profit_names = ("ProfitLossAttributableToParentEntity", "ProfitLoss")
    operating_profit_names = ("OperatingProfitLoss", "ProfitLossFromOperatingActivities", "OperatingIncomeExpense")
    ocf_names = ("NetCashFlowsReceivedFromUsedInOperatingActivities", "NetCashFlowsFromUsedInOperatingActivities")
    asset_names = ("Assets", "TotalAssets")
    liability_names = ("Liabilities", "TotalLiabilities")
    equity_names = ("EquityAttributableToEquityOwnersOfParentEntity", "Equity")
    current_asset_names = ("CurrentAssets",)
    current_liability_names = ("CurrentLiabilities",)
    cash_names = ("CashAndCashEquivalents", "CashAndCashEquivalentsCashFlows")
    debt_names = (
        "ShortTermBankLoans", "LongTermBankLoans", "CurrentMaturitiesOfBankLoans",
        "CurrentMaturitiesOfBondsPayable", "BondsPayable", "LongTermBondsPayable",
        "ShortTermBorrowings", "LongTermBorrowings", "FinanceLeaseLiabilities",
    )
    capex_names = (
        "PaymentsForAcquisitionOfPropertyPlantAndEquipment",
        "PaymentsForAcquisitionOfIntangibleAssets",
        "PaymentsForAcquisitionOfOilAndGasAssets",
        "PaymentsForAcquisitionOfMiningProperties",
        "PaymentsForAcquisitionOfInvestmentProperties",
    )

    revenue = value_for(revenue_names, duration=True, target_end=current_end)
    net_income = value_for(profit_names, duration=True, target_end=current_end)
    operating_income = value_for(operating_profit_names, duration=True, target_end=current_end)
    ocf = value_for(ocf_names, duration=True, target_end=current_end)
    prior_revenue = value_for(revenue_names, duration=True, target_end=prior_end) if pd.notna(prior_end) else np.nan
    prior_net_income = value_for(profit_names, duration=True, target_end=prior_end) if pd.notna(prior_end) else np.nan

    assets = value_for(asset_names, duration=False, target_end=current_end)
    liabilities = value_for(liability_names, duration=False, target_end=current_end)
    equity = value_for(equity_names, duration=False, target_end=current_end)
    current_assets = value_for(current_asset_names, duration=False, target_end=current_end)
    current_liabilities = value_for(current_liability_names, duration=False, target_end=current_end)
    cash = value_for(cash_names, duration=False, target_end=current_end)

    debt_parts = []
    for name in debt_names:
        value = value_for((name,), duration=False, target_end=current_end)
        if np.isfinite(value):
            debt_parts.append(abs(value))
    debt = float(sum(debt_parts)) if debt_parts else np.nan

    capex_parts = []
    for name in capex_names:
        value = value_for((name,), duration=True, target_end=current_end)
        if np.isfinite(value) and abs(value) > 0:
            capex_parts.append(abs(value))
    capex_outflow = -float(sum(capex_parts)) if capex_parts else np.nan
    fcf = ocf + capex_outflow if np.isfinite(ocf) and np.isfinite(capex_outflow) else np.nan

    def pct(current: float, prior: float) -> float:
        return 100.0 * (current / prior - 1.0) if np.isfinite(current) and np.isfinite(prior) and prior != 0 else np.nan

    revenue_yoy = pct(revenue, prior_revenue)
    earnings_yoy = pct(net_income, prior_net_income)
    current_ratio = current_assets / current_liabilities if np.isfinite(current_assets) and np.isfinite(current_liabilities) and current_liabilities != 0 else np.nan
    ide = debt / equity if np.isfinite(debt) and np.isfinite(equity) and equity != 0 else np.nan
    lte = liabilities / equity if np.isfinite(liabilities) and np.isfinite(equity) and equity != 0 else np.nan
    net_debt_to_equity = (debt - cash) / equity if np.isfinite(debt) and np.isfinite(cash) and np.isfinite(equity) and equity != 0 else np.nan
    cash_to_debt = cash / debt if np.isfinite(cash) and np.isfinite(debt) and debt > 0 else np.nan
    margin = 100.0 * net_income / revenue if np.isfinite(net_income) and np.isfinite(revenue) and revenue != 0 else np.nan
    operating_margin = 100.0 * operating_income / revenue if np.isfinite(operating_income) and np.isfinite(revenue) and revenue != 0 else np.nan
    ocf_conversion = ocf / net_income if np.isfinite(ocf) and np.isfinite(net_income) and net_income != 0 else np.nan

    core = [revenue, net_income, ocf, assets, liabilities, equity]
    official_coverage = 100.0 * sum(np.isfinite(v) for v in core) / len(core)
    period_kind = str(period_label or "").upper()
    cash_state = "IDX_OFFICIAL_YTD_OCF_FCF_AVAILABLE" if np.isfinite(ocf) and np.isfinite(fcf) else "IDX_OFFICIAL_YTD_OCF_AVAILABLE_FCF_MISSING" if np.isfinite(ocf) else "IDX_OFFICIAL_CASHFLOW_MISSING"
    if period_kind == "AUDIT" and np.isfinite(ocf):
        cash_state = "IDX_OFFICIAL_TTM_OCF_FCF_AVAILABLE" if np.isfinite(fcf) else "IDX_OFFICIAL_TTM_OCF_AVAILABLE_FCF_MISSING"

    return {
        "ticker": symbol,
        "idx_official_period": period_kind,
        "idx_official_period_end": pd.Timestamp(current_end).date().isoformat(),
        "idx_official_source_url": source_url,
        "idx_official_source_verified": True,
        "idx_official_source_quality_pct": 100.0,
        "idx_official_coverage_pct": round(official_coverage, 1),
        "idx_official_revenue": revenue,
        "idx_official_prior_revenue": prior_revenue,
        "idx_official_net_income": net_income,
        "idx_official_prior_net_income": prior_net_income,
        "idx_official_operating_income": operating_income,
        "idx_official_ocf": ocf,
        "idx_official_capex_proxy": capex_outflow,
        "idx_official_fcf_proxy": fcf,
        "idx_official_assets": assets,
        "idx_official_liabilities": liabilities,
        "idx_official_equity": equity,
        "idx_official_current_assets": current_assets,
        "idx_official_current_liabilities": current_liabilities,
        "idx_official_cash": cash,
        "idx_official_interest_bearing_debt_proxy": debt,
        "idx_official_revenue_growth_yoy_pct": round(revenue_yoy, 2) if np.isfinite(revenue_yoy) else np.nan,
        "idx_official_earnings_growth_yoy_pct": round(earnings_yoy, 2) if np.isfinite(earnings_yoy) else np.nan,
        "idx_official_net_margin_pct": round(margin, 2) if np.isfinite(margin) else np.nan,
        "idx_official_operating_margin_pct": round(operating_margin, 2) if np.isfinite(operating_margin) else np.nan,
        "idx_official_ocf_conversion_ratio": round(ocf_conversion, 3) if np.isfinite(ocf_conversion) else np.nan,
        "idx_official_interest_bearing_debt_to_equity": round(ide, 3) if np.isfinite(ide) else np.nan,
        "idx_official_total_liabilities_to_equity": round(lte, 3) if np.isfinite(lte) else np.nan,
        "idx_official_net_debt_to_equity": round(net_debt_to_equity, 3) if np.isfinite(net_debt_to_equity) else np.nan,
        "idx_official_current_ratio": round(current_ratio, 3) if np.isfinite(current_ratio) else np.nan,
        "idx_official_cash_to_debt_ratio": round(cash_to_debt, 3) if np.isfinite(cash_to_debt) else np.nan,
        "idx_official_cashflow_state": cash_state,
        "idx_official_provenance_state": "IDX_OFFICIAL_XBRL_INSTANCE",
    }


def fetch_idx_official_fundamental(
    ticker: str,
    *,
    now: Any = None,
    timeout: int = 12,
    retries: int = 2,
    max_candidates: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    symbol = normalize_ticker(ticker)
    errors: list[str] = []
    for year, period in candidate_reporting_periods(now, max_candidates=max_candidates):
        url = idx_instance_url(symbol, year, period)
        for attempt in range(max(1, int(retries))):
            try:
                response = requests.get(
                    url,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "application/zip,application/octet-stream,*/*",
                        "Accept-Language": "id-ID,id;q=0.9,en;q=0.7",
                        "Referer": IDX_FINANCIAL_PORTAL,
                    },
                    timeout=timeout,
                )
                if response.status_code in {403, 404}:
                    errors.append(f"{year}-{period}:{response.status_code}")
                    break
                if response.status_code == 429 and attempt + 1 < retries:
                    time.sleep(min(5.0, 0.8 * (2 ** attempt) + random.uniform(0.1, 0.4)))
                    continue
                response.raise_for_status()
                xml_bytes = _extract_instance_bytes(response.content)
                snapshot = parse_idx_xbrl_instance(symbol, xml_bytes, source_url=url, period_label=period)
                coverage = _finite(snapshot.get("idx_official_coverage_pct"), 0)
                status = "OK" if coverage >= 50 else "PARTIAL"
                return snapshot, {
                    "ticker": symbol,
                    "provider": "IDX_OFFICIAL_XBRL",
                    "status": status,
                    "items": 1,
                    "detail": f"period={year}-{period}; coverage={coverage:.1f}; source=IDX instance.zip",
                    "source_url": url,
                    "source_verified": True,
                }
            except Exception as exc:
                errors.append(f"{year}-{period}:{type(exc).__name__}")
                retryable = any(token in str(exc) for token in ("429", "500", "502", "503", "504", "timed out", "Timeout"))
                if attempt + 1 < retries and retryable:
                    time.sleep(min(5.0, 0.8 * (2 ** attempt) + random.uniform(0.1, 0.4)))
                    continue
                break
    return {
        "ticker": symbol,
        "idx_official_source_verified": False,
        "idx_official_coverage_pct": 0.0,
        "idx_official_provenance_state": "IDX_OFFICIAL_XBRL_NOT_AVAILABLE",
    }, {
        "ticker": symbol,
        "provider": "IDX_OFFICIAL_XBRL",
        "status": "NO_ITEMS",
        "items": 0,
        "detail": ";".join(errors[-6:]) or "No current official IDX XBRL instance found.",
        "source_verified": False,
    }


def fetch_many_idx_official_fundamentals(tickers: Iterable[str], *, now: Any = None, max_workers: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = list(dict.fromkeys(normalize_ticker(t) for t in tickers if normalize_ticker(t)))
    snapshots: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    # Keep concurrency deliberately low: IDX is an official public source, not a bulk market-data API.
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 2))) as executor:
        futures = {executor.submit(fetch_idx_official_fundamental, symbol, now=now): symbol for symbol in symbols}
        for future in as_completed(futures):
            snapshot, audit = future.result()
            snapshots.append(snapshot)
            audits.append(audit)
    return pd.DataFrame(snapshots), pd.DataFrame(audits)


__all__ = [
    "IDX_FINANCIAL_PORTAL", "candidate_reporting_periods", "idx_instance_url",
    "parse_idx_xbrl_instance", "fetch_idx_official_fundamental", "fetch_many_idx_official_fundamentals",
]
