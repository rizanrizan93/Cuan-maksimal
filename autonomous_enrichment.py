from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Mapping
import math
import re
import time
import random
import threading

import numpy as np
import pandas as pd
import requests
from date_utils import parse_public_date
from bs4 import BeautifulSoup

from data_providers import USER_AGENT, bare_ticker, normalize_ticker

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

KSEI_PROFILE_URL = "https://web.ksei.co.id/services/registered-securities/shares/lc/{ticker}?setLocale=en-US"
_AUTONOMOUS_RATE_LOCK = threading.Lock()
_AUTONOMOUS_LAST_REQUEST_AT = 0.0
_AUTONOMOUS_MIN_INTERVAL_SECONDS = 0.16


def _pace_autonomous_request() -> None:
    global _AUTONOMOUS_LAST_REQUEST_AT
    with _AUTONOMOUS_RATE_LOCK:
        now = time.monotonic()
        wait = _AUTONOMOUS_MIN_INTERVAL_SECONDS - (now - _AUTONOMOUS_LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _AUTONOMOUS_LAST_REQUEST_AT = time.monotonic()


def _finite(value: Any, default: float = np.nan) -> float:
    if isinstance(value, pd.Series):
        clean = value.dropna()
        if len(clean) != 1:
            return default
        value = clean.iloc[0]
    elif isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if len(flat) != 1:
            return default
        value = flat[0]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except Exception:
        pass
    return False


def _clean_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat", "<na>"} else text


def _verified_flag(value: Any, *, verified: bool, false_state: str = "FALSE_VERIFIED") -> tuple[Any, str]:
    if not verified or _is_missing(value) or not _clean_text(value):
        return np.nan, "UNKNOWN_NOT_VERIFIED"
    normalized = _clean_text(value).lower()
    if normalized in {"1", "true", "yes", "y", "on", "verified", "active", "aktif"}:
        return True, "TRUE_VERIFIED"
    if normalized in {"0", "false", "no", "n", "off", "clear", "none", "not listed"}:
        return False, false_state
    return np.nan, "UNKNOWN_NOT_VERIFIED"


def _clip(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    number = _finite(value, low)
    return float(min(high, max(low, number)))


def _number(text: Any) -> float:
    value = str(text or "").replace("%", "").replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else np.nan


def _first_date(text: Any) -> pd.Timestamp | pd.NaT:
    return parse_public_date(text)


def build_broker_inventory_proxy(features: Mapping[str, Any]) -> dict[str, Any]:
    """OHLCV-only behavioural proxy. It never identifies a broker or beneficial owner.

    v1.7 hardening adds a multi-horizon inventory-cycle proxy so the fallback does not
    overfit the most recent 20 bars. Direct broker data, when supplied, remains superior.
    """
    smart = _finite(features.get("smart_money_score"), np.nan)
    accumulation = _finite(features.get("accumulation_days20"), 0)
    absorption = _finite(features.get("absorption_days20"), 0)
    distribution = _finite(features.get("distribution_days20"), 0)
    failed_absorption = _finite(features.get("failed_absorption_days20"), 0)
    close_acceptance = _finite(features.get("close_acceptance20_pct"), np.nan)
    up_value = _finite(features.get("up_value_ratio20_pct"), np.nan)
    cmf = _finite(features.get("cmf20"), np.nan)
    obv_slope = _finite(features.get("obv_slope20_pct"), np.nan)
    contraction = _finite(features.get("pullback_volume_contraction_score"), np.nan)
    volume_ratio = _finite(features.get("volume_ratio20"), np.nan)
    low20 = _finite(features.get("low20"), np.nan)
    ema20 = _finite(features.get("ema20"), np.nan)
    inventory_cycle = _finite(features.get("inventory_cycle_score"), np.nan)
    inventory_cycle_coverage = _finite(features.get("inventory_cycle_coverage_pct"), np.nan)
    inventory_dryness_multi = _finite(features.get("inventory_dryness_multiyear_score"), np.nan)
    inventory_shift_recent_vs_long = _finite(features.get("inventory_shift_recent_vs_long_score"), np.nan)
    inventory_cycle_phase = _clean_text(features.get("inventory_cycle_phase")) or "UNKNOWN_MULTIYEAR_PROXY"
    history_bars = _finite(features.get("history_bars"), np.nan)

    persistence = _clip(35 + 7.0 * accumulation + 5.0 * absorption - 8.0 * distribution - 5.0 * failed_absorption)
    pressure = _clip(
        0.26 * _finite(smart, 50)
        + 0.18 * _finite(close_acceptance, 50)
        + 0.14 * _finite(up_value, 50)
        + 0.12 * _clip(50 + 250 * _finite(cmf, 0))
        + 0.10 * _clip(50 + 4 * _finite(obv_slope, 0))
        + 0.08 * _finite(contraction, 50)
        + 0.12 * _finite(inventory_cycle, 50)
    )
    dryness = _clip(
        0.30 * persistence
        + 0.20 * _finite(contraction, 50)
        + 0.15 * _finite(close_acceptance, 50)
        + 0.10 * _clip(70 - 12 * max(0, _finite(volume_ratio, 1) - 1))
        + 0.25 * _finite(inventory_dryness_multi, 50)
    )
    score = _clip(0.35 * pressure + 0.25 * persistence + 0.20 * dryness + 0.20 * _finite(inventory_cycle, 50))

    if distribution >= 4 or failed_absorption >= 4 or inventory_cycle_phase == "MULTIYEAR_INVENTORY_RELEASE_PROXY":
        shift = "DISTRIBUTION_OR_INVENTORY_RELEASE_RISK_PROXY"
    elif inventory_cycle_phase == "MULTIYEAR_COLLECTION_PERSISTING_PROXY" and score >= 60:
        shift = "MULTIYEAR_COLLECTION_PERSISTING_PROXY"
    elif inventory_cycle_phase == "BOTTOMING_TO_COLLECTION_PROXY" and score >= 54:
        shift = "BOTTOMING_TO_COLLECTION_PROXY"
    elif score >= 65 and persistence >= 60:
        shift = "COLLECTION_PERSISTING_PROXY"
    elif score >= 55:
        shift = "BOTTOMING_OR_EARLY_COLLECTION_PROXY"
    else:
        shift = "NO_CLEAR_INVENTORY_PROXY"

    defended = np.nan
    if np.isfinite(low20) and np.isfinite(ema20):
        defended = max(low20, min(ema20, _finite(features.get("last_price"), ema20)))

    coverage_fields = [smart, close_acceptance, up_value, cmf, obv_slope, contraction, volume_ratio, inventory_cycle, inventory_dryness_multi]
    raw_coverage = 100 * sum(np.isfinite(v) for v in coverage_fields) / len(coverage_fields)
    # OHLCV evidence receives a deliberate confidence haircut because it is not broker identity.
    proxy_coverage = min(72.0, raw_coverage * 0.72)
    proxy_years = min(3.0, history_bars / 252.0) if np.isfinite(history_bars) and history_bars > 0 else np.nan

    return {
        "broker_summary_score": round(pressure, 1),
        "broker_summary_coverage_pct": round(proxy_coverage, 1),
        "broker_net_ratio": np.nan,
        "broker_inventory_score": round(score, 1),
        "broker_inventory_coverage_pct": round(min(72.0, max(proxy_coverage, 0.72 * _finite(inventory_cycle_coverage, 0))), 1),
        "inventory_coverage_years": round(proxy_years, 2) if np.isfinite(proxy_years) else np.nan,
        "inventory_proxy_lookback_years": round(proxy_years, 2) if np.isfinite(proxy_years) else np.nan,
        "holder_persistence_score": round(persistence, 1),
        "inventory_dryness_score": round(dryness, 1),
        "inventory_cycle_score": round(inventory_cycle, 1) if np.isfinite(inventory_cycle) else np.nan,
        "inventory_cycle_coverage_pct": round(inventory_cycle_coverage, 1) if np.isfinite(inventory_cycle_coverage) else np.nan,
        "inventory_cycle_phase": inventory_cycle_phase,
        "inventory_shift_recent_vs_long_score": round(inventory_shift_recent_vs_long, 1) if np.isfinite(inventory_shift_recent_vs_long) else np.nan,
        "inventory_dryness_multiyear_score": round(inventory_dryness_multi, 1) if np.isfinite(inventory_dryness_multi) else np.nan,
        "retail_exit_score": np.nan,
        "retail_cannibalisation_risk": np.nan,
        "fund_like_flow_score": np.nan,
        "jumbo_crossing_score": np.nan,
        "defended_level": round(defended, 4) if np.isfinite(defended) else np.nan,
        "defended_level_score": round(_clip(0.55 * persistence + 0.30 * pressure + 0.15 * _finite(inventory_cycle, 50)), 1),
        "broker_inventory_shift_state": shift,
        "broker_summary_provenance_state": "MULTIHORIZON_OHLCV_BEHAVIOURAL_PROXY_NOT_BROKER_DATA",
        "beneficial_owner_inference_state": "NOT_INFERRED_FROM_OHLCV_OR_BROKER_CODE",
        "broker_inventory_evidence_type": "MULTIHORIZON_OHLCV_PROXY",
    }


def build_orderbook_proxy(features: Mapping[str, Any]) -> dict[str, Any]:
    """EOD microstructure proxy; not live bid/offer or market depth."""
    acceptance = _finite(features.get("close_acceptance20_pct"), np.nan)
    absorption = _finite(features.get("absorption_score"), np.nan)
    volume_ratio = _finite(features.get("volume_ratio20"), np.nan)
    up_value = _finite(features.get("up_value_ratio20_pct"), np.nan)
    friction = _finite(features.get("execution_friction_score"), np.nan)
    gap_risk = _finite(features.get("gap_risk_score"), np.nan)
    high20 = _finite(features.get("high20"), np.nan)
    last = _finite(features.get("last_price"), np.nan)
    breakout_proximity = _clip(100 - 500 * abs(last / high20 - 1)) if np.isfinite(last) and np.isfinite(high20) and high20 > 0 else np.nan
    volume_confirmation = _clip(35 + 35 * _finite(volume_ratio, 1)) if np.isfinite(volume_ratio) else np.nan
    score_values = [acceptance, absorption, up_value, breakout_proximity, volume_confirmation]
    available = [v for v in score_values if np.isfinite(v)]
    score = _clip(
        0.25 * _finite(acceptance, 50)
        + 0.25 * _finite(absorption, 50)
        + 0.18 * _finite(up_value, 50)
        + 0.17 * _finite(breakout_proximity, 50)
        + 0.15 * _finite(volume_confirmation, 50)
        - 0.10 * max(0, _finite(friction, 45) - 45)
        - 0.08 * max(0, _finite(gap_risk, 40) - 40)
    )
    coverage = 100 * len(available) / len(score_values)
    return {
        "orderbook_trigger_score": round(score, 1),
        "orderbook_coverage_pct": round(coverage * 0.60, 1),
        "precise_trigger_price": round(high20, 4) if np.isfinite(high20) else np.nan,
        "retail_offer_stack_score": np.nan,
        "offer_absorption_speed_score": round(_finite(absorption, np.nan), 1) if np.isfinite(absorption) else np.nan,
        "orderbook_provenance_state": "OHLCV_EOD_MICROSTRUCTURE_PROXY_NOT_LIVE_DEPTH",
        "orderbook_observed_at": str(features.get("last_date") or ""),
        "orderbook_note": "Proxy from EOD acceptance, absorption, volume and breakout proximity; no live bid/offer feed.",
        "orderbook_evidence_type": "OHLCV_EOD_PROXY",
    }


def _label_value(text: str, label: str) -> str:
    pattern = rf"(?:^|\n)\s*{re.escape(label)}\s*\n\s*([^\n]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_ksei_profile_html(ticker: str, html: str, *, source_url: str = "") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    symbol = normalize_ticker(ticker)
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text("\n", strip=True)
    profile = {
        "ticker": symbol,
        "company_name": "",
        "sector": "",
        "listing_date": "",
        "security_status": "",
        "total_shares": np.nan,
        "registered_amount": np.nan,
        "scripless_pct": np.nan,
        "local_pct": np.nan,
        "foreign_pct": np.nan,
        "ksei_source_url": source_url,
        "ksei_source_verified": bool(text),
    }
    # Prefer labelled issuer fields. The first heading on KSEI pages can be a navigation heading such as "Services".
    issuer_name = _label_value(text, "Issuer")
    security_name = _label_value(text, "Security name")
    heading_candidates = [
        heading.get_text(" ", strip=True)
        for heading in soup.find_all(["h1", "h2"])
        if heading.get_text(" ", strip=True)
        and heading.get_text(" ", strip=True).lower() not in {"services", "hot links", "graph", "price history", "corporate action"}
    ]
    profile["company_name"] = issuer_name or security_name or (heading_candidates[0] if heading_candidates else "")
    profile["sector"] = _label_value(text, "Activity Sector")
    profile["listing_date"] = _label_value(text, "Listing Date")
    profile["security_status"] = _label_value(text, "Status")
    profile["registered_amount"] = _number(_label_value(text, "Current Amount"))
    profile["total_shares"] = _number(_label_value(text, "Number of Securities"))
    profile["local_pct"] = _number(_label_value(text, "Local Percentage"))
    profile["foreign_pct"] = _number(_label_value(text, "Foreign Percentage"))
    scripless_match = re.search(r"As of\s+[^\n]+\s*\n\s*([\d.,]+)%\s+Scripless", text, flags=re.IGNORECASE)
    if scripless_match:
        profile["scripless_pct"] = _number(scripless_match.group(1))

    actions: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        table_rows = table.find_all("tr")
        if not table_rows:
            continue
        header_cells = table_rows[0].find_all(["th", "td"])
        headers = [cell.get_text(" ", strip=True).lower() for cell in header_cells]
        if not headers or not any("type of ca" in header or "corporate action" in header for header in headers):
            continue
        for row in table_rows[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if not cells:
                continue
            values = dict(zip(headers, cells))
            action_type = values.get("type of ca") or values.get("corporate action") or cells[0]
            if not action_type or action_type.strip().lower() in {"type of ca", "corporate action"}:
                continue
            actions.append({
                "ticker": symbol,
                "action_type": action_type,
                "ratio": values.get("ratio", ""),
                "cum_date": values.get("cum date", ""),
                "record_date": values.get("record date", ""),
                "distribution_date": values.get("distribution date", ""),
                "status": values.get("status", ""),
                "source_url": source_url,
                "source_verified": True,
            })
    return profile, actions


def fetch_ksei_profile(ticker: str, timeout: int = 18, retries: int = 2) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    symbol = normalize_ticker(ticker)
    url = KSEI_PROFILE_URL.format(ticker=bare_ticker(symbol))
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            _pace_autonomous_request()
            response = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"}, timeout=timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After") if hasattr(response, "headers") else None
                if retry_after:
                    try:
                        time.sleep(min(12.0, max(0.0, float(retry_after))))
                    except (TypeError, ValueError):
                        pass
            response.raise_for_status()
            profile, actions = parse_ksei_profile_html(symbol, response.text, source_url=url)
            ok = bool(profile.get("company_name") or profile.get("sector") or np.isfinite(_finite(profile.get("total_shares"))))
            audit = {"ticker": symbol, "provider": "KSEI_SECURITY_PROFILE", "status": "OK" if ok else "PARSE_EMPTY", "items": 1 if ok else 0, "detail": f"corporate_actions={len(actions)}; attempt={attempt + 1}"}
            return profile, actions, audit
        except Exception as exc:
            last_error = exc
            retryable = any(token in str(exc) for token in ("429", "500", "502", "503", "504", "timed out", "Timeout"))
            if attempt + 1 >= retries or not retryable:
                break
            time.sleep(min(5.0, 0.9 * (2 ** attempt) + random.uniform(0.1, 0.5)))
    exc = last_error or RuntimeError("unknown KSEI error")
    return {"ticker": symbol, "ksei_source_url": url, "ksei_source_verified": False}, [], {"ticker": symbol, "provider": "KSEI_SECURITY_PROFILE", "status": "ERROR", "items": 0, "detail": f"{type(exc).__name__}: {exc}"}


def fetch_many_ksei_profiles(tickers: Iterable[str], max_workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbols = list(dict.fromkeys(normalize_ticker(t) for t in tickers if normalize_ticker(t)))
    profiles: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 4))) as executor:
        futures = {executor.submit(fetch_ksei_profile, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            profile, local_actions, audit = future.result()
            profiles.append(profile)
            actions.extend(local_actions)
            audits.append(audit)
            time.sleep(0.02)
    profile_frame = pd.DataFrame(profiles)
    action_frame = pd.DataFrame(actions)
    audit_frame = pd.DataFrame(audits)
    return profile_frame, action_frame, audit_frame


def ksei_actions_to_events(actions: pd.DataFrame, as_of: Any = None, lookback_days: int = 730) -> pd.DataFrame:
    columns = [
        "ticker", "published_at", "title", "summary", "publisher", "url", "source_tier",
        "collection_provider", "source_verified", "category", "event_role", "narrative_eligible",
    ]
    if actions is None or actions.empty:
        return pd.DataFrame(columns=columns)
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    if now.tzinfo is None:
        now = now.tz_localize("Asia/Jakarta")
    rows: list[dict[str, Any]] = []
    for _, row in actions.iterrows():
        dates = [_first_date(row.get(key)) for key in ("record_date", "distribution_date", "cum_date")]
        event_date = next((date for date in dates if pd.notna(date)), pd.NaT)
        if pd.notna(event_date):
            localized = pd.Timestamp(event_date).tz_localize("Asia/Jakarta") if pd.Timestamp(event_date).tzinfo is None else pd.Timestamp(event_date).tz_convert("Asia/Jakarta")
            if abs((now - localized).days) > lookback_days:
                continue
            published_at = localized.tz_convert("UTC")
        else:
            published_at = now.tz_convert("UTC")
        action_type = _clean_text(row.get("action_type")) or "Corporate Action"
        summary = "; ".join(
            part for part in [
                _clean_text(row.get("ratio")),
                f"status={_clean_text(row.get('status'))}" if _clean_text(row.get("status")) else "",
            ] if part
        )
        rows.append({
            "ticker": normalize_ticker(row.get("ticker")),
            "published_at": published_at,
            "title": f"KSEI corporate action: {action_type}",
            "summary": summary,
            "publisher": "KSEI",
            "url": _clean_text(row.get("source_url")),
            "source_tier": "OFFICIAL",
            "collection_provider": "KSEI_SECURITY_PROFILE",
            "source_verified": True,
            "category": "CORPORATE_ACTION",
            # A bare KSEI action row is integrity/administrative evidence, not a narrative thesis.
            "event_role": "ADMINISTRATIVE_CORPORATE_ACTION",
            "narrative_eligible": False,
        })
    return pd.DataFrame(rows, columns=columns)

def ksei_profiles_to_maps(profiles: pd.DataFrame, actions: pd.DataFrame, as_of: Any = None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    ownership: dict[str, dict[str, Any]] = {}
    integrity: dict[str, dict[str, Any]] = {}
    if profiles is None or profiles.empty:
        return ownership, integrity
    action_groups = {ticker: group for ticker, group in actions.groupby("ticker")} if actions is not None and not actions.empty else {}
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    if now.tzinfo is None:
        now = now.tz_localize("Asia/Jakarta")
    else:
        now = now.tz_convert("Asia/Jakarta")

    for _, row in profiles.iterrows():
        ticker = normalize_ticker(row.get("ticker"))
        verified = bool(row.get("ksei_source_verified", False)) and bool(_clean_text(row.get("ksei_source_url")))
        total_shares = _finite(row.get("total_shares"), np.nan)
        local_pct = _finite(row.get("local_pct"), np.nan)
        foreign_pct = _finite(row.get("foreign_pct"), np.nan)
        scripless_pct = _finite(row.get("scripless_pct"), np.nan)
        ownership_fields = [total_shares, local_pct, foreign_pct, scripless_pct]
        ownership_coverage = 45 * sum(np.isfinite(v) for v in ownership_fields) / len(ownership_fields) if verified else 0
        ownership_score = _clip(50 + 0.15 * (_finite(scripless_pct, 50) - 50)) if verified and np.isfinite(scripless_pct) else np.nan
        ownership[ticker] = {
            "ownership_score": round(ownership_score, 1) if np.isfinite(ownership_score) else np.nan,
            "ownership_coverage_pct": round(ownership_coverage, 1),
            "reported_free_float_pct": np.nan,
            "effective_free_float_pct": np.nan,
            "fake_float_gap_pct": np.nan,
            "ownership_network_score": round(ownership_score, 1) if np.isfinite(ownership_score) else np.nan,
            "buyback_inventory_pct": np.nan,
            "passive_flow_risk_score": np.nan,
            "ownership_provenance_state": "KSEI_REGISTRATION_PROXY_NOT_FREE_FLOAT" if verified else "PROVIDER_FAILED",
            "ownership_note": "KSEI registration/local-foreign composition is not effective free float or beneficial ownership.",
            "total_shares_ksei": round(total_shares, 0) if np.isfinite(total_shares) else np.nan,
            "scripless_pct_ksei": round(scripless_pct, 2) if np.isfinite(scripless_pct) else np.nan,
            "local_pct_ksei": round(local_pct, 2) if np.isfinite(local_pct) else np.nan,
            "foreign_pct_ksei": round(foreign_pct, 2) if np.isfinite(foreign_pct) else np.nan,
        }

        group = action_groups.get(ticker, pd.DataFrame())
        material_keywords = r"RIGHT|STOCK SPLIT|REVERSE|MERGER|CONVERSION|WARRANT|BONUS|STOCK DIVIDEND|BUYBACK|TENDER"
        material = group[group.get("action_type", pd.Series(dtype=str)).astype(str).str.upper().str.contains(material_keywords, regex=True, na=False)] if not group.empty else pd.DataFrame()
        active_material = material[material.get("status", pd.Series("", index=material.index)).astype(str).str.upper().ne("CANCELLED")].copy() if not material.empty else pd.DataFrame()
        if not active_material.empty:
            action_dates = []
            for _, action_row in active_material.iterrows():
                parsed_dates = [_first_date(action_row.get(key)) for key in ("distribution_date", "record_date", "cum_date")]
                valid_dates = [pd.Timestamp(value) for value in parsed_dates if pd.notna(value)]
                action_dates.append(max(valid_dates) if valid_dates else pd.NaT)
            active_material["_event_date"] = action_dates
            now_naive = now.tz_localize(None)
            age_days = (now_naive.normalize() - pd.to_datetime(active_material["_event_date"], errors="coerce").dt.normalize()).dt.days
            # Unknown dates are review-only; old historical actions do not remain permanently active.
            active_material = active_material[(age_days.isna()) | ((age_days >= -365) & (age_days <= 180))]

        security_status = _clean_text(row.get("security_status")).upper()
        status_known = bool(verified and security_status)
        status_hard_block = bool(status_known and security_status not in {"ACTIVE", "AKTIF"})
        suspension_flag: Any = True if status_hard_block else False if status_known else np.nan
        suspension_state = "TRUE_VERIFIED_KSEI_NON_ACTIVE" if status_hard_block else "FALSE_VERIFIED_KSEI_ACTIVE" if status_known else "UNKNOWN_PROVIDER_ERROR"

        hard_reasons = ["KSEI_SECURITY_NOT_ACTIVE"] if status_hard_block else []
        cautions = ["RECENT_OR_ACTIVE_MATERIAL_CORPORATE_ACTION"] if verified and not active_material.empty else []
        if not verified:
            state = "AUTO_PUBLIC_PROVIDER_ERROR"
            score = np.nan
            coverage = 0.0
        elif status_hard_block:
            state = "AUTO_PUBLIC_PROXY_HARD_BLOCK"
            score = 5.0
            coverage = 42.9
        elif cautions:
            state = "AUTO_PUBLIC_PROXY_PARTIAL_CAUTION"
            score = 72.0
            coverage = 42.9
        else:
            state = "AUTO_PUBLIC_PROXY_PARTIAL"
            score = 88.0
            coverage = 42.9

        unknown_critical = 5 + (0 if status_known else 1)  # board, HSC, FCA/special, UMA/sanctions, free float, suspension if provider failed
        integrity[ticker] = {
            "idx_integrity_score": round(score, 1) if np.isfinite(score) else np.nan,
            "idx_integrity_coverage_pct": round(coverage, 1),
            "idx_integrity_state": state,
            "idx_integrity_hard_block": status_hard_block,
            "idx_integrity_block_reasons": " | ".join(hard_reasons) or "NONE",
            "idx_integrity_caution_flags": " | ".join(cautions) or "CRITICAL_IDX_FIELDS_UNKNOWN_NOT_VERIFIED",
            "idx_integrity_provenance_state": "AUTO_PUBLIC_KSEI_PARTIAL_PROXY" if verified else "PROVIDER_FAILED",
            "idx_integrity_observed_at": now.isoformat() if verified else "",
            "idx_integrity_age_days": 0.0 if verified else np.nan,
            "listing_board": "UNKNOWN",
            "listing_board_verification_state": "UNKNOWN_NOT_VERIFIED",
            "hsc_flag": np.nan,
            "hsc_verification_state": "UNKNOWN_NOT_VERIFIED",
            "special_monitoring_flag": np.nan,
            "special_monitoring_verification_state": "UNKNOWN_NOT_VERIFIED",
            "full_call_auction_flag": np.nan,
            "full_call_auction_verification_state": "UNKNOWN_NOT_VERIFIED",
            "suspension_flag": suspension_flag,
            "suspension_verification_state": suspension_state,
            "uma_flag": np.nan,
            "uma_verification_state": "UNKNOWN_NOT_VERIFIED",
            "sanctions_flag": np.nan,
            "sanctions_verification_state": "UNKNOWN_NOT_VERIFIED",
            "regulatory_free_float_pct": np.nan,
            "regulatory_free_float_verification_state": "UNKNOWN_NOT_VERIFIED",
            "over_1pct_disclosure_flag": np.nan,
            "over_1pct_disclosure_verification_state": "UNKNOWN_NOT_VERIFIED",
            "idx_integrity_unknown_critical_count": unknown_critical,
            "corporate_action_flag": bool(verified and not active_material.empty),
            "corporate_action_type": " | ".join(active_material["action_type"].astype(str).head(3).tolist()) if verified and not active_material.empty else "",
            "corporate_action_effective_date": "",
            "corporate_action_review_cleared": bool(verified and not status_hard_block and active_material.empty),
            "idx_integrity_source_url": _clean_text(row.get("ksei_source_url")),
            "idx_integrity_note": (
                "Automatic KSEI partial proxy. Provider failure is UNKNOWN, never suspension. "
                "Board/HSC/FCA/UMA/sanctions/free-float require direct official verification."
            ),
            "security_status_ksei": security_status or "UNKNOWN",
            "ksei_provider_state": "OK" if verified else "PROVIDER_ERROR",
        }
    return ownership, integrity



def apply_regulatory_event_overlay(
    integrity_map: Mapping[str, Mapping[str, Any]],
    events: pd.DataFrame,
    *,
    as_of: Any = None,
    max_age_days: int = 120,
) -> dict[str, dict[str, Any]]:
    """Add a conservative regulatory alert overlay from collected public events.

    Direct regulator URLs/verified records can hard-block. Media-only mentions create caution,
    never a false regulatory clearance. Absence of an event is not proof that no alert exists.
    """
    output = {normalize_ticker(ticker): dict(payload) for ticker, payload in (integrity_map or {}).items()}
    if events is None or events.empty:
        return output
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    if now.tzinfo is None:
        now = now.tz_localize("Asia/Jakarta")
    else:
        now = now.tz_convert("Asia/Jakarta")
    local = events.copy()
    if "ticker" not in local.columns:
        return output
    local["ticker"] = local["ticker"].map(normalize_ticker)
    local["published_at"] = pd.to_datetime(local.get("published_at"), errors="coerce", utc=True)
    local = local[local["published_at"].isna() | ((now.tz_convert("UTC") - local["published_at"]).dt.days.between(0, max_age_days))]
    for ticker, group in local.groupby("ticker"):
        base = dict(output.get(ticker, {}))
        hard_reasons = [reason for reason in str(base.get("idx_integrity_block_reasons") or "").split(" | ") if reason and reason != "NONE"]
        cautions = [reason for reason in str(base.get("idx_integrity_caution_flags") or "").split(" | ") if reason and reason != "NONE"]
        official_alert = False
        media_alert = False
        for _, row in group.iterrows():
            text = " ".join(str(row.get(key) or "") for key in ("title", "summary", "publisher", "url")).lower()
            url = str(row.get("url") or "").lower()
            publisher = str(row.get("publisher") or "").lower()
            verified = str(row.get("source_verified", False)).strip().lower() in {"1", "true", "yes", "verified"}
            regulator = verified or any(domain in url for domain in ("idx.co.id", "ojk.go.id", "ksei.co.id")) or publisher in {"bursa efek indonesia", "indonesia stock exchange", "ojk", "ksei"}
            suspension = bool(re.search(r"\bsuspensi|suspension|suspended", text))
            special = bool(re.search(r"pemantauan khusus|special monitoring|full call auction|\bfca\b", text))
            hsc = bool(re.search(r"high shareholding concentration|konsentrasi kepemilikan tinggi|\bhsc\b", text))
            sanction = bool(re.search(r"sanksi|sanction|pelanggaran pasar modal|market manipulation", text))
            uma = bool(re.search(r"unusual market activity|aktivitas pasar tidak biasa|\buma\b", text))
            if not any((suspension, special, hsc, sanction, uma)):
                continue
            media_alert = True
            official_alert = official_alert or regulator
            if regulator and suspension:
                hard_reasons.append("OFFICIAL_SUSPENSION_ALERT")
                base["suspension_flag"] = True
                base["suspension_verification_state"] = "TRUE_VERIFIED_OFFICIAL_ALERT"
            if regulator and special:
                hard_reasons.append("OFFICIAL_SPECIAL_MONITORING_OR_FCA_ALERT")
                base["special_monitoring_flag"] = True
                base["special_monitoring_verification_state"] = "TRUE_VERIFIED_OFFICIAL_ALERT"
                base["full_call_auction_flag"] = True
                base["full_call_auction_verification_state"] = "TRUE_VERIFIED_OFFICIAL_ALERT"
            if regulator and hsc:
                hard_reasons.append("OFFICIAL_HSC_ALERT")
                base["hsc_flag"] = True
                base["hsc_verification_state"] = "TRUE_VERIFIED_OFFICIAL_ALERT"
            if regulator and sanction:
                hard_reasons.append("OFFICIAL_REGULATORY_SANCTION_ALERT")
                base["sanctions_flag"] = True
                base["sanctions_verification_state"] = "TRUE_VERIFIED_OFFICIAL_ALERT"
            if uma:
                cautions.append("OFFICIAL_UMA_ALERT" if regulator else "MEDIA_UMA_ALERT_REVIEW")
                if regulator:
                    base["uma_flag"] = True
                    base["uma_verification_state"] = "TRUE_VERIFIED_OFFICIAL_ALERT"
                else:
                    base["uma_flag"] = np.nan
                    base["uma_verification_state"] = "MEDIA_ALERT_UNVERIFIED"
            if not regulator:
                cautions.append("MEDIA_REGULATORY_ALERT_REQUIRES_OFFICIAL_CONFIRMATION")
        if not media_alert:
            output[ticker] = base
            continue
        hard_reasons = list(dict.fromkeys(hard_reasons))
        cautions = list(dict.fromkeys(cautions))
        hard_block = bool(hard_reasons)
        verification_keys = (
            "hsc_verification_state", "special_monitoring_verification_state",
            "full_call_auction_verification_state", "suspension_verification_state",
            "uma_verification_state", "sanctions_verification_state",
        )
        unknown_count = sum(str(base.get(key) or "UNKNOWN_NOT_VERIFIED") == "UNKNOWN_NOT_VERIFIED" for key in verification_keys)
        unknown_count += int(str(base.get("listing_board_verification_state") or "UNKNOWN_NOT_VERIFIED") == "UNKNOWN_NOT_VERIFIED")
        unknown_count += int(str(base.get("regulatory_free_float_verification_state") or "UNKNOWN_NOT_VERIFIED") == "UNKNOWN_NOT_VERIFIED")
        base.update({
            "idx_integrity_score": 5.0 if hard_block else min(_finite(base.get("idx_integrity_score"), 72.0), 72.0),
            "idx_integrity_coverage_pct": max(_finite(base.get("idx_integrity_coverage_pct"), 0.0), 68.0 if official_alert else 60.0),
            "idx_integrity_state": "AUTO_PUBLIC_REGULATORY_HARD_BLOCK" if hard_block else "AUTO_PUBLIC_REGULATORY_CAUTION",
            "idx_integrity_hard_block": hard_block or bool(base.get("idx_integrity_hard_block", False)),
            "idx_integrity_block_reasons": " | ".join(hard_reasons) or "NONE",
            "idx_integrity_caution_flags": " | ".join(cautions) or "NONE",
            "idx_integrity_provenance_state": "AUTO_PUBLIC_KSEI_AND_REGULATORY_NEWS",
            "idx_integrity_observed_at": now.isoformat(),
            "idx_integrity_unknown_critical_count": unknown_count,
            "idx_integrity_note": "KSEI base plus public regulatory-event overlay; media alerts are caution until official confirmation.",
        })
        output[ticker] = base
    return output


def _statement_value(frame: pd.DataFrame, labels: Iterable[str], column_index: int = 0) -> float:
    if frame is None or frame.empty:
        return np.nan
    # yfinance column order is not guaranteed across versions. Always evaluate latest reporting periods first.
    local = frame.copy()
    try:
        parsed_columns = pd.to_datetime(local.columns, errors="coerce")
        if pd.notna(parsed_columns).any():
            ordering = sorted(range(len(local.columns)), key=lambda index: parsed_columns[index] if pd.notna(parsed_columns[index]) else pd.Timestamp.min, reverse=True)
            local = local.iloc[:, ordering]
    except Exception:
        local = frame
    normalized = {str(idx).strip().lower().replace(" ", ""): idx for idx in local.index}
    for label in labels:
        key = label.strip().lower().replace(" ", "")
        if key in normalized:
            series = pd.to_numeric(local.loc[normalized[key]], errors="coerce")
            if isinstance(series, pd.Series):
                values = series.dropna()
                if len(values) > column_index:
                    return _finite(values.iloc[column_index], np.nan)
            return _finite(series, np.nan)
    return np.nan


def _statement_series(frame: pd.DataFrame, labels: Iterable[str]) -> pd.Series:
    """Return one statement row indexed by reporting date, newest first.

    yfinance sometimes changes column order. This helper also preserves the reporting
    dates so QoQ and YoY are never silently conflated.
    """
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    normalized = {str(idx).strip().lower().replace(" ", ""): idx for idx in frame.index}
    matched = None
    for label in labels:
        key = label.strip().lower().replace(" ", "")
        if key in normalized:
            matched = normalized[key]
            break
    if matched is None:
        return pd.Series(dtype=float)
    raw = pd.to_numeric(frame.loc[matched], errors="coerce")
    if not isinstance(raw, pd.Series):
        return pd.Series(dtype=float)
    dates = pd.to_datetime(raw.index, errors="coerce")
    local = pd.DataFrame({"date": dates, "value": raw.to_numpy()}).dropna(subset=["value"])
    if local.empty:
        return pd.Series(dtype=float)
    local["sort_date"] = local["date"].fillna(pd.Timestamp.min)
    local = local.sort_values("sort_date", ascending=False).drop_duplicates("date", keep="first")
    return pd.Series(local["value"].to_numpy(dtype=float), index=local["date"].to_list(), dtype=float)


def _same_quarter_prior_year(series: pd.Series) -> float:
    if series is None or len(series) < 2:
        return np.nan
    latest_date = series.index[0]
    if pd.isna(latest_date):
        return np.nan
    candidates: list[tuple[float, float]] = []
    for date, value in series.iloc[1:].items():
        if pd.isna(date) or not np.isfinite(_finite(value, np.nan)):
            continue
        days = (pd.Timestamp(latest_date) - pd.Timestamp(date)).days
        if 300 <= days <= 430:
            candidates.append((abs(days - 365.25), float(value)))
    if not candidates:
        return np.nan
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _sum_latest_quarters(series: pd.Series, periods: int = 4) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < periods:
        return np.nan
    return float(values.iloc[:periods].sum())


def _pct_growth(current: float, base: float) -> float:
    if not (np.isfinite(current) and np.isfinite(base)) or base == 0:
        return np.nan
    return 100.0 * (current / base - 1.0)


def _earnings_growth(current: float, base: float) -> tuple[float, str]:
    if not (np.isfinite(current) and np.isfinite(base)) or base == 0:
        return np.nan, "UNAVAILABLE"
    if base < 0 < current:
        return np.nan, "LOSS_TO_PROFIT"
    if base > 0 > current:
        return np.nan, "PROFIT_TO_LOSS"
    if base < 0 and current < 0:
        improvement = 100.0 * (current - base) / abs(base)
        return improvement, "LOSS_IMPROVEMENT" if improvement > 0 else "LOSS_DETERIORATION"
    return 100.0 * (current / base - 1.0), "NORMAL"


def _weighted_mean_available(items: Iterable[tuple[float, float]]) -> float:
    valid = [(float(value), float(weight)) for value, weight in items if np.isfinite(value) and weight > 0]
    if not valid:
        return np.nan
    weight_sum = sum(weight for _, weight in valid)
    return float(sum(value * weight for value, weight in valid) / weight_sum)


def fetch_yfinance_fundamental_snapshot(ticker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    symbol = normalize_ticker(ticker)
    if yf is None:
        return {"ticker": symbol, "fundamental_provenance_state": "UNAVAILABLE", "fundamental_coverage_pct": 0.0}, {"ticker": symbol, "provider": "YFINANCE_FUNDAMENTALS", "status": "UNAVAILABLE", "items": 0, "detail": "yfinance unavailable"}
    try:
        _pace_autonomous_request()
        obj = yf.Ticker(symbol)
        income = obj.quarterly_income_stmt
        balance = obj.quarterly_balance_sheet
        cashflow = obj.quarterly_cashflow

        revenue_series = _statement_series(income, ["Total Revenue", "Operating Revenue"])
        net_income_series = _statement_series(income, ["Net Income", "Net Income Common Stockholders"])
        operating_income_series = _statement_series(income, ["Operating Income"])
        ocf_series = _statement_series(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        capex_series = _statement_series(cashflow, ["Capital Expenditure"])

        revenue = _finite(revenue_series.iloc[0], np.nan) if len(revenue_series) else np.nan
        revenue_prev_q = _finite(revenue_series.iloc[1], np.nan) if len(revenue_series) > 1 else np.nan
        revenue_prev_y = _same_quarter_prior_year(revenue_series)
        net_income = _finite(net_income_series.iloc[0], np.nan) if len(net_income_series) else np.nan
        net_income_prev_q = _finite(net_income_series.iloc[1], np.nan) if len(net_income_series) > 1 else np.nan
        net_income_prev_y = _same_quarter_prior_year(net_income_series)
        operating_income = _finite(operating_income_series.iloc[0], np.nan) if len(operating_income_series) else np.nan
        ocf = _finite(ocf_series.iloc[0], np.nan) if len(ocf_series) else np.nan
        capex = _finite(capex_series.iloc[0], np.nan) if len(capex_series) else np.nan

        equity = _statement_value(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest"], 0)
        debt = _statement_value(balance, ["Total Debt"], 0)
        liabilities = _statement_value(balance, ["Total Liabilities Net Minority Interest", "Total Liabilities"], 0)
        assets = _statement_value(balance, ["Total Assets"], 0)
        current_assets = _statement_value(balance, ["Current Assets", "Total Current Assets"], 0)
        current_liabilities = _statement_value(balance, ["Current Liabilities", "Total Current Liabilities"], 0)
        cash = _statement_value(balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"], 0)

        revenue_growth_qoq = _pct_growth(revenue, revenue_prev_q)
        revenue_growth_yoy = _pct_growth(revenue, revenue_prev_y)
        earnings_growth_qoq, earnings_qoq_state = _earnings_growth(net_income, net_income_prev_q)
        earnings_growth_yoy, earnings_yoy_state = _earnings_growth(net_income, net_income_prev_y)

        # Backward-compatible growth fields now mean YoY when available; QoQ is an explicit fallback.
        revenue_growth = revenue_growth_yoy if np.isfinite(revenue_growth_yoy) else revenue_growth_qoq
        earnings_growth = earnings_growth_yoy if np.isfinite(earnings_growth_yoy) else earnings_growth_qoq
        growth_basis_state = "YOY_PRIMARY" if np.isfinite(revenue_growth_yoy) or np.isfinite(earnings_growth_yoy) else "QOQ_FALLBACK"

        revenue_ttm = _sum_latest_quarters(revenue_series)
        net_income_ttm = _sum_latest_quarters(net_income_series)
        operating_income_ttm = _sum_latest_quarters(operating_income_series)
        ocf_ttm = _sum_latest_quarters(ocf_series)
        capex_ttm = _sum_latest_quarters(capex_series)
        fcf = ocf + capex if np.isfinite(ocf) and np.isfinite(capex) else np.nan
        fcf_ttm = ocf_ttm + capex_ttm if np.isfinite(ocf_ttm) and np.isfinite(capex_ttm) else np.nan

        margin = 100 * net_income / revenue if np.isfinite(net_income) and np.isfinite(revenue) and revenue != 0 else np.nan
        operating_margin = 100 * operating_income / revenue if np.isfinite(operating_income) and np.isfinite(revenue) and revenue != 0 else np.nan
        net_margin_ttm = 100 * net_income_ttm / revenue_ttm if np.isfinite(net_income_ttm) and np.isfinite(revenue_ttm) and revenue_ttm != 0 else np.nan
        operating_margin_ttm = 100 * operating_income_ttm / revenue_ttm if np.isfinite(operating_income_ttm) and np.isfinite(revenue_ttm) and revenue_ttm != 0 else np.nan
        roe_ttm = 100 * net_income_ttm / equity if np.isfinite(net_income_ttm) and np.isfinite(equity) and equity != 0 else np.nan
        roa_ttm = 100 * net_income_ttm / assets if np.isfinite(net_income_ttm) and np.isfinite(assets) and assets != 0 else np.nan

        interest_debt_to_equity = debt / equity if np.isfinite(debt) and np.isfinite(equity) and equity != 0 else np.nan
        liabilities_to_equity = liabilities / equity if np.isfinite(liabilities) and np.isfinite(equity) and equity != 0 else np.nan
        net_debt = debt - cash if np.isfinite(debt) and np.isfinite(cash) else np.nan
        net_debt_to_equity = net_debt / equity if np.isfinite(net_debt) and np.isfinite(equity) and equity != 0 else np.nan
        current_ratio = current_assets / current_liabilities if np.isfinite(current_assets) and np.isfinite(current_liabilities) and current_liabilities != 0 else np.nan
        cash_to_debt = cash / debt if np.isfinite(cash) and np.isfinite(debt) and debt > 0 else np.nan
        ocf_conversion = ocf_ttm / net_income_ttm if np.isfinite(ocf_ttm) and np.isfinite(net_income_ttm) and net_income_ttm != 0 else (ocf / net_income if np.isfinite(ocf) and np.isfinite(net_income) and net_income != 0 else np.nan)

        rev_score = _clip(50 + revenue_growth) if np.isfinite(revenue_growth) else np.nan
        if earnings_yoy_state == "LOSS_TO_PROFIT" or (growth_basis_state == "QOQ_FALLBACK" and earnings_qoq_state == "LOSS_TO_PROFIT"):
            earn_score = 92.0
        elif earnings_yoy_state == "PROFIT_TO_LOSS" or (growth_basis_state == "QOQ_FALLBACK" and earnings_qoq_state == "PROFIT_TO_LOSS"):
            earn_score = 8.0
        else:
            earn_score = _clip(50 + 0.8 * earnings_growth) if np.isfinite(earnings_growth) else np.nan
        profitability_score = _weighted_mean_available([
            (_clip(45 + 1.5 * _finite(net_margin_ttm, margin)) if np.isfinite(_finite(net_margin_ttm, margin)) else np.nan, 0.35),
            (_clip(45 + 1.0 * roe_ttm) if np.isfinite(roe_ttm) else np.nan, 0.40),
            (_clip(45 + 1.3 * roa_ttm) if np.isfinite(roa_ttm) else np.nan, 0.25),
        ])
        cash_quality_score = _weighted_mean_available([
            (_clip(55 + 22 * ocf_conversion) if np.isfinite(ocf_conversion) else np.nan, 0.55),
            (80.0 if np.isfinite(fcf_ttm) and fcf_ttm > 0 else 35.0 if np.isfinite(fcf_ttm) else 75.0 if np.isfinite(fcf) and fcf > 0 else 35.0 if np.isfinite(fcf) else np.nan, 0.45),
        ])
        solvency_score = _weighted_mean_available([
            (_clip(88 - 30 * max(0, interest_debt_to_equity - 0.35)) if np.isfinite(interest_debt_to_equity) else np.nan, 0.40),
            (_clip(90 - 28 * max(0, liabilities_to_equity - 0.70)) if np.isfinite(liabilities_to_equity) else np.nan, 0.30),
            (_clip(45 + 30 * current_ratio) if np.isfinite(current_ratio) else np.nan, 0.15),
            (_clip(55 + 20 * cash_to_debt) if np.isfinite(cash_to_debt) else np.nan, 0.15),
        ])
        score = _weighted_mean_available([
            (rev_score, 0.18), (earn_score, 0.18), (profitability_score, 0.22),
            (cash_quality_score, 0.22), (solvency_score, 0.20),
        ])

        statement_values = [
            revenue, revenue_prev_q, net_income, net_income_prev_q, operating_income, equity, debt, cash, ocf, capex,
            liabilities, assets, current_assets, current_liabilities,
        ]
        statement_availability = 100.0 * sum(np.isfinite(value) for value in statement_values) / len(statement_values)
        critical_metrics = [
            revenue_growth, earnings_growth if np.isfinite(earnings_growth) else (92.0 if earnings_yoy_state == "LOSS_TO_PROFIT" else np.nan),
            _finite(net_margin_ttm, margin), interest_debt_to_equity, liabilities_to_equity, ocf_conversion, _finite(fcf_ttm, fcf), roe_ttm,
        ]
        critical_completeness = 100.0 * sum(np.isfinite(value) for value in critical_metrics) / len(critical_metrics)
        official_source_coverage = 0.0
        public_source_quality = 45.0
        growth_basis_quality = 100.0 if growth_basis_state == "YOY_PRIMARY" else 55.0
        coverage = 0.48 * critical_completeness + 0.27 * statement_availability + 0.10 * public_source_quality + 0.15 * growth_basis_quality

        if coverage < 35 or not np.isfinite(score):
            state = "FUNDAMENTAL_INCOMPLETE"
        elif score >= 68 and critical_completeness >= 62.5:
            state = "FUTURE_FUNDAMENTAL_SUPPORTIVE"
        elif score >= 48:
            state = "FUNDAMENTAL_MIXED"
        else:
            state = "FUNDAMENTAL_WEAK"

        latest_period = revenue_series.index[0] if len(revenue_series) and pd.notna(revenue_series.index[0]) else pd.NaT
        snapshot = {
            "ticker": symbol,
            "fundamental_latest_period": pd.Timestamp(latest_period).date().isoformat() if pd.notna(latest_period) else "",
            "revenue_latest": revenue, "revenue_ttm": revenue_ttm,
            "net_income_latest": net_income, "net_income_ttm": net_income_ttm,
            "operating_cash_flow_latest": ocf, "operating_cash_flow_ttm": ocf_ttm,
            "free_cash_flow_proxy_latest": fcf, "free_cash_flow_proxy_ttm": fcf_ttm,
            "cash_latest": cash, "debt_latest": debt, "total_liabilities_latest": liabilities,
            "total_assets_latest": assets, "equity_latest": equity,
            "current_assets_latest": current_assets, "current_liabilities_latest": current_liabilities,
            "revenue_growth_pct": round(revenue_growth, 2) if np.isfinite(revenue_growth) else np.nan,
            "earnings_growth_pct": round(earnings_growth, 2) if np.isfinite(earnings_growth) else np.nan,
            "revenue_growth_qoq_pct": round(revenue_growth_qoq, 2) if np.isfinite(revenue_growth_qoq) else np.nan,
            "revenue_growth_yoy_pct": round(revenue_growth_yoy, 2) if np.isfinite(revenue_growth_yoy) else np.nan,
            "earnings_growth_qoq_pct": round(earnings_growth_qoq, 2) if np.isfinite(earnings_growth_qoq) else np.nan,
            "earnings_growth_yoy_pct": round(earnings_growth_yoy, 2) if np.isfinite(earnings_growth_yoy) else np.nan,
            "earnings_growth_yoy_state": earnings_yoy_state, "growth_basis_state": growth_basis_state,
            "net_margin_pct": round(margin, 2) if np.isfinite(margin) else np.nan,
            "operating_margin_pct": round(operating_margin, 2) if np.isfinite(operating_margin) else np.nan,
            "net_margin_ttm_pct": round(net_margin_ttm, 2) if np.isfinite(net_margin_ttm) else np.nan,
            "operating_margin_ttm_pct": round(operating_margin_ttm, 2) if np.isfinite(operating_margin_ttm) else np.nan,
            "roe_ttm_pct": round(roe_ttm, 2) if np.isfinite(roe_ttm) else np.nan,
            "roa_ttm_pct": round(roa_ttm, 2) if np.isfinite(roa_ttm) else np.nan,
            # der_ratio is retained for backward compatibility, but is explicitly interest-bearing debt/equity.
            "der_ratio": round(interest_debt_to_equity, 3) if np.isfinite(interest_debt_to_equity) else np.nan,
            "der_definition_state": "INTEREST_BEARING_DEBT_TO_EQUITY",
            "interest_bearing_debt_to_equity": round(interest_debt_to_equity, 3) if np.isfinite(interest_debt_to_equity) else np.nan,
            "total_liabilities_to_equity": round(liabilities_to_equity, 3) if np.isfinite(liabilities_to_equity) else np.nan,
            "net_debt_to_equity": round(net_debt_to_equity, 3) if np.isfinite(net_debt_to_equity) else np.nan,
            "current_ratio": round(current_ratio, 3) if np.isfinite(current_ratio) else np.nan,
            "cash_to_debt_ratio": round(cash_to_debt, 3) if np.isfinite(cash_to_debt) else np.nan,
            "ocf_conversion_ratio": round(ocf_conversion, 3) if np.isfinite(ocf_conversion) else np.nan,
            "fundamental_profitability_score": round(profitability_score, 1) if np.isfinite(profitability_score) else np.nan,
            "fundamental_cash_quality_score": round(cash_quality_score, 1) if np.isfinite(cash_quality_score) else np.nan,
            "fundamental_solvency_score": round(solvency_score, 1) if np.isfinite(solvency_score) else np.nan,
            "fundamental_conversion_score": round(score, 1) if np.isfinite(score) else np.nan,
            "fundamental_coverage_pct": round(coverage, 1),
            "fundamental_statement_availability_pct": round(statement_availability, 1),
            "fundamental_critical_metric_completeness_pct": round(critical_completeness, 1),
            "fundamental_official_source_coverage_pct": round(official_source_coverage, 1),
            "fundamental_public_source_quality_pct": round(public_source_quality, 1),
            "fundamental_state": state,
            "fundamental_provenance_state": "YFINANCE_PUBLIC_FINANCIAL_STATEMENT_PROXY_NOT_OFFICIAL_FILING",
        }
        audit_status = "OK" if coverage >= 35 else "PARTIAL" if coverage > 0 else "NO_ITEMS"
        return snapshot, {
            "ticker": symbol, "provider": "YFINANCE_FUNDAMENTALS", "status": audit_status,
            "items": int(coverage > 0),
            "detail": f"coverage={coverage:.1f}; critical={critical_completeness:.1f}; statements={statement_availability:.1f}; growth_basis={growth_basis_state}; official=0.0",
        }
    except Exception as exc:
        return {"ticker": symbol, "fundamental_provenance_state": "PROVIDER_FAILED", "fundamental_coverage_pct": 0.0}, {"ticker": symbol, "provider": "YFINANCE_FUNDAMENTALS", "status": "ERROR", "items": 0, "detail": f"{type(exc).__name__}: {exc}"}


def fetch_many_fundamentals(tickers: Iterable[str], max_workers: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = list(dict.fromkeys(normalize_ticker(t) for t in tickers if normalize_ticker(t)))
    snapshots: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 3))) as executor:
        futures = {executor.submit(fetch_yfinance_fundamental_snapshot, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            snapshot, audit = future.result()
            snapshots.append(snapshot)
            audits.append(audit)
    return pd.DataFrame(snapshots), pd.DataFrame(audits)


def autonomous_evidence_frame(
    profiles: pd.DataFrame,
    actions: pd.DataFrame,
    fundamentals: pd.DataFrame,
    broker_proxy_map: Mapping[str, Mapping[str, Any]],
    orderbook_proxy_map: Mapping[str, Mapping[str, Any]],
    as_of: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    observed = pd.Timestamp(as_of).isoformat()
    for frame, evidence_type in ((profiles, "KSEI_SECURITY_PROFILE"), (actions, "KSEI_CORPORATE_ACTION"), (fundamentals, "PUBLIC_FUNDAMENTAL_PROXY")):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for _, row in frame.iterrows():
                record = row.to_dict()
                rows.append({"ticker": normalize_ticker(record.get("ticker")), "evidence_type": evidence_type, "observed_at": observed, "source_verified": bool(record.get("ksei_source_verified", False) or record.get("source_verified", False)), **record})
    for ticker, payload in broker_proxy_map.items():
        rows.append({"ticker": ticker, "evidence_type": "BROKER_INVENTORY_OHLCV_PROXY", "observed_at": observed, "source_verified": False, **dict(payload)})
    for ticker, payload in orderbook_proxy_map.items():
        rows.append({"ticker": ticker, "evidence_type": "BID_OFFER_EOD_PROXY", "observed_at": observed, "source_verified": False, **dict(payload)})
    return pd.DataFrame(rows)


__all__ = [
    "apply_regulatory_event_overlay", "autonomous_evidence_frame", "build_broker_inventory_proxy", "build_orderbook_proxy",
    "fetch_many_fundamentals", "fetch_many_ksei_profiles", "fetch_yfinance_fundamental_snapshot",
    "ksei_actions_to_events", "ksei_profiles_to_maps", "parse_ksei_profile_html",
]
