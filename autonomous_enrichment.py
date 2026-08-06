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
    value = str(text or "").strip()
    if re.fullmatch(r"\d{8}", value):
        return pd.to_datetime(value, format="%Y%m%d", errors="coerce")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(value, errors="coerce", dayfirst=True)


def build_broker_inventory_proxy(features: Mapping[str, Any]) -> dict[str, Any]:
    """OHLCV-only behavioural proxy. It never identifies a broker or beneficial owner."""
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

    persistence = _clip(35 + 7.0 * accumulation + 5.0 * absorption - 8.0 * distribution - 5.0 * failed_absorption)
    pressure = _clip(
        0.28 * _finite(smart, 50)
        + 0.20 * _finite(close_acceptance, 50)
        + 0.16 * _finite(up_value, 50)
        + 0.14 * _clip(50 + 250 * _finite(cmf, 0))
        + 0.12 * _clip(50 + 4 * _finite(obv_slope, 0))
        + 0.10 * _finite(contraction, 50)
    )
    dryness = _clip(0.45 * persistence + 0.25 * _finite(contraction, 50) + 0.20 * _finite(close_acceptance, 50) + 0.10 * _clip(70 - 12 * max(0, _finite(volume_ratio, 1) - 1)))
    score = _clip(0.45 * pressure + 0.35 * persistence + 0.20 * dryness)
    if distribution >= 4 or failed_absorption >= 4:
        shift = "DISTRIBUTION_RISK_PROXY"
    elif score >= 65 and persistence >= 60:
        shift = "COLLECTION_PERSISTING_PROXY"
    elif score >= 55:
        shift = "BOTTOMING_OR_EARLY_COLLECTION_PROXY"
    else:
        shift = "NO_CLEAR_INVENTORY_PROXY"
    defended = np.nan
    if np.isfinite(low20) and np.isfinite(ema20):
        defended = max(low20, min(ema20, _finite(features.get("last_price"), ema20)))
    coverage_fields = [smart, close_acceptance, up_value, cmf, obv_slope, contraction, volume_ratio]
    coverage = 100 * sum(np.isfinite(v) for v in coverage_fields) / len(coverage_fields)
    return {
        "broker_summary_score": round(pressure, 1),
        "broker_summary_coverage_pct": round(coverage * 0.65, 1),
        "broker_net_ratio": np.nan,
        "broker_inventory_score": round(score, 1),
        "broker_inventory_coverage_pct": round(coverage * 0.65, 1),
        "inventory_coverage_years": np.nan,
        "holder_persistence_score": round(persistence, 1),
        "inventory_dryness_score": round(dryness, 1),
        "retail_exit_score": np.nan,
        "retail_cannibalisation_risk": np.nan,
        "fund_like_flow_score": np.nan,
        "jumbo_crossing_score": np.nan,
        "defended_level": round(defended, 4) if np.isfinite(defended) else np.nan,
        "defended_level_score": round(_clip(0.6 * persistence + 0.4 * pressure), 1),
        "broker_inventory_shift_state": shift,
        "broker_summary_provenance_state": "OHLCV_BEHAVIOURAL_PROXY_NOT_BROKER_DATA",
        "beneficial_owner_inference_state": "NOT_INFERRED_FROM_OHLCV_OR_BROKER_CODE",
        "broker_inventory_evidence_type": "OHLCV_PROXY",
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
        revenue = _statement_value(income, ["Total Revenue", "Operating Revenue"], 0)
        revenue_prev = _statement_value(income, ["Total Revenue", "Operating Revenue"], 1)
        net_income = _statement_value(income, ["Net Income", "Net Income Common Stockholders"], 0)
        net_income_prev = _statement_value(income, ["Net Income", "Net Income Common Stockholders"], 1)
        operating_income = _statement_value(income, ["Operating Income"], 0)
        equity = _statement_value(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest"], 0)
        debt = _statement_value(balance, ["Total Debt"], 0)
        cash = _statement_value(balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"], 0)
        ocf = _statement_value(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], 0)
        capex = _statement_value(cashflow, ["Capital Expenditure"], 0)

        revenue_growth = 100 * (revenue / revenue_prev - 1) if np.isfinite(revenue) and np.isfinite(revenue_prev) and revenue_prev != 0 else np.nan
        earnings_growth = 100 * (net_income / abs(net_income_prev) - 1) if np.isfinite(net_income) and np.isfinite(net_income_prev) and net_income_prev != 0 else np.nan
        margin = 100 * net_income / revenue if np.isfinite(net_income) and np.isfinite(revenue) and revenue != 0 else np.nan
        operating_margin = 100 * operating_income / revenue if np.isfinite(operating_income) and np.isfinite(revenue) and revenue != 0 else np.nan
        der = debt / equity if np.isfinite(debt) and np.isfinite(equity) and equity != 0 else np.nan
        ocf_conversion = ocf / net_income if np.isfinite(ocf) and np.isfinite(net_income) and net_income != 0 else np.nan
        fcf = ocf + capex if np.isfinite(ocf) and np.isfinite(capex) else np.nan

        quality_components = [
            _clip(50 + revenue_growth) if np.isfinite(revenue_growth) else np.nan,
            _clip(50 + 0.8 * earnings_growth) if np.isfinite(earnings_growth) else np.nan,
            _clip(50 + 2 * margin) if np.isfinite(margin) else np.nan,
            _clip(60 + 25 * ocf_conversion) if np.isfinite(ocf_conversion) else np.nan,
            _clip(85 - 25 * max(0, der - 0.5)) if np.isfinite(der) else np.nan,
            75.0 if np.isfinite(fcf) and fcf > 0 else 35.0 if np.isfinite(fcf) else np.nan,
        ]
        available = [value for value in quality_components if np.isfinite(value)]
        score = float(np.mean(available)) if available else np.nan

        statement_values = [revenue, revenue_prev, net_income, net_income_prev, operating_income, equity, debt, cash, ocf, capex]
        statement_availability = 100.0 * sum(np.isfinite(value) for value in statement_values) / len(statement_values)
        critical_metrics = [revenue_growth, earnings_growth, margin, der, ocf_conversion, fcf]
        critical_completeness = 100.0 * sum(np.isfinite(value) for value in critical_metrics) / len(critical_metrics)
        # yfinance is a public secondary proxy, not an official filing feed. Keep source coverage explicit and conservative.
        official_source_coverage = 0.0
        public_source_quality = 45.0
        coverage = 0.55 * critical_completeness + 0.30 * statement_availability + 0.15 * public_source_quality

        if coverage < 35 or len(available) < 2:
            state = "FUNDAMENTAL_INCOMPLETE"
        elif np.isfinite(score) and score >= 62 and critical_completeness >= 66.7:
            state = "FUTURE_FUNDAMENTAL_SUPPORTIVE"
        elif np.isfinite(score) and score >= 45:
            state = "FUNDAMENTAL_MIXED"
        elif np.isfinite(score):
            state = "FUNDAMENTAL_WEAK"
        else:
            state = "FUNDAMENTAL_UNAVAILABLE"

        snapshot = {
            "ticker": symbol,
            "revenue_latest": revenue,
            "net_income_latest": net_income,
            "operating_cash_flow_latest": ocf,
            "free_cash_flow_proxy_latest": fcf,
            "cash_latest": cash,
            "debt_latest": debt,
            "equity_latest": equity,
            "revenue_growth_pct": round(revenue_growth, 2) if np.isfinite(revenue_growth) else np.nan,
            "earnings_growth_pct": round(earnings_growth, 2) if np.isfinite(earnings_growth) else np.nan,
            "net_margin_pct": round(margin, 2) if np.isfinite(margin) else np.nan,
            "operating_margin_pct": round(operating_margin, 2) if np.isfinite(operating_margin) else np.nan,
            "der_ratio": round(der, 3) if np.isfinite(der) else np.nan,
            "ocf_conversion_ratio": round(ocf_conversion, 3) if np.isfinite(ocf_conversion) else np.nan,
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
            "detail": f"coverage={coverage:.1f}; critical={critical_completeness:.1f}; statements={statement_availability:.1f}; official=0.0",
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
