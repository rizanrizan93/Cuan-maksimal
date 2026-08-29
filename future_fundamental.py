from __future__ import annotations

from typing import Any, Mapping
import math
import re

import numpy as np
import pandas as pd

from release_contract import SCANNER_RELEASE_VERSION

FUTURE_FUNDAMENTAL_VERSION = "1.0.3-direct-forward-lineage"
SCANNER_VERSION = SCANNER_RELEASE_VERSION

PROJECT_TERMS = (
    "project", "proyek", "expansion", "ekspansi", "capacity", "kapasitas", "plant", "pabrik",
    "smelter", "mine", "tambang", "commissioning", "commercial operation", "operasi komersial",
    "acquisition", "akuisisi", "joint venture", "hilirisasi", "data center", "data centre",
)
CONTRACT_TERMS = (
    "contract", "kontrak", "order book", "orderbook", "backlog", "offtake", "off-take", "purchase order",
    "po ", "framework agreement", "government contract", "proyek pemerintah", "tender won", "menang tender",
)
GUIDANCE_TERMS = (
    "guidance", "target revenue", "target pendapatan", "target laba", "sales target", "production target",
    "utilization", "utilisasi", "volume target", "shipment target", "capacity target",
)
CAPEX_TERMS = ("capex", "capital expenditure", "belanja modal", "investasi", "investment")
ALIGNMENT_TERMS = (
    "buyback", "insider buy", "director buy", "director", "direksi", "commissioner", "komisaris", "ceo", "cfo", "management", "manajemen", "appointed", "appointment", "pengangkatan", "strategic investor", "pemegang saham", "pengendali",
    "acquisition", "akuisisi", "merger", "spin off", "rights issue", "private placement",
)
NEGATIVE_FORWARD_TERMS = (
    "cancelled", "canceled", "dibatalkan", "delay", "delayed", "tertunda", "postponed", "shutdown",
    "default", "gagal bayar", "guidance cut", "target diturunkan", "cost overrun", "over budget",
)


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clip(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    number = _finite(value, low)
    return max(low, min(high, number))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "verified", "on"}


def _event_text(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in ("title", "summary", "category")).lower()


def _event_matches_terms(row: Mapping[str, Any], terms: tuple[str, ...]) -> bool:
    """Match forward terms without turning ticker/company identity into a catalyst.

    Administrative corporate actions and backward-looking earnings filings are
    evidence for other scanner layers, not forward project/contract evidence.
    The bare ticker is stripped before term matching so symbols such as MINE do
    not become a mining-project event merely because the symbol appears in a
    title or KSEI summary.
    """
    category = str(row.get("category") or "").strip().upper()
    role = str(row.get("event_role") or "").strip().upper()
    if role == "ADMINISTRATIVE_CORPORATE_ACTION" or category in {"EARNINGS_CONVERSION", "CORPORATE_ACTION"}:
        return False
    text = _event_text(row)
    ticker = str(row.get("ticker") or "").strip().lower()
    if ticker.endswith(".jk"):
        ticker = ticker[:-3]
    if ticker:
        text = re.sub(rf"(?<![a-z0-9]){re.escape(ticker)}(?![a-z0-9])", " ", text)
    for raw_term in terms:
        term = str(raw_term or "").strip().lower()
        if not term:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text):
            return True
    return False


def _weighted_observed(components: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Return observed quality and fixed-denominator evidence coverage."""
    total_weight = sum(weight for _, weight, _ in components if weight > 0)
    if total_weight <= 0:
        return np.nan, 0.0
    quality_sum = 0.0
    observed_weight = 0.0
    coverage_sum = 0.0
    for value, weight, coverage_pct in components:
        if weight <= 0:
            continue
        coverage_sum += weight * _clip(coverage_pct) / 100.0
        if np.isfinite(value):
            quality_sum += weight * _clip(value)
            observed_weight += weight
    quality = quality_sum / observed_weight if observed_weight > 0 else np.nan
    return quality, 100.0 * coverage_sum / total_weight


def _event_bucket(events: pd.DataFrame | None, terms: tuple[str, ...], as_of: Any = None) -> dict[str, Any]:
    if not isinstance(events, pd.DataFrame) or events.empty:
        return {"score": np.nan, "coverage": 0.0, "count": 0, "verified": 0, "official": 0, "latest_days": np.nan}
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    hits: list[tuple[float, float, bool, bool]] = []
    for _, series in events.iterrows():
        row = series.to_dict()
        text = _event_text(row)
        if not _event_matches_terms(row, terms):
            continue
        published = pd.to_datetime(row.get("published_at") or row.get("event_date"), errors="coerce", utc=True)
        # Point-in-time safety: a project/contract event observed after `as_of`
        # cannot contribute to a historical decision.
        if pd.notna(published) and published > now:
            continue
        age_days = max(0.0, (now - published).total_seconds() / 86400.0) if pd.notna(published) else 120.0
        freshness = 100.0 * math.exp(-age_days / 90.0)
        verified = _truthy(row.get("source_verified"))
        tier = str(row.get("source_tier") or "").upper()
        official = verified and tier in {"OFFICIAL", "ISSUER", "REGULATOR"}
        materiality = _finite(row.get("materiality_score"), 55.0)
        bridge = _finite(row.get("financial_bridge_score"), 50.0)
        source_quality = 100.0 if official else 82.0 if verified else 60.0 if row.get("url") else 35.0
        negative_hits = sum(1 for term in NEGATIVE_FORWARD_TERMS if term in text)
        score = _clip(0.30 * freshness + 0.30 * materiality + 0.20 * bridge + 0.20 * source_quality - 18.0 * negative_hits)
        hits.append((score, age_days, verified, official))
    if not hits:
        return {"score": np.nan, "coverage": 0.0, "count": 0, "verified": 0, "official": 0, "latest_days": np.nan}
    hits.sort(key=lambda item: item[0], reverse=True)
    top = hits[:4]
    score = float(np.average([item[0] for item in top], weights=np.linspace(1.0, 0.6, len(top))))
    verified_count = sum(int(item[2]) for item in hits)
    official_count = sum(int(item[3]) for item in hits)
    coverage = _clip(18 * min(len(hits), 4) + 18 * min(verified_count, 2) + 20 * min(official_count, 1))
    return {
        "score": score,
        "coverage": coverage,
        "count": len(hits),
        "verified": verified_count,
        "official": official_count,
        "latest_days": min(item[1] for item in hits),
    }



def _forward_event_stats(events: pd.DataFrame | None, as_of: Any = None) -> dict[str, Any]:
    if not isinstance(events, pd.DataFrame) or events.empty:
        return {"count": 0, "verified": 0, "official": 0, "latest_days": np.nan}
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    terms = PROJECT_TERMS + CAPEX_TERMS + CONTRACT_TERMS + GUIDANCE_TERMS
    seen: set[tuple[str, str, str]] = set()
    verified = official = 0
    ages: list[float] = []
    for _, series in events.iterrows():
        row = series.to_dict()
        text = _event_text(row)
        if not _event_matches_terms(row, terms):
            continue
        published = pd.to_datetime(row.get("published_at") or row.get("event_date"), errors="coerce", utc=True)
        if pd.notna(published) and published > now:
            continue
        key = (str(row.get("title") or ""), str(row.get("url") or ""), str(row.get("published_at") or row.get("event_date") or ""))
        if key in seen:
            continue
        seen.add(key)
        is_verified = _truthy(row.get("source_verified"))
        tier = str(row.get("source_tier") or "").upper()
        is_official = bool(is_verified and tier in {"OFFICIAL", "ISSUER", "REGULATOR"})
        verified += int(is_verified)
        official += int(is_official)
        if pd.notna(published):
            ages.append(max(0.0, (now - published).total_seconds() / 86400.0))
    return {"count": len(seen), "verified": verified, "official": official, "latest_days": min(ages) if ages else np.nan}


def _funding_capacity(fundamental: Mapping[str, Any]) -> tuple[float, float, list[str]]:
    coverage = _finite(fundamental.get("fundamental_coverage_pct"), 0.0)
    if coverage <= 0:
        return np.nan, 0.0, ["FUNDING_CAPACITY_EVIDENCE_MISSING"]
    cashflow_quality = str(fundamental.get("fundamental_cashflow_quality_state") or "").upper()
    leverage = str(fundamental.get("fundamental_leverage_risk_state") or "").upper()
    current_ratio = _finite(fundamental.get("current_ratio"), np.nan)
    cash_to_debt = _finite(fundamental.get("cash_to_debt_ratio"), np.nan)
    score = 60.0
    flags: list[str] = []
    if cashflow_quality == "CASHFLOW_POSITIVE_CONVERTING":
        score += 22
    elif cashflow_quality == "CASHFLOW_POSITIVE_PARTIAL":
        score += 12
    elif cashflow_quality == "CASHFLOW_POSITIVE_WEAK_CONVERSION":
        score += 3; flags.append("WEAK_CASH_CONVERSION")
    elif cashflow_quality == "FCF_NEGATIVE_CAPEX_REVIEW":
        score += 5; flags.append("FCF_NEGATIVE_CAPEX_REVIEW")
    elif cashflow_quality in {"OCF_NEGATIVE", "OCF_AND_FCF_NEGATIVE"}:
        score -= 35; flags.append("OPERATING_CASH_FLOW_NEGATIVE")
    else:
        flags.append("CASHFLOW_QUALITY_NOT_VERIFIED")
    if leverage == "BALANCE_SHEET_CAPACITY_OK":
        score += 10
    elif leverage == "HIGH_LEVERAGE":
        score -= 16; flags.append("HIGH_LEVERAGE")
    elif leverage == "EXTREME_LEVERAGE":
        score -= 30; flags.append("EXTREME_LEVERAGE")
    if np.isfinite(current_ratio):
        score += 7 if current_ratio >= 1.5 else 2 if current_ratio >= 1.0 else -8
    if np.isfinite(cash_to_debt):
        score += 7 if cash_to_debt >= 1.0 else 2 if cash_to_debt >= 0.5 else -6
    return _clip(score), min(100.0, coverage), flags


def _business_confirmation(fundamental: Mapping[str, Any]) -> tuple[float, float, list[str]]:
    coverage = _finite(fundamental.get("fundamental_coverage_pct"), 0.0)
    conversion = _finite(fundamental.get("fundamental_conversion_score"), np.nan)
    consistency = str(fundamental.get("fundamental_growth_consistency_state") or "").upper()
    freshness = str(fundamental.get("fundamental_period_freshness_state") or "").upper()
    if not np.isfinite(conversion):
        return np.nan, 0.0, ["BUSINESS_CONFIRMATION_MISSING"]
    score = conversion
    flags: list[str] = []
    if consistency == "QUARTER_AND_YTD_CONFIRMED":
        score += 10
    elif consistency in {"TURNAROUND_INFLECTION_UNCONFIRMED", "QUARTER_YTD_DIVERGENCE_REVIEW"}:
        score -= 8; flags.append(consistency)
    elif consistency == "QUARTER_AND_YTD_WEAK":
        score -= 18; flags.append("QUARTER_AND_YTD_WEAK")
    if freshness in {"LAGGING_REPORTING_PERIOD", "STALE_RELATIVE_TO_UNIVERSE", "STALE_QUARTERLY_PERIOD", "UNKNOWN_PERIOD"}:
        score -= 15; flags.append("LATEST_REPORT_REFRESH_REQUIRED")
    return _clip(score), min(100.0, coverage), flags


def calculate_future_fundamental(
    *,
    ticker: str,
    events: pd.DataFrame | None,
    narrative: Mapping[str, Any] | None,
    fundamental: Mapping[str, Any] | None,
    ownership: Mapping[str, Any] | None,
    sector: Mapping[str, Any] | None,
    issuer_context: Mapping[str, Any] | None = None,
    as_of: Any = None,
) -> dict[str, Any]:
    """Forward-looking, evidence-separated future-fundamental layer.

    This does not invent project NPVs or earnings forecasts. It measures visibility of
    public forward evidence and whether the current balance sheet/cash-flow can plausibly
    fund and convert that evidence into revenue/margin/earnings/cash-flow.
    """
    narrative = dict(narrative or {})
    fundamental = dict(fundamental or {})
    ownership = dict(ownership or {})
    sector = dict(sector or {})

    decision_time = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
    fundamental_observed = pd.to_datetime(fundamental.get("fundamental_observed_at"), errors="coerce", utc=True)
    if pd.notna(fundamental_observed) and fundamental_observed > decision_time:
        fundamental = {}
        fundamental_availability_state = "FUTURE_FUNDAMENTAL_SNAPSHOT_EXCLUDED"
    elif pd.notna(fundamental_observed):
        fundamental_availability_state = "AVAILABLE_AS_OF"
    else:
        fundamental_availability_state = "AVAILABILITY_TIMESTAMP_UNVERIFIED"
    project = _event_bucket(events, PROJECT_TERMS + CAPEX_TERMS, as_of=as_of)
    contracts = _event_bucket(events, CONTRACT_TERMS + GUIDANCE_TERMS, as_of=as_of)
    alignment_events = _event_bucket(events, ALIGNMENT_TERMS, as_of=as_of)

    top_down = _finite(narrative.get("top_down_catalyst_score"), np.nan)
    industry = _finite(narrative.get("industry_translation_score"), np.nan)
    narrative_cov = _finite(narrative.get("narrative_coverage_pct"), 0.0)
    sector_score = _finite(sector.get("sector_leadership_score"), np.nan)
    sector_cov = _finite(sector.get("sector_context_coverage_pct"), 0.0)
    sector_runway, sector_runway_cov = _weighted_observed([
        (top_down, 0.35, narrative_cov if np.isfinite(top_down) else 0),
        (industry, 0.35, narrative_cov if np.isfinite(industry) else 0),
        (sector_score, 0.30, sector_cov if np.isfinite(sector_score) else 0),
    ])

    issuer_alignment = _finite(narrative.get("issuer_alignment_score"), np.nan)
    issuer_alignment_cov = _finite(narrative.get("issuer_alignment_coverage_pct"), 0.0)
    ownership_score = _finite(ownership.get("ownership_score"), np.nan)
    ownership_cov = _finite(ownership.get("ownership_coverage_pct"), 0.0)
    management_alignment, management_cov = _weighted_observed([
        (issuer_alignment, 0.50, issuer_alignment_cov if np.isfinite(issuer_alignment) else 0),
        (ownership_score, 0.25, ownership_cov if np.isfinite(ownership_score) else 0),
        (alignment_events["score"], 0.25, alignment_events["coverage"]),
    ])

    funding, funding_cov, funding_flags = _funding_capacity(fundamental)
    confirmation, confirmation_cov, confirmation_flags = _business_confirmation(fundamental)

    project_score = project["score"]
    project_cov = project["coverage"]
    contract_score = contracts["score"]
    contract_cov = contracts["coverage"]

    # Keep the direct project/contract signal separate from the holistic future
    # fundamental score.  The latter deliberately includes funding capacity,
    # current-business confirmation, sector and management alignment; feeding it
    # unchanged into a business-heavy ranking would count those same evidence
    # families twice.  This direct signal is the incremental forward evidence
    # used by Next Leader calibration.
    direct_forward_score, direct_forward_coverage = _weighted_observed([
        (project_score, 0.625, project_cov),
        (contract_score, 0.375, contract_cov),
    ])

    score, coverage = _weighted_observed([
        (project_score, 0.25, project_cov),
        (contract_score, 0.15, contract_cov),
        (sector_runway, 0.15, sector_runway_cov),
        (management_alignment, 0.15, management_cov),
        (funding, 0.15, funding_cov),
        (confirmation, 0.15, confirmation_cov),
    ])

    forward_stats = _forward_event_stats(events, as_of=as_of)
    forward_event_count = int(forward_stats["count"])
    verified_forward_count = int(forward_stats["verified"])
    official_forward_count = int(forward_stats["official"])
    flags: list[str] = [*funding_flags, *confirmation_flags]

    # Missing or unverified forward evidence is a confidence problem, not a reason
    # to collapse otherwise-different issuers onto an identical hard-cap score.
    # Coverage already carries the fixed-denominator evidence penalty; this
    # multiplier applies a second, continuous conviction discount while retaining
    # cross-sectional discrimination from sector, balance-sheet and business data.
    forward_evidence_multiplier = 1.0
    if forward_event_count == 0:
        flags.append("NO_FORWARD_PROJECT_OR_CONTRACT_EVIDENCE")
        forward_evidence_multiplier = 0.82
    elif verified_forward_count == 0:
        flags.append("FORWARD_EVIDENCE_PUBLIC_UNVERIFIED")
        forward_evidence_multiplier = 0.92
    if np.isfinite(score):
        score = _clip(score * forward_evidence_multiplier)

    if "CONTRADICTED_OR_NEGATIVE" in str(narrative.get("narrative_state") or ""):
        flags.append("FORWARD_NARRATIVE_CONTRADICTED")
        if np.isfinite(score):
            score = max(0.0, score - 15.0)

    if not np.isfinite(score) or coverage < 25:
        state = "FUTURE_FUNDAMENTAL_EVIDENCE_PENDING"
    elif score >= 75 and coverage >= 70 and verified_forward_count >= 1:
        state = "FUTURE_FUNDAMENTAL_HIGH_VISIBILITY"
    elif score >= 60 and coverage >= 50:
        state = "FUTURE_FUNDAMENTAL_SUPPORTIVE"
    elif score >= 45:
        state = "FUTURE_FUNDAMENTAL_DEVELOPING"
    else:
        state = "FUTURE_FUNDAMENTAL_WEAK_OR_UNPROVEN"

    latest_forward_age = _finite(forward_stats.get("latest_days"), np.nan)
    horizon = "NEXT_2_4_QUARTERS" if contracts["count"] > 0 or any(term in str(narrative.get("narrative_latest_title") or "").lower() for term in GUIDANCE_TERMS) else "12M_PLUS_PROJECT_RUNWAY" if project["count"] > 0 else "UNPROVEN_HORIZON"

    drivers = []
    if project["count"]:
        drivers.append(f"PROJECT_CAPACITY:{project['count']}")
    if contracts["count"]:
        drivers.append(f"CONTRACT_GUIDANCE:{contracts['count']}")
    if np.isfinite(funding):
        drivers.append(f"FUNDING_CAPACITY:{round(funding, 1)}")
    if np.isfinite(confirmation):
        drivers.append(f"BUSINESS_CONFIRMATION:{round(confirmation, 1)}")
    if np.isfinite(sector_runway):
        drivers.append(f"SECTOR_RUNWAY:{round(sector_runway, 1)}")

    return {
        "future_fundamental_version": FUTURE_FUNDAMENTAL_VERSION,
        "future_fundamental_score": round(float(score), 1) if np.isfinite(score) else np.nan,
        "future_fundamental_coverage_pct": round(float(coverage), 1),
        "future_fundamental_state": state,
        "future_project_capacity_score": round(float(project_score), 1) if np.isfinite(project_score) else np.nan,
        "future_contract_backlog_visibility_score": round(float(contract_score), 1) if np.isfinite(contract_score) else np.nan,
        "future_direct_forward_visibility_score": round(float(direct_forward_score), 1) if np.isfinite(direct_forward_score) else np.nan,
        "future_direct_forward_visibility_coverage_pct": round(float(direct_forward_coverage), 1),
        "future_direct_forward_lineage_state": "DIRECT_PROJECT_CONTRACT_ONLY_NO_CURRENT_FUNDAMENTAL_REUSE",
        "future_sector_policy_runway_score": round(float(sector_runway), 1) if np.isfinite(sector_runway) else np.nan,
        "future_management_alignment_score": round(float(management_alignment), 1) if np.isfinite(management_alignment) else np.nan,
        "future_funding_capacity_score": round(float(funding), 1) if np.isfinite(funding) else np.nan,
        "future_business_confirmation_score": round(float(confirmation), 1) if np.isfinite(confirmation) else np.nan,
        "future_forward_event_count": forward_event_count,
        "future_verified_forward_event_count": verified_forward_count,
        "future_official_forward_event_count": official_forward_count,
        "future_latest_forward_event_age_days": round(float(latest_forward_age), 1) if np.isfinite(latest_forward_age) else np.nan,
        "future_fundamental_horizon_state": horizon,
        "future_fundamental_drivers": " | ".join(drivers) or "NO_FORWARD_DRIVER_CONFIRMED",
        "future_fundamental_risk_flags": " | ".join(dict.fromkeys(flags)) or "NO_MAJOR_FORWARD_FUNDAMENTAL_RISK",
        "future_fundamental_input_availability_state": fundamental_availability_state,
        "future_fundamental_input_observed_at": fundamental_observed.isoformat() if pd.notna(fundamental_observed) else "",
    }


def future_fundamental_evidence_frame(radar: pd.DataFrame | None, observed_at: Any = None) -> pd.DataFrame:
    if not isinstance(radar, pd.DataFrame) or radar.empty or "ticker" not in radar.columns:
        return pd.DataFrame()
    columns = [column for column in radar.columns if column.startswith("future_")]
    if not columns:
        return pd.DataFrame()
    observed = pd.Timestamp.now(tz="UTC") if observed_at is None else pd.Timestamp(observed_at)
    observed = observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
    rows = []
    for _, row in radar[["ticker", *columns]].iterrows():
        payload = row.to_dict()
        rows.append({
            "ticker": str(payload.pop("ticker")),
            "evidence_type": "FUTURE_FUNDAMENTAL_SNAPSHOT",
            "observed_at": observed.isoformat(),
            "source_verified": bool(_finite(payload.get("future_verified_forward_event_count"), 0) > 0),
            **payload,
        })
    return pd.DataFrame(rows)


__all__ = [
    "FUTURE_FUNDAMENTAL_VERSION",
    "SCANNER_VERSION",
    "calculate_future_fundamental",
    "future_fundamental_evidence_frame",
]
