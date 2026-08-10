from __future__ import annotations

from html import escape
from typing import Any, Mapping

import numpy as np
import pandas as pd


BLOCKED_STATES = {
    "EMIR_DATA_INTEGRITY_BLOCK",
    "EMIR_REJECT_IDX_INTEGRITY",
    "EMIR_REJECT_SMART_MONEY_DISTRIBUTION",
    "EMIR_AVOID_RETAIL_EUPHORIA",
    "EMIR_CALIBRATION_REJECTED",
}

STATE_PRIORITY = {
    "EMIR_READY_WITH_PRECISE_TRIGGER": 0,
    "EMIR_AUTO_EOD_READY": 1,
    "EMIR_CORE_THESIS_READY_WAIT_IDX_INTEGRITY": 2,
    "EMIR_THESIS_READY_WAIT_BID_OFFER": 2,
    "EMIR_WATCH_INVENTORY_COLLECTION": 3,
    "EMIR_WAIT_NARRATIVE": 4,
    "EMIR_WAIT_MONEY_FLOW": 5,
    "EMIR_WAIT_FUNDAMENTAL_CONVERSION": 6,
    "EMIR_WAIT_REACCUMULATION": 7,
    "EMIR_FUNDAMENTAL_EVIDENCE_PENDING": 8,
    "EMIR_EVIDENCE_PENDING": 9,
    "EMIR_NO_EDGE_YET": 10,
    "EMIR_RADAR_ONLY_NOT_DEEP_REVIEWED": 11,
}

FACTOR_FIELDS = {
    "Narrative": "narrative_score",
    "Flow": "dashboard_flow_score",
    "Silent Accum": "dashboard_silent_accum_score",
    "Smart Money": "smart_money_score",
    "Structure": "market_structure_score",
    "Momentum": "dashboard_momentum_score",
    "Future Fundamental": "fundamental_conversion_score",
    "Liquidity": "liquidity_score",
}


def _num(value: Any, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _score(value: Any, default: float = 0.0) -> float:
    parsed = _num(value, default)
    return float(np.clip(parsed, 0.0, 100.0))


def _weighted(values: list[tuple[Any, float]], fallback: float = 0.0) -> float:
    valid: list[tuple[float, float]] = []
    for value, weight in values:
        parsed = _num(value)
        if np.isfinite(parsed) and weight > 0:
            valid.append((parsed, weight))
    if not valid:
        return _score(fallback)
    total_weight = sum(weight for _, weight in valid)
    return _score(sum(value * weight for value, weight in valid) / total_weight)


def _optional_score(value: Any) -> float:
    """Finite 0-100 score or NaN when the factor is genuinely unobserved."""
    parsed = _num(value, np.nan)
    return float(np.clip(parsed, 0.0, 100.0)) if np.isfinite(parsed) else np.nan


def _weighted_with_confidence(values: list[tuple[Any, float]], *, neutral: float = 50.0) -> tuple[float, float, float]:
    """Aggregate observed quality, then apply missing-data confidence once.

    Missing factors are neither negative (0) nor synthetic neutral observations
    (50).  They reduce model coverage; the resulting quality is shrunk once
    toward a neutral prior.  This prevents asymmetric missing-data penalties.
    """
    possible = sum(weight for _, weight in values if weight > 0)
    valid: list[tuple[float, float]] = []
    for value, weight in values:
        parsed = _num(value, np.nan)
        if np.isfinite(parsed) and weight > 0:
            valid.append((float(np.clip(parsed, 0.0, 100.0)), weight))
    if possible <= 0 or not valid:
        return float(neutral), float(neutral), 0.0
    observed = sum(weight for _, weight in valid)
    quality = sum(value * weight for value, weight in valid) / observed
    coverage = 100.0 * observed / possible
    adjusted = neutral + (coverage / 100.0) * (quality - neutral)
    return _score(adjusted, neutral), _score(quality, neutral), _score(coverage, 0.0)


def calculate_dashboard_scores(row: Mapping[str, Any]) -> dict[str, float]:
    flow = _weighted(
        [
            (row.get("broker_inventory_score"), 0.40),
            (row.get("smart_money_score"), 0.35),
            (row.get("absorption_score"), 0.15),
            (row.get("close_acceptance20_pct"), 0.10),
        ],
        fallback=row.get("smart_money_score", 0),
    )
    silent_accum = _weighted(
        [
            (row.get("holder_persistence_score"), 0.15),
            (row.get("inventory_dryness_score"), 0.12),
            (row.get("inventory_dryness_multiyear_score"), 0.15),
            (row.get("inventory_cycle_score"), 0.15),
            (row.get("absorption_score"), 0.14),
            (row.get("close_acceptance20_pct"), 0.10),
            (100.0 - _score(row.get("distribution_score"), 100.0), 0.12),
            (row.get("broker_inventory_score"), 0.07),
        ],
        fallback=row.get("broker_inventory_score", 0),
    )
    momentum = _weighted(
        [
            (row.get("trend_score"), 0.70),
            (row.get("markup_quality_score"), 0.20),
            (row.get("relative_strength20_pct"), 0.10),
        ],
        fallback=row.get("trend_score", 0),
    )
    final_score = _score(row.get("emir_conviction_score"), 0.0)
    return {
        "emir_final_score": round(final_score, 1),
        "dashboard_flow_score": round(flow, 1),
        "dashboard_silent_accum_score": round(silent_accum, 1),
        "dashboard_momentum_score": round(momentum, 1),
        "dashboard_accumulation_dominance_pct": round(
            _weighted(
                [
                    (flow, 0.45),
                    (silent_accum, 0.35),
                    (100.0 - _score(row.get("distribution_score"), 100.0), 0.20),
                ]
            ),
            1,
        ),
    }


def _price_change(frame: pd.DataFrame) -> float:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame.columns:
        return np.nan
    closes = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(closes) < 2 or closes.iloc[-2] == 0:
        return np.nan
    return float((closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0)




def _is_bank_like(row: Mapping[str, Any]) -> bool:
    """Detect deposit-taking banks for which industrial-company ratios are not comparable.

    The public Yahoo proxy can still be used for growth/freshness, but OCF/FCF, current ratio,
    cash/debt and generic leverage must not create a false quality advantage for banks.
    """
    sector = str(row.get("sector") or "").upper()
    name = str(row.get("company_name") or "").upper()
    industry = str(row.get("industry") or row.get("industry_name") or "").upper()
    return bool("FINANCIAL" in sector and ("BANK" in name or "BANK" in industry or "BANKING" in industry))


def _growth_score(value: Any, *, revenue: bool) -> float:
    """Robust 0-100 mapping for YoY growth without rewarding absurd base effects."""
    v = _num(value, np.nan)
    if not np.isfinite(v):
        return 50.0
    if revenue:
        xp = np.array([-30.0, -10.0, 0.0, 10.0, 20.0, 30.0, 50.0, 80.0])
        fp = np.array([5.0, 20.0, 35.0, 55.0, 70.0, 85.0, 97.0, 100.0])
    else:
        xp = np.array([-50.0, -15.0, 0.0, 15.0, 30.0, 50.0, 80.0, 150.0])
        fp = np.array([5.0, 20.0, 35.0, 55.0, 70.0, 85.0, 97.0, 100.0])
    return float(np.interp(np.clip(v, xp[0], xp[-1]), xp, fp))


def _business_momentum(row: Mapping[str, Any]) -> tuple[float, str]:
    """Prioritise confirmed YTD growth over short-term price/flow behaviour."""
    q_count = int(_num(row.get("fundamental_ytd_quarters_count"), 0) or 0)
    rev_ytd = _num(row.get("revenue_growth_ytd_yoy_pct"), np.nan)
    earn_ytd = _num(row.get("earnings_growth_ytd_yoy_pct"), np.nan)
    rev_q = _num(row.get("revenue_growth_yoy_pct"), np.nan)
    earn_q = _num(row.get("earnings_growth_yoy_pct"), np.nan)
    consistency = str(row.get("fundamental_growth_consistency_state") or "YTD_NOT_AVAILABLE")

    if q_count >= 2 and (np.isfinite(rev_ytd) or np.isfinite(earn_ytd)):
        rev = _growth_score(rev_ytd, revenue=True)
        earn = _growth_score(earn_ytd, revenue=False)
        score = 0.60 * rev + 0.40 * earn
        basis = "YTD_CONFIRMED_BUSINESS_MOMENTUM"
    else:
        rev = _growth_score(rev_q, revenue=True)
        earn = _growth_score(earn_q, revenue=False)
        score = 0.60 * rev + 0.40 * earn
        basis = "LATEST_QUARTER_FALLBACK_MOMENTUM"

    adjustments = {
        "QUARTER_AND_YTD_CONFIRMED": 5.0,
        "TURNAROUND_INFLECTION_UNCONFIRMED": -10.0,
        "LATEST_QUARTER_DECELERATION_YTD_POSITIVE": -6.0,
        "QUARTER_YTD_DIVERGENCE_REVIEW": -7.0,
        "QUARTER_AND_YTD_WEAK": -15.0,
        "YTD_NOT_AVAILABLE": -3.0,
        "Q1_YTD_EQUALS_QUARTER": -2.0,
    }
    score += adjustments.get(consistency, 0.0)
    return _score(score), basis


def calculate_next_leader_score(row: Mapping[str, Any]) -> dict[str, Any]:
    # v1.9.6: Next Leader is intentionally business-heavy. Execution/flow remains a small
    # confirming overlay; it must not suppress a fundamentally accelerating compounder.
    # Keep gate values numeric, but preserve NaN for the quality aggregator so
    # genuinely missing factors are not silently scored as 0 or 50.
    fundamental_raw = _optional_score(row.get("fundamental_conversion_score"))
    fundamental = _score(fundamental_raw, 0.0)
    fundamental_cov = _score(row.get("fundamental_coverage_pct"))
    data_quality_raw = _optional_score(row.get("fundamental_data_quality_score"))
    data_quality = _score(data_quality_raw, fundamental_cov)
    story = _optional_score(row.get("story_runway_score"))
    conversion = _optional_score(row.get("financial_conversion_score"))
    alignment = _optional_score(row.get("issuer_alignment_score"))
    sector = _optional_score(row.get("sector_leadership_score"))
    smart = _optional_score(row.get("smart_money_score"))
    inventory = _optional_score(row.get("broker_inventory_score"))
    structure = _optional_score(row.get("market_structure_score"))
    liquidity = _optional_score(row.get("liquidity_score"))
    ownership = _optional_score(row.get("ownership_score"))
    momentum, momentum_basis = _business_momentum(row)

    bank_like = _is_bank_like(row)
    model_state = "STANDARD_CORPORATE_FUNDAMENTAL_MODEL"
    quality_flags: list[str] = []
    effective_fundamental = fundamental
    effective_data_quality = data_quality
    if bank_like:
        # Until bank-specific NPL/NIM/CAR/LDR/BOPO fields are ingested, generic OCF/FCF and
        # industrial solvency ratios cannot justify a top-quality rating.
        effective_fundamental = min(fundamental, 70.0)
        effective_data_quality = min(data_quality, 75.0)
        model_state = "BANK_GENERIC_PROXY_LIMITED"
        quality_flags.append("BANK_SPECIFIC_RISK_METRICS_NOT_MODELED")

    net_margin = _num(row.get("net_margin_ttm_pct"), np.nan)
    roe = _num(row.get("roe_ttm_pct"), np.nan)
    loss_making = bool((np.isfinite(net_margin) and net_margin < 0) or (np.isfinite(roe) and roe < 0))
    if loss_making:
        momentum = min(momentum, 60.0)
        quality_flags.append("LOSS_MAKING_GROWTH_REVIEW")

    raw, quality_pre_confidence, model_coverage = _weighted_with_confidence([
        (effective_fundamental if np.isfinite(fundamental_raw) else np.nan, 0.30),
        (momentum, 0.20),
        (effective_data_quality if np.isfinite(data_quality_raw) else np.nan, 0.05),
        (story, 0.12),
        (conversion, 0.09),
        (alignment, 0.06),
        (sector, 0.09),
        (smart, 0.03),
        (inventory, 0.02),
        (structure, 0.02),
        (liquidity, 0.01),
        (ownership, 0.01),
    ])

    penalty = 0.0
    if fundamental_cov < 55: penalty += 12.0
    if effective_data_quality < 55: penalty += 10.0
    cashflow_state = str(row.get("fundamental_cashflow_state") or "")
    # Cash-flow remains evidence-quality information, but missing cash flow no longer dominates
    # a business ranking; Real Money Gate separately enforces manual verification.
    if not bank_like and cashflow_state == "CASHFLOW_TTM_MISSING": penalty += 4.0
    if _score(row.get("fundamental_official_source_coverage_pct")) <= 0: penalty += 4.0

    freshness = str(row.get("fundamental_period_freshness_state") or "")
    if freshness == "AGING_QUARTERLY_PERIOD": penalty += 6.0
    elif freshness == "LAGGING_REPORTING_PERIOD": penalty += 10.0
    elif freshness in {"STALE_QUARTERLY_PERIOD", "STALE_RELATIVE_TO_UNIVERSE", "UNKNOWN_PERIOD"}: penalty += 15.0

    consistency = str(row.get("fundamental_growth_consistency_state") or "")
    if consistency == "TURNAROUND_INFLECTION_UNCONFIRMED": penalty += 10.0
    elif consistency == "LATEST_QUARTER_DECELERATION_YTD_POSITIVE": penalty += 8.0
    elif consistency == "QUARTER_YTD_DIVERGENCE_REVIEW": penalty += 6.0
    elif consistency == "QUARTER_AND_YTD_WEAK": penalty += 12.0
    elif consistency == "YTD_NOT_AVAILABLE": penalty += 2.0

    leverage = str(row.get("fundamental_leverage_risk_state") or "")
    if not bank_like:
        if leverage == "HIGH_LEVERAGE": penalty += 8.0
        elif leverage == "EXTREME_LEVERAGE": penalty += 16.0

    q_count = int(_num(row.get("fundamental_ytd_quarters_count"), 0) or 0)
    rev_ytd = _num(row.get("revenue_growth_ytd_yoy_pct"), np.nan)
    earn_ytd = _num(row.get("earnings_growth_ytd_yoy_pct"), np.nan)
    if q_count >= 2 and np.isfinite(rev_ytd) and np.isfinite(earn_ytd) and rev_ytd < 10.0 and earn_ytd >= 40.0:
        penalty += 4.0
        quality_flags.append("EARNINGS_LED_LOW_TOPLINE_REVIEW")
    if loss_making:
        penalty += 8.0

    business_quality_adjustment = 0.0
    if (
        q_count >= 2
        and consistency == "QUARTER_AND_YTD_CONFIRMED"
        and np.isfinite(rev_ytd) and rev_ytd >= 20.0
        and np.isfinite(earn_ytd) and earn_ytd >= 25.0
        and not loss_making
    ):
        business_quality_adjustment += 3.0
        quality_flags.append("BROAD_BASED_YTD_GROWTH_CONFIRMED")

    # Distribution belongs mainly to Execution. Only extreme distribution gets a modest
    # business-ranking haircut; moderate distribution must not erase compounder quality.
    distribution = _score(row.get("distribution_score"))
    if distribution >= 70: penalty += 5.0

    if bank_like:
        penalty += 8.0

    score = _score(raw + business_quality_adjustment - penalty)
    fundamental_state = str(row.get("fundamental_state") or "")
    finite_fundamental = np.isfinite(_num(row.get("fundamental_conversion_score")))
    evidence_minimum = bool(fundamental_cov >= 35 and data_quality >= 35 and finite_fundamental)
    fundamental_disqualified = fundamental_state in {"FUNDAMENTAL_WEAK", "PROVIDER_FAILED"} and fundamental < 42
    eligible = bool(evidence_minimum and not fundamental_disqualified)
    if not eligible:
        state = "NEXT_LEADER_NOT_QUALIFIED"
    elif score >= 70 and effective_fundamental >= 65 and effective_data_quality >= 60:
        state = "NEXT_LEADER_HIGH_CONVICTION"
    elif score >= 55 and effective_fundamental >= 55:
        state = "NEXT_LEADER_WATCH"
    else:
        state = "NEXT_LEADER_RESEARCH"
    return {
        "next_leader_score": round(score, 1),
        "next_leader_quality_pre_confidence": round(quality_pre_confidence, 1),
        "next_leader_model_coverage_pct": round(model_coverage, 1),
        "next_leader_state": state,
        "next_leader_eligible": eligible,
        "next_leader_penalty": round(penalty, 1),
        "next_leader_business_momentum_score": round(momentum, 1),
        "next_leader_business_momentum_basis": momentum_basis,
        "next_leader_business_quality_adjustment": round(business_quality_adjustment, 1),
        "next_leader_sector_model_state": model_state,
        "next_leader_quality_flags": " | ".join(quality_flags) or "NONE",
    }

def select_next_leaders(radar: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if radar is None or radar.empty:
        return pd.DataFrame()
    local = radar.copy()
    if "next_leader_score" not in local.columns:
        local = enrich_dashboard_scores(local)
    local = local[local.get("next_leader_eligible", pd.Series(False, index=local.index)).fillna(False)].copy()
    if local.empty:
        return local
    local = local.sort_values(["next_leader_score", "fundamental_conversion_score", "story_runway_score", "sector_leadership_score", "liquidity_score"], ascending=[False, False, False, False, False], na_position="last").head(max(1, int(limit))).copy()
    if "next_leader_rank" in local.columns:
        local = local.drop(columns=["next_leader_rank"])
    local.insert(0, "next_leader_rank", range(1, len(local)+1))
    return local


def calculate_real_money_candidate_score(row: Mapping[str, Any]) -> dict[str, Any]:
    """Rank only candidates that already passed the real-money hard gates.

    This score is deliberately execution-heavy, but retains business quality so a liquid
    setup with weak fundamentals cannot dominate purely on price/flow. It is not an
    authorization gate; ``real_money_entry_candidate`` remains the prerequisite.
    """
    next_leader = _score(row.get("next_leader_score"), 0)
    final_score = _score(row.get("emir_final_score", row.get("emir_conviction_score")), 0)
    silent = _score(row.get("dashboard_silent_accum_score"), 0)
    flow = _score(row.get("dashboard_flow_score"), 0)
    structure = _score(row.get("market_structure_score"), 0)
    liquidity = _score(row.get("liquidity_score"), 0)
    fundamental = _score(row.get("fundamental_conversion_score"), 0)
    smart = _score(row.get("smart_money_score"), 0)
    distribution = _score(row.get("distribution_score"), 100)
    rr_values = [
        _num(row.get("rr_tp1_at_entry_high"), np.nan),
        _num(row.get("breakout_rr_tp1"), np.nan),
        _num(row.get("rr_tp1"), np.nan),
    ]
    rr_valid = [value for value in rr_values if np.isfinite(value)]
    rr = max(rr_valid) if rr_valid else np.nan
    rr_score = _score((rr / 3.0) * 100.0 if np.isfinite(rr) else 0.0)
    score = _weighted([
        (next_leader, 0.15),
        (final_score, 0.10),
        (silent, 0.14),
        (flow, 0.11),
        (structure, 0.14),
        (liquidity, 0.10),
        (fundamental, 0.10),
        (smart, 0.05),
        (100.0 - distribution, 0.05),
        (rr_score, 0.06),
    ])
    return {
        "real_money_candidate_score": round(score, 1),
        "real_money_rr_score": round(rr_score, 1),
    }


def select_real_money_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Return actionable manual/direct candidates only. WAIT/NO_EDGE never enter this table."""
    if radar is None or radar.empty:
        return pd.DataFrame()
    local = radar.copy()
    if "real_money_candidate_score" not in local.columns:
        local = enrich_dashboard_scores(local)
    entry_mask = local.get("real_money_entry_candidate", pd.Series(False, index=local.index)).fillna(False)
    local = local.loc[entry_mask].copy()
    if local.empty:
        return local
    local["_ready_priority"] = local.get("real_money_ready", pd.Series(False, index=local.index)).fillna(False).map({True: 0, False: 1})
    for col in ("real_money_candidate_score", "next_leader_score", "liquidity_score", "emir_final_score"):
        if col not in local.columns:
            local[col] = np.nan
    local = local.sort_values(
        ["_ready_priority", "real_money_candidate_score", "next_leader_score", "liquidity_score", "emir_final_score"],
        ascending=[True, False, False, False, False],
        na_position="last",
    ).drop(columns=["_ready_priority"]).head(max(0, int(limit))).reset_index(drop=True)
    local.insert(0, "real_money_rank", range(1, len(local) + 1))
    return local

def enrich_dashboard_scores(radar: pd.DataFrame, frames: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    if radar.empty:
        return radar.copy()
    frames = frames or {}
    local = radar.copy()
    scores = local.apply(lambda row: pd.Series(calculate_dashboard_scores(row)), axis=1)
    for column in scores.columns:
        local[column] = scores[column]
    leader_scores = local.apply(lambda row: pd.Series(calculate_next_leader_score(row)), axis=1)
    for column in leader_scores.columns:
        local[column] = leader_scores[column]

    # Banks require bank-specific asset-quality/capital/liquidity metrics before being presented
    # as real-money candidates. Generic corporate proxy ratios are sufficient for research only.
    bank_limited = local.get("next_leader_sector_model_state", pd.Series("", index=local.index)).eq("BANK_GENERIC_PROXY_LIMITED")
    bank_metric_coverage = pd.to_numeric(local.get("bank_metric_coverage_pct", pd.Series(0.0, index=local.index)), errors="coerce").fillna(0.0)
    bank_block = bank_limited & bank_metric_coverage.lt(60.0)
    if bank_block.any():
        for idx in local.index[bank_block]:
            existing = str(local.at[idx, "real_money_block_reasons"] if "real_money_block_reasons" in local.columns else "")
            reasons = [part.strip() for part in existing.split("|") if part.strip() and part.strip() != "NONE"]
            if "BANK_SPECIFIC_RISK_METRICS_NOT_MODELED" not in reasons:
                reasons.append("BANK_SPECIFIC_RISK_METRICS_NOT_MODELED")
            local.at[idx, "real_money_block_reasons"] = " | ".join(reasons)
            local.at[idx, "real_money_candidate"] = False
            local.at[idx, "real_money_entry_candidate"] = False
            local.at[idx, "real_money_ready"] = False
            local.at[idx, "real_money_gate_state"] = "REAL_MONEY_BLOCKED"
            local.at[idx, "entry_authorization_state"] = "NO_ENTRY_AUTHORIZATION"
            local.at[idx, "guarded_position_cap_after_manual_confirmation_pct"] = 0.0

    real_money_scores = local.apply(lambda row: pd.Series(calculate_real_money_candidate_score(row)), axis=1)
    for column in real_money_scores.columns:
        local[column] = real_money_scores[column]
    local["dashboard_price_change_pct"] = [
        _price_change(frames.get(str(ticker), pd.DataFrame())) for ticker in local.get("ticker", pd.Series(dtype=str))
    ]
    local["dashboard_rank_eligible"] = ~local.get("emir_decision_state", pd.Series(index=local.index, dtype=str)).isin(BLOCKED_STATES)
    local["dashboard_recommendation"] = local.apply(lambda row: recommendation_meta(str(row.get("emir_decision_state", "")))[0], axis=1)
    local["dashboard_state_priority"] = local.get("emir_decision_state", pd.Series(index=local.index, dtype=str)).map(STATE_PRIORITY).fillna(90).astype(int)
    local["dashboard_exclusion_reason"] = np.where(local["dashboard_rank_eligible"], "ELIGIBLE", "BLOCKED_DECISION_STATE")
    ranked = local.loc[local["dashboard_rank_eligible"].fillna(False)].copy()
    if not ranked.empty:
        ranked["_deep"] = ranked.get("deep_review_state", pd.Series(index=ranked.index, dtype=str)).eq("DEEP_REVIEWED").map({True: 0, False: 1})
        for col in ("emir_final_score", "emir_evidence_coverage_pct", "dashboard_silent_accum_score", "dashboard_flow_score", "liquidity_score"):
            if col not in ranked.columns:
                ranked[col] = np.nan
        ranked = ranked.sort_values(["_deep", "dashboard_state_priority", "emir_final_score", "emir_evidence_coverage_pct", "dashboard_silent_accum_score", "dashboard_flow_score", "liquidity_score"], ascending=[True, True, False, False, False, False, False], na_position="last")
        rank_map = {idx: rank for rank, idx in enumerate(ranked.index, start=1)}
        local["dashboard_universe_rank"] = [rank_map.get(idx, np.nan) for idx in local.index]
    else:
        local["dashboard_universe_rank"] = np.nan
    leader_ranked = local.loc[local.get("next_leader_eligible", pd.Series(False, index=local.index)).fillna(False)].copy()
    if not leader_ranked.empty:
        leader_ranked = leader_ranked.sort_values(["next_leader_score", "fundamental_conversion_score", "story_runway_score", "sector_leadership_score", "liquidity_score"], ascending=[False, False, False, False, False], na_position="last")
        leader_map = {idx: rank for rank, idx in enumerate(leader_ranked.index, start=1)}
        local["next_leader_universe_rank"] = [leader_map.get(idx, np.nan) for idx in local.index]
    else:
        local["next_leader_universe_rank"] = np.nan
    return local


def select_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    if radar.empty:
        return radar.copy()
    local = radar.copy()
    if "emir_final_score" not in local.columns:
        local = enrich_dashboard_scores(local)
    local = local.loc[local["dashboard_rank_eligible"].fillna(False)].copy()
    if local.empty:
        return local
    local["_priority"] = local.get("emir_decision_state", pd.Series(index=local.index, dtype=str)).map(STATE_PRIORITY).fillna(90)
    local["_deep"] = local.get("deep_review_state", pd.Series(index=local.index, dtype=str)).eq("DEEP_REVIEWED").map({True: 0, False: 1})
    for _column in ("emir_evidence_coverage_pct", "dashboard_silent_accum_score", "dashboard_flow_score", "liquidity_score"):
        if _column not in local.columns:
            local[_column] = np.nan
    local = local.sort_values(
        ["_deep", "_priority", "emir_final_score", "emir_evidence_coverage_pct", "dashboard_silent_accum_score", "dashboard_flow_score", "liquidity_score"],
        ascending=[True, True, False, False, False, False, False],
        na_position="last",
    )
    local = local.drop(columns=["_priority", "_deep"]).head(max(0, int(limit))).reset_index(drop=True)
    local.insert(0, "dashboard_rank", range(1, len(local) + 1))
    return local


def recommendation_meta(state: str) -> tuple[str, str, str]:
    mapping = {
        "EMIR_READY_WITH_PRECISE_TRIGGER": ("BUY READY", "green", "Direct trigger terverifikasi"),
        "EMIR_AUTO_EOD_READY": ("BUY / EOD READY", "green", "Public/proxy evidence memenuhi gate EOD"),
        "EMIR_CORE_THESIS_READY_WAIT_IDX_INTEGRITY": ("WATCH BUY", "gold", "Thesis kuat; tunggu IDX integrity"),
        "EMIR_THESIS_READY_WAIT_BID_OFFER": ("WATCH BUY", "gold", "Thesis kuat; tunggu trigger langsung"),
        "EMIR_WATCH_INVENTORY_COLLECTION": ("WATCH ACCUM", "gold", "Inventory/akumulasi perlu diteruskan"),
        "EMIR_WAIT_REACCUMULATION": ("WAIT REACCUM", "orange", "Markup advanced; tunggu pullback/base atau breakout-retest"),
        "EMIR_WAIT_NARRATIVE": ("WAIT NARRATIVE", "orange", "Flow ada; katalis belum cukup"),
        "EMIR_WAIT_MONEY_FLOW": ("WAIT FLOW", "orange", "Cerita ada; money flow belum mengonfirmasi"),
        "EMIR_WAIT_FUNDAMENTAL_CONVERSION": ("WAIT FUNDAMENTAL", "orange", "Bisnis belum mengonversi cerita/flow"),
        "EMIR_FUNDAMENTAL_EVIDENCE_PENDING": ("FUNDAMENTAL PENDING", "blue", "Laporan/growth/cash-flow belum cukup"),
        "EMIR_EVIDENCE_PENDING": ("EVIDENCE PENDING", "blue", "Evidence belum memadai"),
        "EMIR_NO_EDGE_YET": ("WATCH ONLY", "blue", "Belum ada edge yang jelas"),
        "EMIR_RADAR_ONLY_NOT_DEEP_REVIEWED": ("RADAR ONLY", "blue", "Belum deep review"),
    }
    return mapping.get(state, ("RESEARCH ONLY", "blue", "Belum memenuhi gate eksekusi"))


def _fmt_score(value: Any) -> str:
    parsed = _num(value)
    return "—" if not np.isfinite(parsed) else f"{parsed:.0f}"


def _fmt_rupiah(value: Any) -> str:
    parsed = _num(value)
    if not np.isfinite(parsed) or parsed <= 0:
        return "—"
    return "Rp" + f"{parsed:,.0f}".replace(",", ".")


def _fmt_pct(value: Any, signed: bool = False) -> str:
    parsed = _num(value)
    if not np.isfinite(parsed):
        return "—"
    return f"{parsed:+.2f}%" if signed else f"{parsed:.1f}%"


def _stars(value: Any) -> str:
    score = _score(value)
    filled = int(np.clip(np.round(score / 20.0), 0, 5))
    return "★" * filled + "☆" * (5 - filled)


def _risk_badges(row: Mapping[str, Any]) -> list[str]:
    badges: list[str] = []
    if str(row.get("execution_capacity_state", "")) == "EXECUTION_CAPACITY_OK":
        badges.append("EXECUTABLE CAPACITY")
    if _score(row.get("liquidity_score")) >= 60:
        badges.append("LIQUID")
    if str(row.get("sector_rrg_state", "")) in {"LEADING", "IMPROVING"}:
        badges.append(str(row.get("sector_rrg_state")))
    lifecycle = str(row.get("emir_lifecycle", "")).replace("_", " ")
    if lifecycle:
        badges.append(lifecycle)
    return badges[:4]


def _reason_lines(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    narrative = _score(row.get("narrative_score"))
    flow = _score(row.get("dashboard_flow_score"))
    silent = _score(row.get("dashboard_silent_accum_score"))
    structure = _score(row.get("market_structure_score"))
    momentum = _score(row.get("dashboard_momentum_score"))
    fundamental = _score(row.get("fundamental_conversion_score"))

    reasons.append("narasi kuat dan terkonversi" if narrative >= 70 else "narasi mulai terbentuk" if narrative >= 50 else "narasi masih lemah")
    reasons.append("flow mendukung" if flow >= 65 else "flow perlu konfirmasi")
    reasons.append("akumulasi senyap terdeteksi" if silent >= 65 else "silent accumulation belum kuat")
    reasons.append("struktur bullish/terjaga" if structure >= 65 else "struktur belum matang")
    reasons.append("momentum positif" if momentum >= 60 else "momentum belum meyakinkan")
    reasons.append("future fundamental mendukung" if fundamental >= 65 else "konversi future fundamental belum terbukti")
    return reasons


def _esc(value: Any) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def _factor_rows(row: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for label, field in FACTOR_FIELDS.items():
        score = _score(row.get(field))
        rows.append(
            f'<div class="es-factor"><span>{_esc(label)}</span>'
            f'<div class="es-bar"><i style="width:{score:.1f}%"></i></div><b>{score:.0f}</b></div>'
        )
    return "".join(rows)


def _report_rows(row: Mapping[str, Any]) -> str:
    return "".join(
        f'<div class="es-report-row"><span>{_esc(label)}</span><b>{_esc(_stars(row.get(field)))}</b></div>'
        for label, field in FACTOR_FIELDS.items()
        if label in {"Narrative", "Flow", "Silent Accum", "Structure", "Momentum", "Future Fundamental"}
    )


def _plan_rows(row: Mapping[str, Any]) -> str:
    accum_rr_high = _num(row.get("rr_tp2_at_entry_high"))
    accum_rr_mid = _num(row.get("rr_tp2"))
    breakout_rr = _num(row.get("breakout_rr_tp2"))
    plan = [
        ("ACCUM Entry", f"{_fmt_rupiah(row.get('entry_low'))} – {_fmt_rupiah(row.get('entry_high'))}", "entry"),
        ("ACCUM Stop", "< " + _fmt_rupiah(row.get("stop_loss")), "stop"),
        ("ACCUM TP2", _fmt_rupiah(row.get("tp2")), "target"),
        ("ACCUM RR @ high", f"1 : {accum_rr_high:.2f}" if np.isfinite(accum_rr_high) else "—", "rr"),
        ("ACCUM RR @ mid", f"1 : {accum_rr_mid:.2f}" if np.isfinite(accum_rr_mid) else "—", "rr"),
        ("BREAKOUT Entry", _fmt_rupiah(row.get("breakout_entry", row.get("trigger"))), "trigger"),
        ("BREAKOUT Stop", "< " + _fmt_rupiah(row.get("breakout_stop_loss")), "stop"),
        ("BREAKOUT TP2", _fmt_rupiah(row.get("breakout_tp2")), "target"),
        ("BREAKOUT RR", f"1 : {breakout_rr:.2f}" if np.isfinite(breakout_rr) else "—", "rr"),
    ]
    return "".join(
        f'<div class="es-plan-row {kind}"><span>{_esc(label)}</span><b>{_esc(value)}</b></div>'
        for label, value, kind in plan
    )


def _card_html(row: Mapping[str, Any], rank: int) -> str:
    recommendation, tone, recommendation_note = recommendation_meta(str(row.get("emir_decision_state", "")))
    score = _score(row.get("emir_final_score"))
    accumulation = _score(row.get("dashboard_accumulation_dominance_pct"))
    ticker = str(row.get("ticker", "")).replace(".JK", "")
    company = row.get("company_name") or ""
    sector = row.get("sector") or "SECTOR UNKNOWN"
    change = _num(row.get("dashboard_price_change_pct"))
    change_class = "up" if np.isfinite(change) and change >= 0 else "down"
    change_text = _fmt_pct(change, signed=True)
    confidence = "HIGH" if score >= 75 else "GOOD" if score >= 60 else "MODERATE" if score >= 45 else "LOW"
    badges = "".join(f'<span class="es-chip">{_esc(item)}</span>' for item in _risk_badges(row))
    reasons = "".join(f'<li>{_esc(item)}</li>' for item in _reason_lines(row))
    evidence_type = str(row.get("broker_inventory_evidence_type") or "OHLCV_PROXY")
    flow_note = "DIRECT BROKER EVIDENCE" if "DIRECT" in evidence_type else "OHLCV PROXY — BUKAN IDENTITAS BROKER"
    summary = row.get("why_now") or row.get("thesis_statement") or recommendation_note
    risk_flags = str(row.get("risk_flags") or "NONE")
    top_color = {1: "rank1", 2: "rank2", 3: "rank3"}.get(rank, "rank3")
    return f"""
    <section class="es-card {top_color}">
      <div class="es-card-head">
        <div class="es-rank"><small>TOP</small><strong>{rank}</strong></div>
        <div class="es-identity"><h2>{_esc(ticker)}</h2><p>{_esc(company)}</p><em>{_esc(sector)}</em><div>{badges}</div></div>
        <div class="es-score"><span>FINAL SCORE</span><strong>{score:.0f}</strong><small>/100</small></div>
        <div class="es-rec {tone}"><span>REKOMENDASI</span><strong>{_esc(recommendation)}</strong><small>{_esc(recommendation_note)}</small></div>
        <div class="es-price"><span>HARGA SAAT INI</span><strong>{_fmt_rupiah(row.get('last_price'))}</strong><small class="{change_class}">{_esc(change_text)}</small></div>
      </div>
      <div class="es-grid-main">
        <div class="es-panel"><h3>PLAN TRADING</h3>{_plan_rows(row)}</div>
        <div class="es-panel"><h3>FAKTOR SCANNER</h3>{_factor_rows(row)}</div>
        <div class="es-panel es-flow"><h3>FLOW / INVENTORY</h3>
          <div class="es-gauge" style="--pct:{accumulation:.0f}%"><div><strong>{accumulation:.0f}%</strong><small>AKUMULASI</small></div></div>
          <div class="es-flow-stats">
            <span>Accum days <b>{_fmt_score(row.get('accumulation_days20'))}</b></span>
            <span>Absorption <b>{_fmt_score(row.get('absorption_days20'))}</b></span>
            <span>Distribution <b>{_fmt_score(row.get('distribution_days20'))}</b></span>
            <span>Inventory <b>{_fmt_score(row.get('broker_inventory_score'))}</b></span>
          </div><p>{_esc(flow_note)}</p>
        </div>
      </div>
      <div class="es-grid-bottom">
        <div class="es-panel"><h3>REPORT CARD</h3>{_report_rows(row)}</div>
        <div class="es-panel"><h3>RINGKASAN ALASAN</h3><ul class="es-reasons">{reasons}</ul></div>
        <div class="es-panel es-highlight"><h3>HIGHLIGHT / KESIMPULAN</h3><p>{_esc(summary)}</p>
          <div class="es-confidence {tone}">CONFIDENCE {confidence}</div>
          <small>Risk flags: {_esc(risk_flags)}</small>
        </div>
      </div>
    </section>
    """


def render_top3_dashboard_html(
    top3: pd.DataFrame,
    *,
    scan_id: str = "",
    as_of: Any = "",
    market_regime: str = "",
) -> str:
    cards = "".join(_card_html(row, int(row.get("dashboard_rank", index + 1))) for index, row in top3.iterrows())
    return f"""
    <style>
    .es-wrap{{font-family:Inter,Arial,sans-serif;color:#eaf7ff;background:linear-gradient(180deg,#06111e,#071827);padding:18px;border:1px solid #164968;border-radius:18px}}
    .es-title{{text-align:center;margin:0;font-size:clamp(28px,5vw,54px);letter-spacing:.5px}} .es-title b{{color:#48e89b}}
    .es-method{{margin:12px auto 18px;max-width:1100px;text-align:center;padding:12px;border:1px solid #1b5878;border-radius:12px;background:#071421}}
    .es-method strong{{color:#56efad}} .es-method p{{margin:7px 0 0;color:#a9c6d8;font-size:13px}}
    .es-card{{margin:18px 0;padding:14px;background:linear-gradient(135deg,#071a2b,#061320);border:1px solid #1d5e7e;border-radius:17px;box-shadow:0 12px 32px rgba(0,0,0,.28)}}
    .es-card.rank1{{border-color:#20c979}} .es-card.rank2{{border-color:#b9a839}} .es-card.rank3{{border-color:#3486d7}}
    .es-card-head{{display:grid;grid-template-columns:74px minmax(180px,2fr) minmax(130px,.8fr) minmax(160px,1fr) minmax(150px,1fr);gap:12px;align-items:stretch}}
    .es-rank{{display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:12px;background:linear-gradient(180deg,#17b769,#08613e);font-weight:800}} .rank2 .es-rank{{background:linear-gradient(180deg,#b59c31,#665514)}} .rank3 .es-rank{{background:linear-gradient(180deg,#2b88d6,#164577)}}
    .es-rank small{{font-size:14px}} .es-rank strong{{font-size:42px;line-height:1}}
    .es-identity,.es-score,.es-rec,.es-price{{padding:12px;border-radius:12px;background:#081c2c;border:1px solid #1a4a64}}
    .es-identity h2{{font-size:34px;margin:0}} .es-identity p{{margin:2px 0;color:#c4d9e5;font-size:13px}} .es-identity em{{font-style:normal;color:#59e5a5;font-weight:700;font-size:12px}}
    .es-chip{{display:inline-block;margin:7px 5px 0 0;padding:3px 7px;border:1px solid #2c6b82;border-radius:20px;color:#b6d7e7;font-size:10px}}
    .es-score,.es-rec,.es-price{{text-align:center;display:flex;flex-direction:column;justify-content:center}} .es-score span,.es-rec span,.es-price span{{font-size:11px;color:#a9c8d8;font-weight:700}}
    .es-score strong{{font-size:45px;color:#74f7bd;line-height:1}} .es-score small{{color:#a9c8d8}}
    .es-rec strong{{font-size:23px;line-height:1.15;margin:6px 0}} .es-rec.green strong{{color:#53ed9c}} .es-rec.gold strong{{color:#ffd85a}} .es-rec.orange strong{{color:#ff9a4c}} .es-rec.blue strong{{color:#7fc7ff}}
    .es-rec small{{font-size:10px;color:#abc3d2}} .es-price strong{{font-size:25px;margin:6px 0}} .es-price small.up{{color:#54ed9e}} .es-price small.down{{color:#ff725f}}
    .es-grid-main{{display:grid;grid-template-columns:1fr 1.25fr 1fr;gap:12px;margin-top:12px}} .es-grid-bottom{{display:grid;grid-template-columns:.9fr 1.15fr 1fr;gap:12px;margin-top:12px}}
    .es-panel{{background:#071a2a;border:1px solid #174965;border-radius:12px;padding:12px;min-width:0}} .es-panel h3{{font-size:13px;color:#a9d8ef;text-align:center;margin:0 0 10px}}
    .es-plan-row{{display:flex;justify-content:space-between;gap:8px;padding:5px 0;border-bottom:1px solid rgba(93,151,177,.15);font-size:12px}} .es-plan-row b{{color:#dcecf4}} .es-plan-row.stop b{{color:#ff6c5d}} .es-plan-row.target b{{color:#60ec9f}}
    .es-factor{{display:grid;grid-template-columns:110px 1fr 28px;align-items:center;gap:8px;font-size:11px;margin:7px 0}} .es-bar{{height:8px;background:#102d3b;border-radius:10px;overflow:hidden}} .es-bar i{{display:block;height:100%;background:linear-gradient(90deg,#2fc47c,#79f5b8)}}
    .es-flow{{text-align:center}} .es-gauge{{--pct:50;width:118px;height:118px;margin:4px auto 10px;border-radius:50%;background:conic-gradient(#37d486 var(--pct),#ef604d 0);display:grid;place-items:center;position:relative}} .es-gauge:before{{content:"";width:82px;height:82px;border-radius:50%;background:#071a2a;position:absolute}} .es-gauge div{{position:relative;z-index:1;display:flex;flex-direction:column}} .es-gauge strong{{font-size:25px}} .es-gauge small{{font-size:9px}}
    .es-flow-stats{{display:grid;grid-template-columns:1fr 1fr;gap:5px;text-align:left;font-size:10px}} .es-flow-stats span{{padding:5px;background:#0a2233;border-radius:7px}} .es-flow p{{font-size:9px;color:#88aab9;margin:9px 0 0}}
    .es-report-row{{display:flex;justify-content:space-between;padding:4px 0;font-size:11px;border-bottom:1px solid rgba(93,151,177,.14)}} .es-report-row b{{color:#ffd052;letter-spacing:1px}}
    .es-reasons{{list-style:none;margin:0;padding:0;font-size:11px}} .es-reasons li{{padding:4px 0}} .es-reasons li:before{{content:"✓";color:#4aed94;font-weight:800;margin-right:7px}}
    .es-highlight p{{font-size:12px;line-height:1.5;color:#d5e6ef}} .es-highlight small{{display:block;margin-top:8px;color:#839fac;font-size:9px;word-break:break-word}}
    .es-confidence{{text-align:center;padding:7px;border-radius:7px;font-size:11px;font-weight:800;background:#12334a}} .es-confidence.green{{color:#54efa0;border:1px solid #25ba73}} .es-confidence.gold{{color:#ffe16d;border:1px solid #a88928}} .es-confidence.orange{{color:#ffac6b;border:1px solid #b56327}} .es-confidence.blue{{color:#8dceff;border:1px solid #3478aa}}
    .es-footer{{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:16px;padding:10px;border-top:1px solid #174965;color:#8facbb;font-size:10px}}
    @media(max-width:900px){{.es-card-head{{grid-template-columns:56px 1fr 105px}} .es-rec,.es-price{{grid-column:span 1}} .es-identity h2{{font-size:28px}} .es-grid-main,.es-grid-bottom{{grid-template-columns:1fr}}}}
    @media(max-width:580px){{.es-wrap{{padding:9px}} .es-card{{padding:9px}} .es-card-head{{grid-template-columns:48px 1fr}} .es-score,.es-rec,.es-price{{grid-column:span 1}} .es-score strong{{font-size:36px}} .es-rank strong{{font-size:32px}} .es-factor{{grid-template-columns:95px 1fr 25px}}}}
    </style>
    <div class="es-wrap">
      <h1 class="es-title">TOP 3 <b>EMIR-STYLE SCANNER</b></h1>
      <div class="es-method"><strong>Kriteria Scanner Emir Style</strong><br>Narrative • Flow • Silent Accum • Smart Money • Future Fundamental • Struktur • Momentum • Liquidity
      <p>Final Score memakai <b>Emir Conviction Score</b> yang sudah ada. Flow dan silent accumulation adalah proxy transparan bila direct broker data tidak tersedia.</p></div>
      {cards if cards else '<div class="es-panel">Tidak ada kandidat yang lolos filter Top 3. Kandidat blocked/reject tidak dipaksakan masuk.</div>'}
      <div class="es-footer"><span>Scan ID: {_esc(scan_id)}</span><span>As-of: {_esc(as_of)}</span><span>Market regime: {_esc(market_regime)}</span><span>Decision support — bukan eksekusi otomatis</span></div>
    </div>
    """


__all__ = [
    "BLOCKED_STATES",
    "FACTOR_FIELDS",
    "calculate_dashboard_scores", "calculate_next_leader_score", "calculate_real_money_candidate_score",
    "enrich_dashboard_scores", "select_next_leaders", "select_real_money_top3",
    "select_top3",
    "recommendation_meta",
    "render_top3_dashboard_html",
]
