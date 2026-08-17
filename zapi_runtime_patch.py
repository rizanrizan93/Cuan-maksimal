from __future__ import annotations

"""Runtime hook: bounded ZAPI foreign-flow confirmation for Emir scanner.

ZAPI remains a confirmation layer. It never identifies brokers/beneficial owners,
never fabricates a cost basis, and never self-authorizes a real-money entry.
This patch also separates hard blockers from direct-evidence gaps and applies a
coverage-aware foreign-flow shock guard to execution timing.
"""

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from zapi_flow_enrichment import (
    ZAPI_FLOW_ENRICHMENT_VERSION,
    blend_emir_dashboard_output,
    enrich_emir_radar,
)

PATCH_VERSION = "1.1.0-emir-zapi-execution-guard"

_EVIDENCE_GAP_FLAGS = {
    "DIRECT_BID_OFFER_TRIGGER_MISSING_PROXY_USED",
    "DIRECT_BID_OFFER_TRIGGER_MISSING",
    "IDX_CRITICAL_FIELDS_UNKNOWN_NOT_VERIFIED",
    "IDX_INTEGRITY_EVIDENCE_MISSING",
    "IDX_DIRECT_INTEGRITY_MISSING_AUTO_PUBLIC_PROXY_USED",
}


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "ready"}


def _tokens(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.upper() in {"NONE", "NAN"}:
        return []
    return list(dict.fromkeys(part.strip() for part in text.split("|") if part.strip() and part.strip().upper() != "NONE"))


def _join_tokens(values: list[str]) -> str:
    clean = list(dict.fromkeys(value.strip() for value in values if str(value).strip()))
    return " | ".join(clean) if clean else "NONE"


def _recompute_real_money(owner: Any, frame: pd.DataFrame) -> pd.DataFrame:
    calculator = getattr(owner, "calculate_real_money_candidate_score", None)
    if not callable(calculator) or frame.empty:
        return frame
    out = frame.copy()
    try:
        scores = out.apply(lambda row: pd.Series(calculator(row)), axis=1)
        for column in scores.columns:
            out[column] = scores[column]
    except Exception:
        return frame
    return out


def _adjust_cost_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "smart_money_cost_confidence_pct" not in frame.columns:
        return frame
    out = frame.copy()
    deltas: list[float] = []
    for idx, row in out.iterrows():
        evidence = str(row.get("smart_money_cost_evidence_type") or "").upper()
        if "DIRECT_BROKER" in evidence:
            deltas.append(0.0)
            continue
        score = pd.to_numeric(pd.Series([row.get("zapi_smart_money_confirmation_score")]), errors="coerce").iloc[0]
        cov = pd.to_numeric(pd.Series([row.get("zapi_foreign_flow_coverage_pct")]), errors="coerce").iloc[0]
        base = pd.to_numeric(pd.Series([row.get("smart_money_cost_confidence_pct")]), errors="coerce").iloc[0]
        if not (np.isfinite(score) and np.isfinite(cov) and np.isfinite(base) and cov > 0):
            deltas.append(0.0)
            continue
        directional = float(np.clip((score - 50.0) / 50.0, -1.0, 1.0))
        delta = float(np.clip(8.0 * directional * cov / 100.0, -6.0, 8.0))
        out.at[idx, "smart_money_cost_confidence_pct"] = round(float(np.clip(base + delta, 0.0, 75.0)), 1)
        deltas.append(delta)
    out["zapi_smart_money_cost_confidence_delta"] = deltas
    return out


def _apply_foreign_shock_guard(frame: pd.DataFrame) -> pd.DataFrame:
    """Use ZAPI for execution timing only when coverage is sufficient.

    One-day selling shocks do not rewrite the business thesis or final score.
    A severe one-day shock with constructive 5D/20D history requires absorption
    or reclaim confirmation. Persistent multi-window distribution demotes an
    otherwise actionable candidate to WAIT_TIMING. Positive ZAPI flow can only
    confirm; it can never promote READY by itself.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    states: list[str] = []
    severities: list[float] = []
    actions: list[str] = []
    reasons: list[str] = []
    pre_gate: list[str] = []
    pre_ready: list[bool] = []

    for idx, row in out.iterrows():
        gate_before = str(row.get("real_money_gate_state") or "")
        ready_before = _truthy(row.get("real_money_ready"))
        pre_gate.append(gate_before)
        pre_ready.append(ready_before)

        coverage = _finite(row.get("zapi_foreign_flow_coverage_pct"), np.nan)
        one_day = _finite(row.get("zapi_foreign_net_participation_1d"), np.nan)
        five_day = _finite(row.get("zapi_foreign_net_participation_5d"), np.nan)
        twenty_day = _finite(row.get("zapi_foreign_net_participation_20d"), np.nan)
        foreign_state = str(row.get("zapi_foreign_state") or "").upper()

        state = "NO_EXECUTION_OVERRIDE"
        severity = 0.0
        action = "KEEP_EXISTING_GATE"
        reason = "ZAPI confirmation does not alter authorization."

        if not np.isfinite(coverage) or coverage < 60 or not np.isfinite(one_day):
            state = "ZAPI_INSUFFICIENT_OR_STALE_FOR_EXECUTION_GUARD"
            reason = "Foreign-flow coverage below execution-guard threshold; fail-soft and keep existing gate."
        else:
            if one_day <= -0.15:
                severity = 100.0
                shock_label = "EXTREME"
            elif one_day <= -0.08:
                severity = 75.0
                shock_label = "SEVERE"
            elif one_day <= -0.04:
                severity = 45.0
                shock_label = "MODERATE"
            else:
                shock_label = "NONE"

            persistent_distribution = bool(
                shock_label in {"SEVERE", "EXTREME"}
                and (
                    (np.isfinite(five_day) and five_day <= -0.01)
                    or (np.isfinite(twenty_day) and twenty_day <= -0.01)
                    or foreign_state == "NET_DISTRIBUTION"
                )
            )

            if persistent_distribution:
                state = "PERSISTENT_FOREIGN_DISTRIBUTION_WAIT"
                action = "WAIT_FLOW_STABILIZATION_AND_RECLAIM"
                reason = "Severe foreign selling is confirmed by a negative 5D/20D window or NET_DISTRIBUTION state."
                manual = _tokens(row.get("real_money_manual_conditions"))
                manual.append("ZAPI_FOREIGN_DISTRIBUTION_WAIT_STABILIZATION")
                out.at[idx, "real_money_manual_conditions"] = _join_tokens(manual)
                out.at[idx, "real_money_ready"] = False
                out.at[idx, "production_ready"] = False
                if _truthy(row.get("real_money_candidate")):
                    out.at[idx, "real_money_entry_candidate"] = False
                    out.at[idx, "real_money_gate_state"] = "REAL_MONEY_WAIT_TIMING"
                    out.at[idx, "entry_authorization_state"] = "WAIT_TIMING_NO_ENTRY"
                    out.at[idx, "production_tier"] = "WAIT_TIMING"
                    if "guarded_position_cap_after_manual_confirmation_pct" in out.columns:
                        out.at[idx, "guarded_position_cap_after_manual_confirmation_pct"] = 0.0
            elif shock_label in {"SEVERE", "EXTREME"}:
                state = f"{shock_label}_ONE_DAY_FOREIGN_SELL_SHOCK_RECLAIM_REQUIRED"
                action = "REQUIRE_ABSORPTION_OR_RECLAIM_BEFORE_ENTRY"
                reason = "Large 1D foreign sell shock while medium-window flow is not persistently negative; thesis stays intact but execution needs absorption/reclaim."
                manual = _tokens(row.get("real_money_manual_conditions"))
                manual.append("ZAPI_FOREIGN_SHOCK_REQUIRE_ABSORPTION_RECLAIM")
                out.at[idx, "real_money_manual_conditions"] = _join_tokens(manual)
                out.at[idx, "real_money_ready"] = False
                out.at[idx, "production_ready"] = False
                if _truthy(row.get("real_money_entry_candidate")):
                    out.at[idx, "real_money_gate_state"] = "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
                    out.at[idx, "entry_authorization_state"] = "MANUAL_CONFIRMATION_REQUIRED"
                    out.at[idx, "production_tier"] = "MANUAL_CONFIRMATION_REQUIRED"
            elif shock_label == "MODERATE":
                state = "MODERATE_ONE_DAY_FOREIGN_SELL_CAUTION"
                action = "MONITOR_ABSORPTION_NO_AUTHORIZATION_CHANGE"
                reason = "Moderate 1D foreign selling is diagnostic only; existing hard-risk and timing gates remain unchanged."
            elif one_day >= 0.08:
                state = "STRONG_ONE_DAY_FOREIGN_ACCUMULATION_CONFIRMATION_ONLY"
                action = "CONFIRMATION_ONLY_NO_READY_PROMOTION"
                reason = "Strong foreign accumulation confirms flow but cannot authorize an entry without the existing Emir gates."
            else:
                state = "FOREIGN_FLOW_NO_SHOCK"
                reason = "No material 1D foreign-flow shock; existing Emir authorization rules remain controlling."

        states.append(state)
        severities.append(severity)
        actions.append(action)
        reasons.append(reason)

    out["zapi_pre_guard_real_money_gate_state"] = pre_gate
    out["zapi_pre_guard_real_money_ready"] = pre_ready
    out["zapi_foreign_shock_state"] = states
    out["zapi_foreign_shock_severity"] = severities
    out["zapi_execution_flow_guard_state"] = actions
    out["zapi_execution_flow_guard_reason"] = reasons
    out["zapi_execution_guard_policy"] = "ZAPI_CANNOT_PROMOTE_READY_SELL_SHOCK_CAN_ONLY_DELAY_OR_REQUIRE_CONFIRMATION"
    return out


def _apply_gate_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate hard blockers, manual checks, and missing direct evidence."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    hard_counts: list[int] = []
    manual_counts: list[int] = []
    gap_counts: list[int] = []
    gaps_text: list[str] = []
    material_risks_text: list[str] = []
    gate_classes: list[str] = []
    explanations: list[str] = []

    for _, row in out.iterrows():
        blockers = _tokens(row.get("real_money_block_reasons"))
        manual = _tokens(row.get("real_money_manual_conditions"))
        risks = _tokens(row.get("risk_flags"))
        gaps = [flag for flag in risks if flag in _EVIDENCE_GAP_FLAGS]
        material_risks = [flag for flag in risks if flag not in _EVIDENCE_GAP_FLAGS]
        gate = str(row.get("real_money_gate_state") or "UNKNOWN")

        if gate == "REAL_MONEY_DIRECT_VERIFIED_READY":
            gate_class = "DIRECT_VERIFIED_READY"
        elif gate == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED":
            gate_class = "MANUAL_CONFIRMATION"
        elif gate == "REAL_MONEY_WAIT_TIMING":
            gate_class = "WAIT_TIMING"
        elif blockers:
            gate_class = "HARD_BLOCK"
        else:
            gate_class = "NOT_AUTHORIZED_OTHER"

        hard_counts.append(len(blockers))
        manual_counts.append(len(manual))
        gap_counts.append(len(gaps))
        gaps_text.append(_join_tokens(gaps))
        material_risks_text.append(_join_tokens(material_risks))
        gate_classes.append(gate_class)
        explanations.append(
            f"{gate_class}: hard_blockers={len(blockers)}, manual_conditions={len(manual)}, "
            f"direct_evidence_gaps={len(gaps)}, material_risk_flags={len(material_risks)}"
        )

    out["real_money_hard_block_count"] = hard_counts
    out["real_money_manual_condition_count"] = manual_counts
    out["real_money_evidence_gap_count"] = gap_counts
    out["real_money_evidence_gap_flags"] = gaps_text
    out["real_money_material_risk_flags"] = material_risks_text
    out["real_money_gate_class"] = gate_classes
    out["real_money_gate_explanation"] = explanations
    out["real_money_gate_audit_policy"] = "HARD_BLOCKERS_DISTINCT_FROM_MANUAL_CHECKS_AND_DIRECT_EVIDENCE_GAPS"
    return out


def _wrap_dashboard_scores(owner: Any) -> None:
    original = getattr(owner, "enrich_dashboard_scores", None)
    if not callable(original) or getattr(original, "__zapi_flow_confirmation_v1__", False):
        return

    @wraps(original)
    def wrapped(radar: pd.DataFrame, *args: Any, **kwargs: Any):
        enriched = radar
        try:
            if isinstance(radar, pd.DataFrame) and not radar.empty:
                enriched = enrich_emir_radar(radar)
        except Exception:
            enriched = radar
        out = original(enriched, *args, **kwargs)
        if isinstance(out, pd.DataFrame) and not out.empty:
            try:
                out = blend_emir_dashboard_output(out)
                out = _recompute_real_money(owner, out)
                out = _adjust_cost_confidence(out)
                out = _apply_foreign_shock_guard(out)
                out = _apply_gate_audit(out)
            except Exception:
                pass
        return out

    wrapped.__zapi_flow_confirmation_v1__ = True
    setattr(owner, "enrich_dashboard_scores", wrapped)


def install() -> dict[str, str]:
    import top3_dashboard_legacy
    import top3_dashboard

    # Install before runtime_integrity_patch. The existing ranking-contract wrapper
    # then remains outermost and ranks the already-ZAPI-confirmed result.
    _wrap_dashboard_scores(top3_dashboard_legacy)
    _wrap_dashboard_scores(top3_dashboard)
    return {
        "patch_version": PATCH_VERSION,
        "zapi_version": ZAPI_FLOW_ENRICHMENT_VERSION,
        "conviction_policy": "BOUNDED_PLUS_MINUS_2_5_POINT_FOREIGN_FLOW_CONFIRMATION",
        "smart_money_policy": "MAX_30_PERCENT_CONFIRMATION_WEIGHT_COVERAGE_AWARE",
        "smc_policy": "PRICE_STRUCTURE_PRIMARY_ZAPI_FLOW_CONFIRMATION_ONLY",
        "cost_policy": "ZAPI_MAY_ADJUST_PROXY_CONFIDENCE_NOT_COST_PRICE_DIRECT_BROKER_WINS",
        "execution_guard_policy": "SELL_SHOCK_CAN_DELAY_OR_REQUIRE_RECLAIM_ZAPI_CAN_NEVER_PROMOTE_READY",
        "gate_audit_policy": "HARD_BLOCKERS_SEPARATE_FROM_MANUAL_CONDITIONS_AND_DIRECT_EVIDENCE_GAPS",
        "identity_policy": "FOREIGN_FLOW_IS_NOT_BROKER_OR_BENEFICIAL_OWNER_IDENTITY",
    }


__all__ = [
    "PATCH_VERSION",
    "install",
    "_apply_foreign_shock_guard",
    "_apply_gate_audit",
]
