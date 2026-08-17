from __future__ import annotations

"""Post-ZAPI shadow calibration and authorization-tier refinement for Emir.

This module does not relax any hard-risk threshold. It separates a genuine hard
block from an unavailable direct-evidence condition and records pre/post ZAPI
values for later 5D/20D/60D forward calibration.
"""

from typing import Any

import numpy as np
import pandas as pd


POST_CALIBRATION_VERSION = "1.0.0-emir-zapi-post-calibration"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "ready", "verified"}


def enrich_emir_shadow(frame: pd.DataFrame) -> pd.DataFrame:
    """Persist auditable pre/post ZAPI values without changing any score."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()

    post_conviction = pd.to_numeric(
        out.get("emir_conviction_score", out.get("emir_final_score", pd.Series(np.nan, index=out.index))),
        errors="coerce",
    )
    conviction_delta = pd.to_numeric(
        out.get("zapi_emir_conviction_delta", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    pre_conviction = post_conviction - conviction_delta

    post_smart = pd.to_numeric(out.get("smart_money_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    zapi_smart = pd.to_numeric(
        out.get("zapi_smart_money_confirmation_score", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    weight = pd.to_numeric(
        out.get("zapi_smart_money_confirmation_weight_pct", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0).clip(0.0, 99.999) / 100.0
    denominator = (1.0 - weight).replace(0.0, np.nan)
    derived_pre_smart = (post_smart - weight * zapi_smart) / denominator
    derived_pre_smart = derived_pre_smart.where(zapi_smart.notna() & post_smart.notna(), post_smart)

    post_cost_conf = pd.to_numeric(
        out.get("smart_money_cost_confidence_pct", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    cost_delta = pd.to_numeric(
        out.get("zapi_smart_money_cost_confidence_delta", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    pre_cost_conf = post_cost_conf - cost_delta

    coverage = pd.to_numeric(
        out.get("zapi_foreign_flow_coverage_pct", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)

    out["zapi_shadow_pre_conviction_score"] = pre_conviction.round(3)
    out["zapi_shadow_post_conviction_score"] = post_conviction.round(3)
    out["zapi_shadow_conviction_delta"] = conviction_delta.round(3)
    out["zapi_shadow_pre_smart_money_score"] = derived_pre_smart.round(3)
    out["zapi_shadow_post_smart_money_score"] = post_smart.round(3)
    out["zapi_shadow_smart_money_delta"] = (post_smart - derived_pre_smart).round(3)
    out["zapi_shadow_pre_cost_confidence_pct"] = pre_cost_conf.round(3)
    out["zapi_shadow_post_cost_confidence_pct"] = post_cost_conf.round(3)
    out["zapi_shadow_cost_confidence_delta"] = cost_delta.round(3)
    out["zapi_shadow_calibration_state"] = np.where(coverage.gt(0.0), "PENDING_FORWARD_OUTCOME", "NO_ZAPI_EVIDENCE")
    out["zapi_shadow_forward_horizons"] = "5D|20D|60D"
    out["zapi_shadow_policy"] = "CAPTURE_PRE_POST_NOW_RECALIBRATE_ONLY_AFTER_FORWARD_OOS"
    out["zapi_post_calibration_version"] = POST_CALIBRATION_VERSION
    return out


def refine_emir_proxy_authorization_tier(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose a safe proxy/manual tier when only direct evidence is missing.

    A row is never promoted to direct READY here. If the legacy/base gate says
    BLOCKED while the post-audit hard-block count is zero and the row is already
    both a real-money candidate and an entry candidate, an evidence-gap-only
    block is converted to MANUAL_CONFIRMATION_REQUIRED. Genuine hard blockers
    remain untouched.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    tiers: list[str] = []
    eligible: list[bool] = []
    notes: list[str] = []

    for idx, row in out.iterrows():
        hard_count = int(_finite(row.get("real_money_hard_block_count"), 0.0) or 0)
        gap_count = int(_finite(row.get("real_money_evidence_gap_count"), 0.0) or 0)
        manual_count = int(_finite(row.get("real_money_manual_condition_count"), 0.0) or 0)
        candidate = _truthy(row.get("real_money_candidate"))
        entry_candidate = _truthy(row.get("real_money_entry_candidate"))
        gate = str(row.get("real_money_gate_state") or "UNKNOWN").upper()

        if hard_count > 0:
            tier = "HARD_BLOCKED"
            is_eligible = False
            note = "Genuine hard-risk blocker remains controlling; ZAPI/evidence-gap logic cannot override it."
        elif gate == "REAL_MONEY_DIRECT_VERIFIED_READY":
            tier = "DIRECT_VERIFIED_READY"
            is_eligible = True
            note = "Direct evidence and existing Emir gates authorize the candidate."
        elif candidate and entry_candidate and gap_count > 0:
            tier = "PROXY_EXECUTION_ELIGIBLE_MANUAL_CONFIRMATION"
            is_eligible = True
            note = "No hard blocker; direct evidence is incomplete, so manual confirmation is required before order placement."
            if gate == "REAL_MONEY_BLOCKED":
                out.at[idx, "real_money_gate_state"] = "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
                out.at[idx, "entry_authorization_state"] = "MANUAL_CONFIRMATION_REQUIRED"
                out.at[idx, "real_money_ready"] = False
                if "production_ready" in out.columns:
                    out.at[idx, "production_ready"] = False
                if "production_tier" in out.columns:
                    out.at[idx, "production_tier"] = "MANUAL_CONFIRMATION_REQUIRED"
        elif candidate and entry_candidate and manual_count > 0:
            tier = "MANUAL_CONFIRMATION_REQUIRED"
            is_eligible = True
            note = "No hard blocker, but one or more manual risk/data checks remain outstanding."
        elif candidate:
            tier = "WAIT_TIMING"
            is_eligible = False
            note = "Business candidate survives; execution timing is not yet actionable."
        else:
            tier = "RESEARCH_ONLY_OR_NOT_CANDIDATE"
            is_eligible = False
            note = "Not an authorized real-money candidate under the existing Emir hard gates."

        tiers.append(tier)
        eligible.append(is_eligible)
        notes.append(note)

    out["real_money_authorization_tier"] = tiers
    out["real_money_proxy_authorization_eligible"] = eligible
    out["real_money_authorization_tier_note"] = notes
    out["real_money_authorization_tier_policy"] = "DIRECT_GAPS_MAY_REQUIRE_MANUAL_CONFIRMATION_BUT_NEVER_OVERRIDE_GENUINE_HARD_BLOCKERS"
    out["zapi_post_calibration_version"] = POST_CALIBRATION_VERSION
    return out


__all__ = [
    "POST_CALIBRATION_VERSION",
    "enrich_emir_shadow",
    "refine_emir_proxy_authorization_tier",
]
