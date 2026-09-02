from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import narrative_flow_engine as engine
import top3_dashboard_legacy as dashboard
import zapi_runtime_patch as zapi_patch


def _strong_nonfundamental_row() -> dict[str, object]:
    return {
        "ticker": "TEST.JK",
        "story_runway_score": 82.0,
        "financial_conversion_score": 80.0,
        "narrative_coverage_pct": 90.0,
        "issuer_alignment_score": 80.0,
        "issuer_alignment_coverage_pct": 90.0,
        "sector_leadership_score": 82.0,
        "sector_context_coverage_pct": 90.0,
        "smart_money_score": 82.0,
        "smart_money_coverage_pct": 100.0,
        "broker_inventory_score": 78.0,
        "broker_inventory_coverage_pct": 80.0,
        "market_structure_score": 80.0,
        "liquidity_score": 80.0,
        "distribution_score": 10.0,
        "future_direct_forward_visibility_score": 78.0,
        "future_direct_forward_visibility_coverage_pct": 80.0,
        "future_fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
    }


def test_missing_fundamental_is_not_zero_or_next_leader_disqualifier() -> None:
    row = _strong_nonfundamental_row()
    row.update({
        "fundamental_conversion_score": np.nan,
        "fundamental_coverage_pct": 0.0,
        "fundamental_data_quality_score": np.nan,
        "fundamental_state": "PROVIDER_FAILED",
    })

    result = dashboard.calculate_next_leader_score(row)

    assert result["next_leader_fundamental_scoring_state"] == "MISSING_NOT_SCORED"
    assert bool(result["next_leader_eligible"])
    assert result["next_leader_state"] in {"NEXT_LEADER_WATCH", "NEXT_LEADER_RESEARCH"}
    assert "FUNDAMENTAL_MISSING_NOT_SCORED" in result["next_leader_quality_flags"]
    assert np.isfinite(float(result["next_leader_score"]))


def test_explicit_bad_fundamental_still_disqualifies() -> None:
    row = _strong_nonfundamental_row()
    row.update({
        "fundamental_conversion_score": 20.0,
        "fundamental_coverage_pct": 80.0,
        "fundamental_data_quality_score": 90.0,
        "fundamental_state": "FUNDAMENTAL_WEAK",
        "fundamental_official_source_coverage_pct": 80.0,
    })

    result = dashboard.calculate_next_leader_score(row)

    assert result["next_leader_fundamental_scoring_state"] == "OBSERVED_SCORED"
    assert not bool(result["next_leader_eligible"])
    assert result["next_leader_state"] == "NEXT_LEADER_NOT_QUALIFIED"


def test_real_money_candidate_score_excludes_missing_fundamental() -> None:
    row = {
        "next_leader_score": 80.0,
        "emir_final_score": 80.0,
        "dashboard_silent_accum_score": 80.0,
        "dashboard_flow_score": 80.0,
        "market_structure_score": 80.0,
        "liquidity_score": 80.0,
        "fundamental_conversion_score": np.nan,
        "smart_money_score": 80.0,
        "distribution_score": 20.0,
        "execution_rr_tp1": 2.4,
    }

    result = dashboard.calculate_real_money_candidate_score(row)

    assert np.isfinite(float(result["real_money_candidate_score"]))
    assert float(result["real_money_candidate_score"]) > 0.0


def test_missing_fundamental_is_evidence_gap_not_material_risk() -> None:
    frame = pd.DataFrame([{
        "real_money_gate_state": "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED",
        "real_money_block_reasons": "NONE",
        "real_money_manual_conditions": "FUNDAMENTAL_NOT_AVAILABLE_MANUAL_VERIFY",
        "risk_flags": "FUNDAMENTAL_NOT_AVAILABLE_NOT_SCORED",
    }])

    out = zapi_patch._apply_gate_audit(frame)

    assert out.loc[0, "real_money_evidence_gap_flags"] == "FUNDAMENTAL_NOT_AVAILABLE_NOT_SCORED"
    assert out.loc[0, "real_money_material_risk_flags"] == "NONE"
    assert int(out.loc[0, "real_money_hard_block_count"]) == 0


def test_emir_profile_source_keeps_missing_fundamental_out_of_hard_block_rules() -> None:
    source = inspect.getsource(engine.build_emir_profile)

    assert "FUNDAMENTAL_NOT_AVAILABLE_MANUAL_VERIFY" in source
    assert "FUNDAMENTAL_NOT_AVAILABLE_NOT_SCORED" in source
    assert "if fundamental_observed and data_quality < 75" in source
    assert "if fundamental_observed:" in source
