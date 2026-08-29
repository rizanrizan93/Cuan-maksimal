from __future__ import annotations

import pandas as pd

from evidence_governance import apply_three_rank_contract, calibrate_guardrails_walk_forward, validate_official_evidence


def test_official_evidence_is_fail_closed():
    ok = validate_official_evidence(
        source_url="https://issuer.example/presentation.pdf",
        source_urls=["https://issuer.example/presentation.pdf", "https://issuer.example/disclosure"],
        evidence_date="2026-06-19",
        entity_match_verified=True,
        source_verified=True,
        quorum_required=True,
    )
    assert ok["evidence_production_valid"] is True
    bad = validate_official_evidence(
        source_url="http://issuer.example/presentation.pdf",
        source_urls=["http://issuer.example/presentation.pdf"],
        evidence_date=None,
        entity_match_verified=False,
        source_verified=True,
        quorum_required=True,
    )
    assert bad["evidence_production_valid"] is False


def test_three_rank_contract_requires_completed_authorization_for_production_rank():
    frame = pd.DataFrame({
        "ticker": ["AAA.JK", "BBB.JK"],
        "emir_conviction_score": [90.0, 85.0],
        "next_leader_score": [70.0, 80.0],
        "real_money_candidate_score": [75.0, 83.0],
        "real_money_candidate": [True, True],
        "real_money_entry_candidate": [False, True],
        "real_money_ready": [False, True],
        "emir_decision_state": ["EMIR_WAIT_REACCUMULATION", "EMIR_READY_WITH_PRECISE_TRIGGER"],
        "emir_action_state": ["EMIR_READY_WITH_PRECISE_TRIGGER", "EMIR_WAIT_MONEY_FLOW"],
        "real_money_gate_state": ["REAL_MONEY_WAIT_TIMING", "REAL_MONEY_DIRECT_VERIFIED_READY"],
        "entry_authorization_state": ["WAIT_TIMING_NO_ENTRY", "SCANNER_AUTHORIZED_DIRECT_VERIFIED"],
        "real_money_block_reasons": ["NONE", "NONE"],
        "execution_geometry_valid": [True, True],
        "execution_entry_reference": [100.0, 100.0],
        "execution_stop_loss": [90.0, 90.0],
        "execution_tp1": [116.0, 116.0],
        "execution_tp2": [125.0, 125.0],
        "preferred_execution_path": ["ACCUMULATION_PULLBACK", "ACCUMULATION_PULLBACK"],
        "execution_rr_tp1": [1.6, 1.6],
        "execution_min_rr_pass": [True, True],
    })
    out = apply_three_rank_contract(frame).set_index("ticker")
    assert int(out.loc["AAA.JK", "raw_research_rank"]) == 1
    assert int(out.loc["BBB.JK", "guarded_decision_priority_rank"]) == 1
    assert pd.isna(out.loc["AAA.JK", "production_real_money_rank"])
    assert int(out.loc["BBB.JK", "production_real_money_rank"]) == 1
    assert out.loc["AAA.JK", "emir_action_state"] == "EMIR_WAIT_REACCUMULATION"
    assert out.loc["BBB.JK", "emir_action_state"] == "EMIR_READY_WITH_PRECISE_TRIGGER"


def test_unverified_outcomes_never_activate_calibration():
    outcomes = pd.DataFrame({
        "signal_date": ["2026-08-12", "2026-08-13"],
        "forward_return_20d": [None, None],
        "outcome_verified": [False, False],
    })
    result = calibrate_guardrails_walk_forward(outcomes)
    assert result["active"] is False
    assert "INSUFFICIENT" in result["calibration_state"]
