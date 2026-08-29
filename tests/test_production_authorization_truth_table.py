from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from evidence_governance import apply_three_rank_contract
from top3_dashboard_legacy import select_real_money_top3
from top3_lane_patch import select_execution_top3, select_guarded_top3, select_research_top3


def _authorized_row(ticker: str = "VALID.JK") -> dict[str, object]:
    return {
        "ticker": ticker,
        "emir_conviction_score": 82.0,
        "next_leader_score": 78.0,
        "real_money_candidate_score": 80.0,
        "real_money_candidate": True,
        "real_money_entry_candidate": True,
        "real_money_ready": True,
        "production_ready": True,
        "auto_eod_ready": False,
        "emir_decision_state": "EMIR_READY_WITH_PRECISE_TRIGGER",
        "emir_action_state": "EMIR_READY_WITH_PRECISE_TRIGGER",
        "real_money_gate_state": "REAL_MONEY_DIRECT_VERIFIED_READY",
        "entry_authorization_state": "SCANNER_AUTHORIZED_DIRECT_VERIFIED",
        "real_money_authorization_tier": "DIRECT_VERIFIED_READY",
        "real_money_gate_class": "DIRECT_VERIFIED_READY",
        "real_money_hard_block_count": 0,
        "real_money_block_reasons": "NONE",
        "execution_geometry_valid": True,
        "execution_entry_reference": 100.0,
        "execution_stop_loss": 90.0,
        "execution_tp1": 116.0,
        "execution_tp2": 125.0,
        "preferred_execution_path": "ACCUMULATION_PULLBACK",
        "execution_rr_tp1": 1.6,
        "execution_min_rr_pass": True,
    }


def _case(case: str) -> dict[str, object]:
    row = deepcopy(_authorized_row(case + ".JK"))
    if case == "HARD_BLOCK":
        row.update(real_money_hard_block_count=1, real_money_block_reasons="EXECUTION_CAPACITY_BLOCK")
    elif case == "WAIT_TIMING":
        row.update(
            emir_decision_state="EMIR_WAIT_REACCUMULATION",
            real_money_entry_candidate=False,
            real_money_ready=False,
            real_money_gate_state="REAL_MONEY_WAIT_TIMING",
            entry_authorization_state="WAIT_TIMING_NO_ENTRY",
        )
    elif case == "AUTO_EOD_PROXY":
        row.update(
            emir_decision_state="EMIR_AUTO_EOD_READY",
            auto_eod_ready=True,
            real_money_ready=False,
            real_money_gate_state="REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED",
            entry_authorization_state="MANUAL_CONFIRMATION_REQUIRED",
            real_money_authorization_tier="PROXY_EXECUTION_ELIGIBLE_MANUAL_CONFIRMATION",
        )
    elif case == "INVALID_GEOMETRY":
        row.update(execution_geometry_valid=True, execution_stop_loss=105.0)
    elif case == "RR_BELOW_MINIMUM":
        # A stale true summary flag cannot override the selected path's numeric RR.
        row.update(execution_min_rr_pass=True, execution_rr_tp1=1.2)
    elif case == "AUTHORIZATION_MISSING":
        row.update(
            real_money_ready=False,
            real_money_gate_state="REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED",
            entry_authorization_state="MANUAL_CONFIRMATION_REQUIRED",
        )
    elif case == "EVIDENCE_GAP_ONLY":
        row.update(
            emir_decision_state="EMIR_THESIS_READY_WAIT_BID_OFFER",
            real_money_ready=False,
            real_money_gate_state="REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED",
            entry_authorization_state="MANUAL_CONFIRMATION_REQUIRED",
            real_money_authorization_tier="PROXY_EXECUTION_ELIGIBLE_MANUAL_CONFIRMATION",
            real_money_gate_class="EVIDENCE_GAP_ONLY_OR_OTHER",
            real_money_block_reasons="IDX_INTEGRITY_EVIDENCE_MISSING | DIRECT_BID_OFFER_TRIGGER_MISSING_PROXY_USED",
        )
    elif case == "EVIDENCE_GAP_STRONG_RESEARCH":
        row.update(
            real_money_ready=False,
            real_money_gate_state="REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED",
            entry_authorization_state="MANUAL_CONFIRMATION_REQUIRED",
            real_money_authorization_tier="PROXY_EXECUTION_ELIGIBLE_MANUAL_CONFIRMATION",
            real_money_gate_class="EVIDENCE_GAP_ONLY_OR_OTHER",
            real_money_block_reasons="IDX_INTEGRITY_EVIDENCE_MISSING | DIRECT_BID_OFFER_TRIGGER_MISSING_PROXY_USED",
        )
    elif case == "EVIDENCE_GAP_WAIT_TIMING":
        row.update(
            emir_decision_state="EMIR_WAIT_REACCUMULATION",
            real_money_entry_candidate=False,
            real_money_ready=False,
            real_money_gate_state="REAL_MONEY_WAIT_TIMING",
            entry_authorization_state="WAIT_TIMING_NO_ENTRY",
            real_money_authorization_tier="WAIT_TIMING",
            real_money_gate_class="EVIDENCE_GAP_ONLY_OR_OTHER",
            real_money_block_reasons="IDX_INTEGRITY_EVIDENCE_MISSING | DIRECT_BID_OFFER_TRIGGER_MISSING_PROXY_USED",
        )
    elif case == "VALID_ACTIONABLE":
        pass
    elif case == "UNKNOWN_STATE":
        row.update(emir_decision_state="EMIR_UNKNOWN_FUTURE_STATE")
    elif case == "STALE_LEGACY_DISAGREES":
        row.update(
            emir_decision_state="EMIR_WAIT_MONEY_FLOW",
            emir_action_state="EMIR_READY_WITH_PRECISE_TRIGGER",
        )
    else:
        raise AssertionError(f"unknown case {case}")
    return row


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("HARD_BLOCK", False),
        ("WAIT_TIMING", False),
        ("AUTO_EOD_PROXY", False),
        ("INVALID_GEOMETRY", False),
        ("RR_BELOW_MINIMUM", False),
        ("AUTHORIZATION_MISSING", False),
        ("VALID_ACTIONABLE", True),
        ("UNKNOWN_STATE", False),
        ("STALE_LEGACY_DISAGREES", False),
    ],
)
def test_production_and_execution_truth_table(case: str, expected: bool) -> None:
    source = pd.DataFrame([_case(case)])
    ranked = apply_three_rank_contract(source)

    assert bool(ranked.iloc[0]["production_authorization_pass"]) is expected
    assert bool(ranked.iloc[0]["execution_authorized"]) is expected
    assert bool(pd.notna(ranked.iloc[0]["production_real_money_rank"])) is expected
    assert bool(pd.notna(ranked.iloc[0]["raw_research_rank"])) is True
    assert (not select_execution_top3(source, 3).empty) is expected
    assert (not select_real_money_top3(source, 3).empty) is expected


def test_canonical_decision_state_overwrites_stale_legacy_mirror() -> None:
    ranked = apply_three_rank_contract(pd.DataFrame([_case("STALE_LEGACY_DISAGREES")]))
    assert ranked.iloc[0]["emir_action_state"] == "EMIR_WAIT_MONEY_FLOW"
    assert pd.isna(ranked.iloc[0]["production_real_money_rank"])


def test_broad_flags_alone_never_create_production_rank() -> None:
    row = _case("WAIT_TIMING")
    row.update(real_money_candidate=True, auto_eod_ready=True, production_ready=True)
    ranked = apply_three_rank_contract(pd.DataFrame([row]))
    assert bool(ranked.iloc[0]["production_authorization_pass"]) is False
    assert pd.isna(ranked.iloc[0]["production_real_money_rank"])


@pytest.mark.parametrize(
    ("case", "guarded", "production", "execution"),
    [
        ("EVIDENCE_GAP_ONLY", True, False, False),
        ("HARD_BLOCK", False, False, False),
        ("EVIDENCE_GAP_STRONG_RESEARCH", True, False, False),
        ("EVIDENCE_GAP_WAIT_TIMING", True, False, False),
        ("VALID_ACTIONABLE", True, True, True),
    ],
)
def test_evidence_gap_and_hard_block_lane_contract(
    case: str, guarded: bool, production: bool, execution: bool,
) -> None:
    source = pd.DataFrame([_case(case)])
    ranked = apply_three_rank_contract(source)

    assert not select_research_top3(source, 3).empty
    assert (not select_guarded_top3(source, 3).empty) is guarded
    assert bool(pd.notna(ranked.iloc[0]["production_real_money_rank"])) is production
    assert bool(ranked.iloc[0]["production_ready"]) is production
    assert (not select_execution_top3(source, 3).empty) is execution
