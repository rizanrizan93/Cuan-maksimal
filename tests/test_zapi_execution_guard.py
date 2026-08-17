import pandas as pd

from zapi_runtime_patch import _apply_foreign_shock_guard, _apply_gate_audit


def _row(**overrides):
    base = {
        "ticker": "TEST.JK",
        "zapi_foreign_flow_coverage_pct": 100.0,
        "zapi_foreign_net_participation_1d": 0.0,
        "zapi_foreign_net_participation_5d": 0.02,
        "zapi_foreign_net_participation_20d": 0.03,
        "zapi_foreign_state": "NET_ACCUMULATION",
        "real_money_candidate": True,
        "real_money_entry_candidate": True,
        "real_money_ready": True,
        "real_money_gate_state": "REAL_MONEY_DIRECT_VERIFIED_READY",
        "real_money_block_reasons": "NONE",
        "real_money_manual_conditions": "NONE",
        "risk_flags": "NO_MAJOR_EMIR_IDX_FRAMEWORK_RISK",
        "production_ready": True,
        "production_tier": "GUARDED_DIRECT_VERIFIED",
        "entry_authorization_state": "SCANNER_AUTHORIZED_DIRECT_VERIFIED",
        "guarded_position_cap_after_manual_confirmation_pct": 5.0,
    }
    base.update(overrides)
    return base


def test_extreme_one_day_sell_shock_requires_reclaim_without_hard_blocking_thesis():
    frame = pd.DataFrame([
        _row(
            ticker="OMED.JK",
            zapi_foreign_net_participation_1d=-0.254126,
            zapi_foreign_net_participation_5d=0.004359,
            zapi_foreign_net_participation_20d=0.001841,
            zapi_foreign_state="MIXED_NEUTRAL",
        )
    ])
    out = _apply_foreign_shock_guard(frame).iloc[0]
    assert out["zapi_foreign_shock_state"] == "EXTREME_ONE_DAY_FOREIGN_SELL_SHOCK_RECLAIM_REQUIRED"
    assert out["real_money_candidate"] is True or bool(out["real_money_candidate"]) is True
    assert bool(out["real_money_entry_candidate"]) is True
    assert bool(out["real_money_ready"]) is False
    assert bool(out["production_ready"]) is False
    assert out["real_money_gate_state"] == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    assert "ZAPI_FOREIGN_SHOCK_REQUIRE_ABSORPTION_RECLAIM" in out["real_money_manual_conditions"]


def test_persistent_foreign_distribution_demotes_actionable_candidate_to_wait_timing():
    frame = pd.DataFrame([
        _row(
            zapi_foreign_net_participation_1d=-0.12,
            zapi_foreign_net_participation_5d=-0.04,
            zapi_foreign_net_participation_20d=-0.03,
            zapi_foreign_state="NET_DISTRIBUTION",
        )
    ])
    out = _apply_foreign_shock_guard(frame).iloc[0]
    assert out["zapi_foreign_shock_state"] == "PERSISTENT_FOREIGN_DISTRIBUTION_WAIT"
    assert bool(out["real_money_candidate"]) is True
    assert bool(out["real_money_entry_candidate"]) is False
    assert bool(out["real_money_ready"]) is False
    assert out["real_money_gate_state"] == "REAL_MONEY_WAIT_TIMING"
    assert out["entry_authorization_state"] == "WAIT_TIMING_NO_ENTRY"
    assert "ZAPI_FOREIGN_DISTRIBUTION_WAIT_STABILIZATION" in out["real_money_manual_conditions"]
    assert float(out["guarded_position_cap_after_manual_confirmation_pct"]) == 0.0


def test_strong_positive_zapi_flow_never_promotes_ready():
    frame = pd.DataFrame([
        _row(
            zapi_foreign_net_participation_1d=0.11,
            zapi_foreign_net_participation_5d=0.10,
            zapi_foreign_net_participation_20d=0.08,
            real_money_entry_candidate=False,
            real_money_ready=False,
            real_money_gate_state="REAL_MONEY_WAIT_TIMING",
            production_ready=False,
            production_tier="WAIT_TIMING",
            entry_authorization_state="WAIT_TIMING_NO_ENTRY",
        )
    ])
    out = _apply_foreign_shock_guard(frame).iloc[0]
    assert out["zapi_foreign_shock_state"] == "STRONG_ONE_DAY_FOREIGN_ACCUMULATION_CONFIRMATION_ONLY"
    assert bool(out["real_money_entry_candidate"]) is False
    assert bool(out["real_money_ready"]) is False
    assert bool(out["production_ready"]) is False
    assert out["real_money_gate_state"] == "REAL_MONEY_WAIT_TIMING"


def test_low_coverage_zapi_is_fail_soft_and_does_not_change_gate():
    frame = pd.DataFrame([
        _row(
            zapi_foreign_flow_coverage_pct=40.0,
            zapi_foreign_net_participation_1d=-0.30,
        )
    ])
    out = _apply_foreign_shock_guard(frame).iloc[0]
    assert out["zapi_foreign_shock_state"] == "ZAPI_INSUFFICIENT_OR_STALE_FOR_EXECUTION_GUARD"
    assert bool(out["real_money_ready"]) is True
    assert out["real_money_gate_state"] == "REAL_MONEY_DIRECT_VERIFIED_READY"


def test_gate_audit_separates_universal_evidence_gaps_from_hard_blockers():
    frame = pd.DataFrame([
        _row(
            real_money_candidate=False,
            real_money_entry_candidate=False,
            real_money_ready=False,
            real_money_gate_state="REAL_MONEY_BLOCKED",
            real_money_block_reasons="LIQUIDITY_LT_45 | FUNDAMENTAL_CONVERSION_LT_55",
            real_money_manual_conditions="PROXY_FUNDAMENTAL_MANUAL_VERIFY",
            risk_flags=(
                "DIRECT_BID_OFFER_TRIGGER_MISSING_PROXY_USED | "
                "IDX_CRITICAL_FIELDS_UNKNOWN_NOT_VERIFIED | "
                "IDX_INTEGRITY_EVIDENCE_MISSING | LIQUIDITY_RISK"
            ),
        )
    ])
    out = _apply_gate_audit(frame).iloc[0]
    assert int(out["real_money_hard_block_count"]) == 2
    assert int(out["real_money_manual_condition_count"]) == 1
    assert int(out["real_money_evidence_gap_count"]) == 3
    assert out["real_money_gate_class"] == "HARD_BLOCK"
    assert "LIQUIDITY_RISK" in out["real_money_material_risk_flags"]
    assert "DIRECT_BID_OFFER_TRIGGER_MISSING_PROXY_USED" not in out["real_money_material_risk_flags"]
    assert "IDX_INTEGRITY_EVIDENCE_MISSING" in out["real_money_evidence_gap_flags"]
