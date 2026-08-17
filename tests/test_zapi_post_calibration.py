import pandas as pd

from zapi_post_calibration import enrich_emir_shadow, refine_emir_proxy_authorization_tier
from zapi_runtime_patch import _apply_gate_audit


def test_gate_audit_moves_direct_evidence_gap_out_of_hard_block_count():
    frame = pd.DataFrame([
        {
            "ticker": "TEST.JK",
            "real_money_gate_state": "REAL_MONEY_BLOCKED",
            "real_money_block_reasons": "IDX_INTEGRITY_EVIDENCE_MISSING | DIRECT_BID_OFFER_TRIGGER_MISSING_PROXY_USED",
            "real_money_manual_conditions": "NONE",
            "risk_flags": "NO_MAJOR_EMIR_IDX_FRAMEWORK_RISK | IDX_CRITICAL_FIELDS_UNKNOWN_NOT_VERIFIED",
        }
    ])
    out = _apply_gate_audit(frame).iloc[0]
    assert int(out["real_money_hard_block_count"]) == 0
    assert int(out["real_money_evidence_gap_count"]) == 3
    assert out["real_money_gate_class"] == "EVIDENCE_GAP_ONLY_OR_OTHER"
    assert out["real_money_material_risk_flags"] == "NONE"


def test_proxy_tier_reclassifies_evidence_gap_only_block_to_manual_not_ready():
    frame = pd.DataFrame([
        {
            "ticker": "TEST.JK",
            "real_money_gate_state": "REAL_MONEY_BLOCKED",
            "real_money_gate_class": "EVIDENCE_GAP_ONLY_OR_OTHER",
            "real_money_gate_explanation": "old",
            "real_money_hard_block_count": 0,
            "real_money_evidence_gap_count": 2,
            "real_money_manual_condition_count": 0,
            "real_money_candidate": True,
            "real_money_entry_candidate": True,
            "real_money_ready": False,
            "production_ready": False,
            "production_tier": "BLOCKED",
            "entry_authorization_state": "NO_ENTRY_AUTHORIZATION",
        }
    ])
    out = refine_emir_proxy_authorization_tier(frame).iloc[0]
    assert out["real_money_authorization_tier"] == "PROXY_EXECUTION_ELIGIBLE_MANUAL_CONFIRMATION"
    assert bool(out["real_money_proxy_authorization_eligible"]) is True
    assert out["real_money_gate_state"] == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    assert out["real_money_gate_class"] == "MANUAL_CONFIRMATION"
    assert bool(out["real_money_ready"]) is False
    assert bool(out["production_ready"]) is False
    assert out["entry_authorization_state"] == "MANUAL_CONFIRMATION_REQUIRED"


def test_proxy_tier_never_overrides_genuine_hard_blocker():
    frame = pd.DataFrame([
        {
            "ticker": "PKPK.JK",
            "real_money_gate_state": "REAL_MONEY_BLOCKED",
            "real_money_hard_block_count": 1,
            "real_money_evidence_gap_count": 3,
            "real_money_manual_condition_count": 0,
            "real_money_candidate": True,
            "real_money_entry_candidate": True,
            "real_money_ready": False,
        }
    ])
    out = refine_emir_proxy_authorization_tier(frame).iloc[0]
    assert out["real_money_authorization_tier"] == "HARD_BLOCKED"
    assert bool(out["real_money_proxy_authorization_eligible"]) is False
    assert out["real_money_gate_state"] == "REAL_MONEY_BLOCKED"
    assert bool(out["real_money_ready"]) is False


def test_shadow_audit_reconstructs_pre_zapi_conviction_and_smart_money():
    frame = pd.DataFrame([
        {
            "ticker": "TEST.JK",
            "emir_conviction_score": 72.0,
            "zapi_emir_conviction_delta": 2.0,
            "smart_money_score": 74.0,
            "zapi_smart_money_confirmation_score": 90.0,
            "zapi_smart_money_confirmation_weight_pct": 20.0,
            "smart_money_cost_confidence_pct": 58.0,
            "zapi_smart_money_cost_confidence_delta": 4.0,
            "zapi_foreign_flow_coverage_pct": 100.0,
        }
    ])
    out = enrich_emir_shadow(frame).iloc[0]
    assert float(out["zapi_shadow_pre_conviction_score"]) == 70.0
    assert float(out["zapi_shadow_post_conviction_score"]) == 72.0
    assert float(out["zapi_shadow_conviction_delta"]) == 2.0
    # 74 = 80% * 70 + 20% * 90
    assert float(out["zapi_shadow_pre_smart_money_score"]) == 70.0
    assert float(out["zapi_shadow_post_smart_money_score"]) == 74.0
    assert float(out["zapi_shadow_smart_money_delta"]) == 4.0
    assert float(out["zapi_shadow_pre_cost_confidence_pct"]) == 54.0
    assert float(out["zapi_shadow_post_cost_confidence_pct"]) == 58.0
    assert out["zapi_shadow_calibration_state"] == "PENDING_FORWARD_OUTCOME"
    assert out["zapi_shadow_forward_horizons"] == "5D|20D|60D"
