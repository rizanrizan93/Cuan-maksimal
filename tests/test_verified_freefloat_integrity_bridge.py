import numpy as np

from resumable_scan import _bridge_verified_ownership_free_float_to_integrity


def test_verified_direct_freefloat_fills_only_integrity_freefloat_dimension():
    ownership = {
        "MARK.JK": {
            "effective_free_float_pct": 19.73,
            "reported_free_float_pct": 19.73,
            "ownership_provenance_state": "DIRECT_SOURCE_VERIFIED+KSEI_REGISTRATION_PROXY_NOT_FREE_FLOAT",
        }
    }
    integrity = {
        "MARK.JK": {
            "idx_integrity_score": 88.0,
            "idx_integrity_coverage_pct": 42.9,
            "idx_integrity_hard_block": False,
            "idx_integrity_block_reasons": "NONE",
            "idx_integrity_caution_flags": "CRITICAL_IDX_FIELDS_UNKNOWN_NOT_VERIFIED",
            "regulatory_free_float_pct": np.nan,
            "regulatory_free_float_verification_state": "UNKNOWN_NOT_VERIFIED",
            "hsc_verification_state": "UNKNOWN_NOT_VERIFIED",
            "full_call_auction_verification_state": "UNKNOWN_NOT_VERIFIED",
            "uma_verification_state": "UNKNOWN_NOT_VERIFIED",
            "sanctions_verification_state": "UNKNOWN_NOT_VERIFIED",
        }
    }
    out = _bridge_verified_ownership_free_float_to_integrity(ownership, integrity)["MARK.JK"]
    assert out["regulatory_free_float_pct"] == 19.73
    assert out["regulatory_free_float_verification_state"] == "VERIFIED_FROM_DIRECT_OWNERSHIP_SOURCE"
    assert out["idx_integrity_coverage_pct"] == 57.2
    assert out["idx_integrity_hard_block"] is False
    assert out["hsc_verification_state"] == "UNKNOWN_NOT_VERIFIED"
    assert out["full_call_auction_verification_state"] == "UNKNOWN_NOT_VERIFIED"
    assert out["uma_verification_state"] == "UNKNOWN_NOT_VERIFIED"
    assert out["sanctions_verification_state"] == "UNKNOWN_NOT_VERIFIED"


def test_low_verified_freefloat_adds_caution_without_clearing_unknowns():
    ownership = {
        "TSPC.JK": {
            "effective_free_float_pct": 9.13,
            "ownership_provenance_state": "DIRECT_SOURCE_VERIFIED",
        }
    }
    integrity = {
        "TSPC.JK": {
            "idx_integrity_score": 88.0,
            "idx_integrity_coverage_pct": 42.9,
            "idx_integrity_hard_block": False,
            "idx_integrity_block_reasons": "NONE",
            "idx_integrity_caution_flags": "CRITICAL_IDX_FIELDS_UNKNOWN_NOT_VERIFIED",
            "regulatory_free_float_pct": np.nan,
            "regulatory_free_float_verification_state": "UNKNOWN_NOT_VERIFIED",
        }
    }
    out = _bridge_verified_ownership_free_float_to_integrity(ownership, integrity)["TSPC.JK"]
    assert out["idx_integrity_coverage_pct"] == 57.2
    assert out["idx_integrity_hard_block"] is False
    assert "FREE_FLOAT_BELOW_15PCT" in out["idx_integrity_caution_flags"]


def test_extreme_low_verified_freefloat_hard_blocks_but_does_not_infer_other_flags():
    ownership = {
        "LOW.JK": {
            "effective_free_float_pct": 6.0,
            "ownership_provenance_state": "DIRECT_SOURCE_VERIFIED",
        }
    }
    integrity = {
        "LOW.JK": {
            "idx_integrity_score": 88.0,
            "idx_integrity_coverage_pct": 42.9,
            "idx_integrity_hard_block": False,
            "idx_integrity_block_reasons": "NONE",
            "idx_integrity_caution_flags": "NONE",
            "hsc_verification_state": "UNKNOWN_NOT_VERIFIED",
            "regulatory_free_float_pct": np.nan,
            "regulatory_free_float_verification_state": "UNKNOWN_NOT_VERIFIED",
        }
    }
    out = _bridge_verified_ownership_free_float_to_integrity(ownership, integrity)["LOW.JK"]
    assert out["idx_integrity_hard_block"] is True
    assert "EXTREME_LOW_FREE_FLOAT" in out["idx_integrity_block_reasons"]
    assert out["hsc_verification_state"] == "UNKNOWN_NOT_VERIFIED"


def test_unverified_or_proxy_only_freefloat_never_promoted_to_integrity():
    ownership = {
        "PROXY.JK": {
            "effective_free_float_pct": 20.0,
            "ownership_provenance_state": "KSEI_REGISTRATION_PROXY_NOT_FREE_FLOAT",
        }
    }
    integrity = {
        "PROXY.JK": {
            "idx_integrity_coverage_pct": 42.9,
            "regulatory_free_float_pct": np.nan,
            "regulatory_free_float_verification_state": "UNKNOWN_NOT_VERIFIED",
        }
    }
    out = _bridge_verified_ownership_free_float_to_integrity(ownership, integrity)["PROXY.JK"]
    assert out["idx_integrity_coverage_pct"] == 42.9
    assert np.isnan(out["regulatory_free_float_pct"])
    assert out["regulatory_free_float_verification_state"] == "UNKNOWN_NOT_VERIFIED"
