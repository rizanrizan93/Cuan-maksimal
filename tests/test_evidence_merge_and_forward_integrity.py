import numpy as np
import pandas as pd

from future_fundamental import calculate_future_fundamental
from resumable_scan import _bridge_verified_ownership_free_float_to_integrity, _merge_evidence_profile_maps


def _base_context():
    narrative = {
        "top_down_catalyst_score": 70.0,
        "industry_translation_score": 70.0,
        "narrative_coverage_pct": 90.0,
        "issuer_alignment_score": 65.0,
        "issuer_alignment_coverage_pct": 90.0,
    }
    fundamental = {
        "fundamental_coverage_pct": 90.0,
        "fundamental_cashflow_quality_state": "CASHFLOW_POSITIVE_CONVERTING",
        "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK",
        "current_ratio": 1.5,
        "cash_to_debt_ratio": 1.0,
        "fundamental_conversion_score": 75.0,
        "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED",
        "fundamental_period_freshness_state": "CURRENT",
        "fundamental_observed_at": "2026-01-01T00:00:00Z",
        "fundamental_availability_state": "POINT_IN_TIME_OBSERVED",
    }
    ownership = {"ownership_score": 60.0, "ownership_coverage_pct": 35.0}
    sector = {"sector_leadership_score": 65.0, "sector_context_coverage_pct": 90.0}
    return narrative, fundamental, ownership, sector


def test_ticker_name_mine_is_not_a_forward_project_event():
    narrative, fundamental, ownership, sector = _base_context()
    events = pd.DataFrame([
        {
            "ticker": "MINE.JK",
            "published_at": "2026-06-30T00:00:00Z",
            "title": "IDX Official Financial Statement MINE 2026-06-30",
            "summary": "Official IDX XBRL filing; revenue and earnings update",
            "category": "EARNINGS_CONVERSION",
            "source_verified": True,
            "source_tier": "OFFICIAL",
        },
        {
            "ticker": "MINE.JK",
            "published_at": "2026-05-04T00:00:00Z",
            "title": "KSEI corporate action: Cash Dividend",
            "summary": "MINE cash dividend; status Active",
            "category": "CORPORATE_ACTION",
            "event_role": "ADMINISTRATIVE_CORPORATE_ACTION",
            "source_verified": True,
            "source_tier": "OFFICIAL",
        },
    ])
    out = calculate_future_fundamental(
        ticker="MINE.JK", events=events, narrative=narrative,
        fundamental=fundamental, ownership=ownership, sector=sector,
        as_of="2026-08-14T00:00:00Z",
    )
    assert out["future_forward_event_count"] == 0
    assert out["future_verified_forward_event_count"] == 0
    assert out["future_official_forward_event_count"] == 0


def test_ticker_name_wapo_does_not_become_purchase_order_evidence():
    narrative, fundamental, ownership, sector = _base_context()
    events = pd.DataFrame([
        {
            "ticker": "WAPO.JK",
            "published_at": "2026-06-30T00:00:00Z",
            "title": "IDX Official Financial Statement WAPO 2026-06-30",
            "summary": "Official IDX XBRL filing; revenue and earnings update",
            "category": "EARNINGS_CONVERSION",
            "source_verified": True,
            "source_tier": "OFFICIAL",
        },
        {
            "ticker": "WAPO.JK",
            "published_at": "2026-06-03T00:00:00Z",
            "title": "KSEI corporate action: Proxy Voting",
            "summary": "WAPO status Active",
            "category": "CORPORATE_ACTION",
            "event_role": "ADMINISTRATIVE_CORPORATE_ACTION",
            "source_verified": True,
            "source_tier": "OFFICIAL",
        },
    ])
    out = calculate_future_fundamental(
        ticker="WAPO.JK", events=events, narrative=narrative,
        fundamental=fundamental, ownership=ownership, sector=sector,
        as_of="2026-08-14T00:00:00Z",
    )
    assert out["future_forward_event_count"] == 0
    assert out["future_verified_forward_event_count"] == 0
    assert out["future_official_forward_event_count"] == 0


def test_real_issuer_capacity_event_remains_forward_evidence():
    narrative, fundamental, ownership, sector = _base_context()
    events = pd.DataFrame([{
        "ticker": "MARK.JK",
        "published_at": "2026-06-19T00:00:00Z",
        "title": "MARK capacity expansion and customer orders through Q3 2026",
        "summary": "Issuer increased production capacity and has customer orders through Q3 2026.",
        "category": "CAPACITY_AND_BACKLOG",
        "source_verified": True,
        "source_tier": "ISSUER",
    }])
    out = calculate_future_fundamental(
        ticker="MARK.JK", events=events, narrative=narrative,
        fundamental=fundamental, ownership=ownership, sector=sector,
        as_of="2026-08-14T00:00:00Z",
    )
    assert out["future_forward_event_count"] >= 1
    assert out["future_verified_forward_event_count"] >= 1
    assert out["future_official_forward_event_count"] >= 1


def test_direct_ownership_does_not_destroy_ksei_context_coverage():
    auto = {
        "MARK.JK": {
            "ownership_score": np.nan,
            "ownership_coverage_pct": 33.8,
            "total_shares_ksei": 3_800_000_310,
            "local_pct_ksei": 80.0,
            "foreign_pct_ksei": 20.0,
            "ownership_provenance_state": "KSEI_REGISTRATION_PROXY_NOT_FREE_FLOAT",
        }
    }
    direct = {
        "MARK.JK": {
            "ownership_score": 85.0,
            "ownership_coverage_pct": 15.0,
            "reported_free_float_pct": 19.73,
            "effective_free_float_pct": 19.73,
            "ownership_provenance_state": "DIRECT_SOURCE_VERIFIED",
        }
    }
    merged = _merge_evidence_profile_maps(
        auto, direct,
        coverage_key="ownership_coverage_pct",
        provenance_key="ownership_provenance_state",
    )["MARK.JK"]
    assert merged["ownership_score"] == 85.0
    assert merged["reported_free_float_pct"] == 19.73
    assert merged["total_shares_ksei"] == 3_800_000_310
    assert merged["ownership_coverage_pct"] == 33.8
    assert "DIRECT_SOURCE_VERIFIED" in merged["ownership_provenance_state"]
    assert "KSEI_REGISTRATION_PROXY_NOT_FREE_FLOAT" in merged["ownership_provenance_state"]


def test_disjoint_ownership_evidence_coverage_is_unioned_without_double_counting_integrity():
    auto = {
        "MARK.JK": {
            "ownership_coverage_pct": 33.8,
            "total_shares_ksei": 3_800_000_310,
            "local_pct_ksei": 80.0,
            "foreign_pct_ksei": 20.0,
            "ownership_provenance_state": "KSEI_REGISTRATION_PROXY_NOT_FREE_FLOAT",
        }
    }
    direct = {
        "MARK.JK": {
            "ownership_coverage_pct": 15.0,
            "effective_free_float_pct": 19.73,
            "ownership_provenance_state": "DIRECT_SOURCE_VERIFIED",
        }
    }
    merged = _merge_evidence_profile_maps(
        auto, direct,
        coverage_key="ownership_coverage_pct",
        provenance_key="ownership_provenance_state",
        coverage_mode="union_disjoint",
    )["MARK.JK"]
    assert merged["ownership_coverage_pct"] == 48.8


def test_direct_integrity_cannot_clear_existing_hard_block():
    auto = {
        "TEST.JK": {
            "idx_integrity_score": 5.0,
            "idx_integrity_coverage_pct": 60.0,
            "idx_integrity_hard_block": True,
            "idx_integrity_block_reasons": "SUSPENSION_TRUE_VERIFIED",
            "idx_integrity_provenance_state": "AUTO_PUBLIC_KSEI_AND_REGULATORY_NEWS",
        }
    }
    direct = {
        "TEST.JK": {
            "idx_integrity_score": 80.0,
            "idx_integrity_coverage_pct": 57.1,
            "idx_integrity_hard_block": False,
            "idx_integrity_block_reasons": "NONE",
            "idx_integrity_provenance_state": "DIRECT_SOURCE_VERIFIED",
        }
    }
    merged = _merge_evidence_profile_maps(
        auto, direct,
        coverage_key="idx_integrity_coverage_pct",
        provenance_key="idx_integrity_provenance_state",
        hard_block_key="idx_integrity_hard_block",
        reason_key="idx_integrity_block_reasons",
    )["TEST.JK"]
    assert merged["idx_integrity_hard_block"] is True
    assert "SUSPENSION_TRUE_VERIFIED" in merged["idx_integrity_block_reasons"]
    assert merged["idx_integrity_coverage_pct"] == 60.0


def test_verified_free_float_bridges_exactly_one_integrity_dimension():
    ownership = {
        "MARK.JK": {
            "effective_free_float_pct": 19.73,
            "ownership_provenance_state": "DIRECT_SOURCE_VERIFIED+KSEI_REGISTRATION_PROXY_NOT_FREE_FLOAT",
        }
    }
    integrity = {
        "MARK.JK": {
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
    assert out["idx_integrity_coverage_pct"] == 57.2
    assert out["idx_integrity_hard_block"] is False
    assert out["hsc_verification_state"] == "UNKNOWN_NOT_VERIFIED"
    assert out["full_call_auction_verification_state"] == "UNKNOWN_NOT_VERIFIED"
    assert out["uma_verification_state"] == "UNKNOWN_NOT_VERIFIED"
    assert out["sanctions_verification_state"] == "UNKNOWN_NOT_VERIFIED"


def test_extreme_low_verified_free_float_hard_blocks_without_inventing_other_regulatory_states():
    ownership = {
        "LOW.JK": {
            "effective_free_float_pct": 6.0,
            "ownership_provenance_state": "DIRECT_SOURCE_VERIFIED",
        }
    }
    integrity = {
        "LOW.JK": {
            "idx_integrity_coverage_pct": 42.9,
            "idx_integrity_hard_block": False,
            "idx_integrity_block_reasons": "NONE",
            "idx_integrity_caution_flags": "NONE",
            "regulatory_free_float_pct": np.nan,
            "regulatory_free_float_verification_state": "UNKNOWN_NOT_VERIFIED",
            "hsc_verification_state": "UNKNOWN_NOT_VERIFIED",
        }
    }
    out = _bridge_verified_ownership_free_float_to_integrity(ownership, integrity)["LOW.JK"]
    assert out["idx_integrity_hard_block"] is True
    assert "EXTREME_LOW_FREE_FLOAT" in out["idx_integrity_block_reasons"]
    assert out["hsc_verification_state"] == "UNKNOWN_NOT_VERIFIED"


def test_future_fundamental_excludes_post_asof_events():
    narrative, fundamental, ownership, sector = _base_context()
    events = pd.DataFrame([{
        "ticker": "MARK.JK",
        "published_at": "2026-09-01T00:00:00Z",
        "title": "MARK signs major customer contract and expands capacity",
        "summary": "Contract backlog and capacity expansion support future revenue.",
        "category": "CAPACITY_AND_BACKLOG",
        "source_verified": True,
        "source_tier": "ISSUER",
    }])
    out = calculate_future_fundamental(
        ticker="MARK.JK", events=events, narrative=narrative,
        fundamental=fundamental, ownership=ownership, sector=sector,
        as_of="2026-08-14T00:00:00Z",
    )
    assert out["future_forward_event_count"] == 0
    assert out["future_verified_forward_event_count"] == 0
    assert out["future_official_forward_event_count"] == 0
    assert np.isnan(out["future_direct_forward_visibility_score"])


def test_future_fundamental_excludes_snapshot_observed_after_decision_time():
    narrative, fundamental, ownership, sector = _base_context()
    fundamental["fundamental_observed_at"] = "2026-09-01T00:00:00Z"
    out = calculate_future_fundamental(
        ticker="MARK.JK",
        events=pd.DataFrame(),
        narrative=narrative,
        fundamental=fundamental,
        ownership=ownership,
        sector=sector,
        as_of="2026-08-14T00:00:00Z",
    )
    assert out["future_fundamental_input_availability_state"] == "FUTURE_FUNDAMENTAL_SNAPSHOT_EXCLUDED"
    assert np.isnan(out["future_funding_capacity_score"])
    assert np.isnan(out["future_business_confirmation_score"])
