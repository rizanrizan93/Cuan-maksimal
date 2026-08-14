from resumable_scan import _merge_evidence_profile_maps


def test_disjoint_ownership_coverage_unions_ksei_and_direct_fields():
    base = {
        "MARK.JK": {
            "ownership_score": 57.0,
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
        base, direct,
        coverage_key="ownership_coverage_pct",
        provenance_key="ownership_provenance_state",
        coverage_mode="union_disjoint",
    )["MARK.JK"]
    assert merged["ownership_coverage_pct"] == 48.8
    assert merged["reported_free_float_pct"] == 19.73
    assert merged["total_shares_ksei"] == 3_800_000_310


def test_integrity_overlap_remains_conservative_max_coverage():
    base = {"X.JK": {"idx_integrity_coverage_pct": 42.9, "idx_integrity_provenance_state": "AUTO"}}
    direct = {"X.JK": {"idx_integrity_coverage_pct": 57.1, "idx_integrity_provenance_state": "DIRECT"}}
    merged = _merge_evidence_profile_maps(
        base, direct,
        coverage_key="idx_integrity_coverage_pct",
        provenance_key="idx_integrity_provenance_state",
    )["X.JK"]
    assert merged["idx_integrity_coverage_pct"] == 57.1
