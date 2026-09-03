from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resumable_scan import choose_shortlist  # noqa: E402
from dashboard_persistence import build_database_transfer_summary, database_transfer_totals  # noqa: E402


def _fast_frame(count: int = 100, failed: int = 4) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append({
            "ticker": f"T{index:03d}.JK",
            "feature_state": "PROVIDER_FAILED" if index < failed else "OK",
            "smart_money_score": 90 - index * 0.1,
            "market_structure_score": 85 - index * 0.05,
            "seller_exhaustion_score": 70,
            "absorption_score": 68,
            "relative_strength60_pct": 10 - index * 0.1,
            "trend_score": 75,
            "liquidity_score": 80 - index * 0.1,
            "distribution_score": 10,
            "crowding_score": 20,
        })
    return pd.DataFrame(rows)


def test_all_eligible_progressive_scope_reviews_every_valid_ticker():
    fast = _fast_frame(100, failed=4)
    shortlist = choose_shortlist(
        fast,
        sector_map={},
        scan_mode="EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
        deep_limit=30,
        deep_review_scope="ALL_ELIGIBLE",
    )
    assert len(shortlist) == 96
    assert all(ticker not in shortlist for ticker in ["T000.JK", "T001.JK", "T002.JK", "T003.JK"])


def test_fast_balanced_and_custom_scopes_are_bounded():
    fast = _fast_frame(220, failed=0)
    assert len(choose_shortlist(fast, {}, "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP", 100, "FAST_TOP_30")) == 30
    assert len(choose_shortlist(fast, {}, "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP", 100, "BALANCED_TOP_60")) == 60
    assert len(choose_shortlist(fast, {}, "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP", 75, "CUSTOM_LIMIT")) == 75


def test_radar_only_scope_never_schedules_deep_review():
    fast = _fast_frame(20, failed=0)
    assert choose_shortlist(fast, {}, "EMIR_FLOW_RADAR_ONLY", 20, "ALL_ELIGIBLE") == []


def test_database_transfer_summary_reconciles_all_result_tables():
    radar = pd.DataFrame([{"ticker": "A.JK"}, {"ticker": "B.JK"}])
    result = {
        "radar": radar,
        "expected_events": 3,
        "expected_provider_audit": 4,
        "expected_direct_evidence": 0,
        "expected_autonomous_evidence": 6,
        "expected_outcomes": 0,
    }
    counts = {
        "cak_scan_runs": 1,
        "cak_radar_snapshots": 2,
        "cak_narrative_events": 3,
        "cak_provider_audit": 4,
        "cak_direct_evidence": 0,
        "cak_autonomous_evidence": 6,
        "cak_outcome_memory": 0,
    }
    result["write_report"] = pd.DataFrame([
        {"table": table, "rows_written": count, "state": "WRITTEN" if count else "EMPTY_EXPECTED"}
        for table, count in counts.items()
    ])
    result["verification"] = pd.DataFrame([
        {"table": table, "rows_verified": count, "state": "VERIFIED_EXACT"}
        for table, count in counts.items()
    ])
    summary = build_database_transfer_summary(result)
    totals = database_transfer_totals(summary)
    assert len(summary) == 7
    assert set(summary["transfer_state"]) <= {"VERIFIED_IN_DATABASE", "VERIFIED_EMPTY"}
    assert totals == {
        "expected": sum(counts.values()),
        "written": sum(counts.values()),
        "verified": sum(counts.values()),
        "state": "DATABASE_RESULT_TRANSFER_VERIFIED",
    }


def test_database_transfer_summary_separates_outcome_maintenance_observations():
    result = {
        "radar": pd.DataFrame([{"ticker": "A.JK"}]),
        "expected_events": 0,
        "expected_provider_audit": 0,
        "expected_direct_evidence": 0,
        "expected_autonomous_evidence": 0,
        "expected_outcomes": 0,
    }
    counts = {
        "cak_scan_runs": 1,
        "cak_radar_snapshots": 1,
        "cak_narrative_events": 0,
        "cak_provider_audit": 0,
        "cak_direct_evidence": 0,
        "cak_autonomous_evidence": 0,
        "cak_outcome_memory": 0,
    }
    result["write_report"] = pd.DataFrame([
        {"table": table, "rows_written": count, "state": "WRITTEN" if count else "EMPTY_EXPECTED"}
        for table, count in counts.items()
    ])
    result["verification"] = pd.DataFrame([
        {
            "table": table,
            "rows_verified": count,
            "rows_observed": 180 if table == "cak_outcome_memory" else count,
            "state": "VERIFIED_AT_LEAST_EXPECTED" if table == "cak_outcome_memory" else "VERIFIED_EXACT",
        }
        for table, count in counts.items()
    ])

    summary = build_database_transfer_summary(result)
    outcome = summary.loc[summary["table"].eq("cak_outcome_memory")].iloc[0]
    totals = database_transfer_totals(summary)

    assert outcome["expected_rows"] == 0
    assert outcome["verified_rows"] == 0
    assert outcome["observed_rows"] == 180
    assert outcome["transfer_state"] == "VERIFIED_WITH_MAINTENANCE_ROWS"
    assert totals == {
        "expected": 2,
        "written": 2,
        "verified": 2,
        "state": "DATABASE_RESULT_TRANSFER_VERIFIED",
    }


def test_dashboard_has_one_click_scan_and_rescan_buttons():
    source = (ROOT / "app.py").read_text()
    assert "🚀 Mulai Scan" in source
    assert "🔄 Scan Ulang Universe Ini" in source
    assert "🔄 Scan Ulang dari Dashboard" in source
    assert '"deep_review_scope": deep_review_scope' in source
    assert "DATABASE_RESULT_TRANSFER_VERIFIED" in source


def test_dashboard_scores_are_in_radar_database_payload(monkeypatch):
    import persistence
    from persistence import DatabaseConfig, persist_scan

    captured: dict[str, list[dict]] = {}

    def fake_post(_config, *, table, conflict, payload, chunk_size, return_rows=True):
        captured[table] = payload
        return len(payload)

    monkeypatch.setattr(persistence, "_post_payload_in_chunks", fake_post)
    monkeypatch.setattr(persistence, "_set_scan_status", lambda *_args, **_kwargs: "PERSISTED_PENDING_READBACK")
    radar = pd.DataFrame([{
        "ticker": "ELSA.JK",
        "emir_decision_state": "EMIR_WAIT_NARRATIVE",
        "action": "WAIT_STORY_CONFIRMATION",
        "emir_conviction_score": 72.5,
        "emir_evidence_coverage_pct": 81.0,
        "production_ready": False,
        "emir_final_score": 72.5,
        "dashboard_flow_score": 77.0,
        "dashboard_silent_accum_score": 74.0,
        "dashboard_recommendation": "WAIT NARRATIVE",
    }])
    config = DatabaseConfig(True, "https://fixture.supabase.co", "sb_secret_fixture", key_type="SECRET")
    report = persist_scan(
        config,
        scan_id="scan-dashboard-persist",
        as_of=pd.Timestamp("2026-08-06T20:00:00+07:00"),
        radar=radar,
        events=pd.DataFrame(),
        provider_audit=pd.DataFrame(),
        direct_evidence=pd.DataFrame(),
        autonomous_evidence=pd.DataFrame(),
        outcomes=pd.DataFrame(),
    )
    assert str(report.iloc[0]["state"]) == "WRITE_ALL_TABLES"
    payload = captured["cak_radar_snapshots"][0]["payload"]
    assert payload["emir_final_score"] == 72.5
    assert payload["dashboard_flow_score"] == 77.0
    assert payload["dashboard_silent_accum_score"] == 74.0
    assert payload["dashboard_recommendation"] == "WAIT NARRATIVE"


def test_daily_recall_150_is_bounded_and_preserves_growth_turnaround_recall():
    fast = _fast_frame(220, failed=0)
    # Put a technically lower-ranked name well outside the primary 55-name core.
    rescue = "T210.JK"
    fundamentals = pd.DataFrame([{
        "ticker": rescue,
        "fundamental_raw_score": 78.0,
        "fundamental_conversion_score": 80.0,
        "revenue_growth_yoy_pct": 28.0,
        "earnings_growth_yoy_pct": 45.0,
        "roe_ttm_pct": 18.0,
        "fundamental_solvency_score": 88.0,
        "fundamental_data_quality_score": 84.0,
        "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
        "earnings_growth_yoy_state": "NORMAL",
    }])
    shortlist = choose_shortlist(
        fast,
        sector_map={},
        scan_mode="EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
        deep_limit=100,
        deep_review_scope="DAILY_RECALL_150",
        fundamental_recall=fundamentals,
    )
    assert len(shortlist) == 150
    assert rescue in shortlist
    assert len(shortlist) == len(set(shortlist))


def test_daily_recall_150_does_not_require_fundamental_data():
    fast = _fast_frame(220, failed=4)
    shortlist = choose_shortlist(
        fast,
        sector_map={},
        scan_mode="EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
        deep_limit=100,
        deep_review_scope="DAILY_RECALL_150",
        fundamental_recall=pd.DataFrame(),
    )
    assert len(shortlist) == 150
    assert all(ticker not in shortlist for ticker in ["T000.JK", "T001.JK", "T002.JK", "T003.JK"])


def test_explicit_full_deep_refresh_still_reviews_every_eligible_ticker():
    fast = _fast_frame(120, failed=3)
    shortlist = choose_shortlist(
        fast,
        sector_map={},
        scan_mode="EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
        deep_limit=80,
        deep_review_scope="ALL_ELIGIBLE",
    )
    assert len(shortlist) == 117


def test_dashboard_defaults_to_recall_150_not_all_eligible():
    source = (ROOT / "app.py").read_text()
    assert '"Optimal harian — Recall 150"' in source
    assert '"Optimal harian — Recall 150": "DAILY_RECALL_80"' in source
    assert '"Semua ticker eligible (full deep refresh)": "ALL_ELIGIBLE"' in source
    assert '"Batas IDX official deep review"' in source
    assert "\n        150,\n        10," in source


def test_daily_recall_150_records_selection_lane_reasons():
    fast = _fast_frame(220, failed=0)
    rescue = "T210.JK"
    fundamentals = pd.DataFrame([{
        "ticker": rescue,
        "fundamental_raw_score": 82.0,
        "fundamental_conversion_score": 84.0,
        "revenue_growth_yoy_pct": 32.0,
        "earnings_growth_yoy_pct": 48.0,
        "roe_ttm_pct": 20.0,
        "fundamental_solvency_score": 90.0,
        "fundamental_data_quality_score": 88.0,
        "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
        "earnings_growth_yoy_state": "NORMAL",
    }])
    reasons = {}
    shortlist = choose_shortlist(
        fast,
        sector_map={},
        scan_mode="EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP",
        deep_limit=150,
        deep_review_scope="DAILY_RECALL_150",
        fundamental_recall=fundamentals,
        selection_reasons=reasons,
    )
    assert len(shortlist) == 150
    assert reasons[rescue] == "FUNDAMENTAL_RECALL"
    assert set(reasons).issubset(set(shortlist))
    assert set(reasons.values()).issubset({
        "CORE_RANK",
        "FUNDAMENTAL_RECALL",
        "SMART_MONEY_RECALL",
        "STRUCTURE_RECALL",
        "REVERSAL_RECALL",
        "CORE_RANK_FILL",
    })
