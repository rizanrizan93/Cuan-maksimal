from __future__ import annotations

import json
import pandas as pd

import persistence
from dashboard_persistence import build_database_transfer_summary, database_transfer_totals
from persistence import DatabaseConfig, persist_scan


def main() -> None:
    radar = pd.DataFrame([
        {
            "ticker": "A.JK",
            "emir_decision_state": "EMIR_WAIT_NARRATIVE",
            "action": "WAIT_STORY_CONFIRMATION",
            "emir_conviction_score": 72.5,
            "emir_evidence_coverage_pct": 81.0,
            "production_ready": False,
            "emir_final_score": 72.5,
            "dashboard_flow_score": 77.0,
            "dashboard_silent_accum_score": 74.0,
            "dashboard_recommendation": "WAIT NARRATIVE",
        },
        {
            "ticker": "B.JK",
            "emir_decision_state": "EMIR_WATCH_INVENTORY_COLLECTION",
            "action": "WATCH_SMART_MONEY_COLLECTION",
            "emir_conviction_score": 70.0,
            "emir_evidence_coverage_pct": 78.0,
            "production_ready": False,
            "emir_final_score": 70.0,
            "dashboard_flow_score": 80.0,
            "dashboard_silent_accum_score": 82.0,
            "dashboard_recommendation": "WATCH ACCUM",
        },
    ])
    captured: dict[str, list[dict]] = {}

    def fake_post(_config, *, table, conflict, payload, chunk_size, return_rows=True):
        captured[table] = payload
        return len(payload)

    old_post = persistence._post_payload_in_chunks
    old_status = persistence._set_scan_status
    persistence._post_payload_in_chunks = fake_post
    persistence._set_scan_status = lambda *_args, **_kwargs: "PERSISTED_PENDING_READBACK"
    try:
        config = DatabaseConfig(True, "https://fixture.supabase.co", "sb_secret_fixture", key_type="SECRET")
        write_report = persist_scan(
            config,
            scan_id="db-transfer-v163",
            as_of=pd.Timestamp("2026-08-06T20:00:00+07:00"),
            radar=radar,
            events=pd.DataFrame([{"ticker": "A.JK", "title": "Material event"}]),
            provider_audit=pd.DataFrame([{"ticker": "A.JK", "provider": "FIXTURE", "status": "OK"}]),
            direct_evidence=pd.DataFrame(),
            autonomous_evidence=pd.DataFrame([{"ticker": "A.JK", "evidence_type": "OHLCV_PROXY"}]),
            outcomes=pd.DataFrame(),
        )
    finally:
        persistence._post_payload_in_chunks = old_post
        persistence._set_scan_status = old_status

    expected = {
        "cak_scan_runs": 1,
        "cak_radar_snapshots": 2,
        "cak_narrative_events": 1,
        "cak_provider_audit": 1,
        "cak_direct_evidence": 0,
        "cak_autonomous_evidence": 1,
        "cak_outcome_memory": 0,
    }
    verification = pd.DataFrame([
        {
            "table": table,
            "rows_verified": count,
            "state": "VERIFIED_EXACT",
        }
        for table, count in expected.items()
    ])
    result = {
        "radar": radar,
        "expected_events": 1,
        "expected_provider_audit": 1,
        "expected_direct_evidence": 0,
        "expected_autonomous_evidence": 1,
        "expected_outcomes": 0,
        "write_report": write_report,
        "verification": verification,
    }
    summary = build_database_transfer_summary(result)
    totals = database_transfer_totals(summary)
    payload = captured["cak_radar_snapshots"][0]["payload"]
    assert payload["emir_final_score"] == 72.5
    assert payload["dashboard_flow_score"] == 77.0
    assert payload["dashboard_silent_accum_score"] == 74.0
    assert totals["state"] == "DATABASE_RESULT_TRANSFER_VERIFIED"
    assert totals["expected"] == totals["written"] == totals["verified"] == sum(expected.values())
    output = {
        "state": "PASS",
        "result_tables": len(summary),
        "expected_rows": totals["expected"],
        "written_rows": totals["written"],
        "verified_rows": totals["verified"],
        "transfer_state": totals["state"],
        "dashboard_fields_in_radar_payload": [
            "emir_final_score",
            "dashboard_flow_score",
            "dashboard_silent_accum_score",
            "dashboard_recommendation",
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
