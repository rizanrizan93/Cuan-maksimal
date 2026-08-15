from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


RESULT_TABLE_EXPECTED_KEYS = {
    "cak_scan_runs": None,
    "cak_radar_snapshots": "radar",
    "cak_narrative_events": "expected_events",
    "cak_provider_audit": "expected_provider_audit",
    "cak_direct_evidence": "expected_direct_evidence",
    "cak_autonomous_evidence": "expected_autonomous_evidence",
    "cak_outcome_memory": "expected_outcomes",
}


def _row_map(frame: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "table" not in frame.columns:
        return {}
    return {str(row.get("table") or ""): row for row in frame.to_dict(orient="records")}


def _expected_rows(result: Mapping[str, Any], table: str, key: str | None) -> int:
    if table == "cak_scan_runs":
        return 1
    if key == "radar":
        radar = result.get("radar")
        return len(radar) if isinstance(radar, pd.DataFrame) else 0
    return int(result.get(key or "", 0) or 0)


def build_database_transfer_summary(result: Mapping[str, Any]) -> pd.DataFrame:
    """Reconcile final result rows with Supabase write and exact-readback reports.

    This is a reporting helper only. It does not turn partial persistence into analytical
    evidence and it does not change scanner publication or quality gates.
    """
    write_map = _row_map(result.get("write_report"))
    verify_map = _row_map(result.get("verification"))
    rows: list[dict[str, Any]] = []
    for table, key in RESULT_TABLE_EXPECTED_KEYS.items():
        expected = _expected_rows(result, table, key)
        write_row = write_map.get(table, {})
        verify_row = verify_map.get(table, {})
        written = int(float(write_row.get("rows_written", 0) or 0))
        verified = int(float(verify_row.get("rows_verified", 0) or 0))
        observed = int(float(verify_row.get("rows_observed", verified) or 0))
        readback_state = str(verify_row.get("state", ""))
        contract_verified = readback_state in {"VERIFIED_EXACT", "VERIFIED_AT_LEAST_EXPECTED"}
        if expected == 0 and contract_verified and observed > 0:
            state = "VERIFIED_WITH_MAINTENANCE_ROWS"
        elif expected == 0 and contract_verified:
            state = "VERIFIED_EMPTY"
        elif expected > 0 and verified == expected and contract_verified:
            state = "VERIFIED_IN_DATABASE"
        elif expected > 0 and written == expected:
            state = "WRITTEN_NOT_FULLY_VERIFIED"
        elif written > 0:
            state = "PARTIAL_DATABASE_TRANSFER"
        else:
            state = "NOT_CONFIRMED"
        rows.append({
            "table": table,
            "expected_rows": expected,
            "written_rows": written,
            "verified_rows": verified,
            "observed_rows": observed,
            "transfer_state": state,
            "write_state": str(write_row.get("state", "") or ""),
            "readback_state": readback_state,
        })
    return pd.DataFrame(rows)


def database_transfer_totals(summary: pd.DataFrame) -> dict[str, int | str]:
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return {"expected": 0, "written": 0, "verified": 0, "state": "NOT_CONFIRMED"}
    expected = int(pd.to_numeric(summary["expected_rows"], errors="coerce").fillna(0).sum())
    written = int(pd.to_numeric(summary["written_rows"], errors="coerce").fillna(0).sum())
    verified = int(pd.to_numeric(summary["verified_rows"], errors="coerce").fillna(0).sum())
    if expected > 0 and verified == expected:
        state = "DATABASE_RESULT_TRANSFER_VERIFIED"
    elif written > 0:
        state = "DATABASE_RESULT_TRANSFER_PARTIAL"
    else:
        state = "DATABASE_RESULT_TRANSFER_NOT_CONFIRMED"
    return {"expected": expected, "written": written, "verified": verified, "state": state}


__all__ = ["RESULT_TABLE_EXPECTED_KEYS", "build_database_transfer_summary", "database_transfer_totals"]
