from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import persistence as ps


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._payload


KEY_FIELDS = {
    "cak_scan_runs": ("scan_id",),
    "cak_radar_snapshots": ("scan_id", "ticker"),
    "cak_narrative_events": ("event_id",),
    "cak_provider_audit": ("audit_id",),
    "cak_direct_evidence": ("evidence_id",),
    "cak_autonomous_evidence": ("evidence_id",),
    "cak_outcome_memory": ("outcome_id",),
}


def config():
    return ps.config_from_mapping({
        "CAK_DATABASE_ENABLED": "true",
        "SUPABASE_URL": "https://fixture.supabase.co",
        "SUPABASE_SECRET_KEY": "sb_secret_fixture",
    })


def radar(n=400):
    return pd.DataFrame([
        {
            "ticker": f"T{i:03d}.JK",
            "emir_decision_state": "EMIR_NO_EDGE_YET",
            "action": "WAIT_FOR_EDGE",
            "emir_conviction_score": 50.0,
            "emir_evidence_coverage_pct": 70.0,
            "production_ready": False,
        }
        for i in range(n)
    ])


def request_factory(store, fail_table=None, extra_table=None):
    def request(config, method, table, *, params=None, payload=None, **kwargs):
        scan_id = str((params or {}).get("scan_id", "eq.")).removeprefix("eq.")
        if method == "POST":
            if table == fail_table:
                raise RuntimeError("injected write failure")
            incoming = list(payload or [])
            rows = store.setdefault(table, [])
            for item in incoming:
                keys = KEY_FIELDS[table]
                rows[:] = [row for row in rows if not all(row.get(k) == item.get(k) for k in keys)]
                rows.append(dict(item))
            return FakeResponse(incoming)
        if method == "PATCH":
            updated=[]
            for row in store.get(table, []):
                if row.get("scan_id") == scan_id:
                    row.update(dict(payload or {}))
                    updated.append(dict(row))
            return FakeResponse(updated)
        rows=[dict(row) for row in store.get(table, []) if row.get("scan_id") == scan_id]
        if table == extra_table:
            rows.append({"scan_id": scan_id})
        return FakeResponse(rows)
    return request


def execute(scan_id, request):
    original = ps._request
    ps._request = request
    try:
        write, readback, commit = ps.persist_verify_commit_scan(
            config(),
            scan_id=scan_id,
            as_of="2026-08-05T13:00:00+07:00",
            radar=radar(),
            events=pd.DataFrame(),
            provider_audit=pd.DataFrame([{"ticker": "T000.JK", "provider": "FIXTURE", "status": "OK"}]),
            direct_evidence=pd.DataFrame(),
            autonomous_evidence=pd.DataFrame(),
            outcomes=pd.DataFrame(),
            chunk_size=100,
        )
        return {
            "write_state": str(write.iloc[0]["state"]),
            "readback_state": str(readback.iloc[0]["state"]),
            "verification_pct": float(readback.iloc[0]["verification_pct"]),
            "commit_state": str(commit.iloc[0]["state"]),
            "observed_status": str(commit.iloc[0]["observed_status"]),
            "publishable": ps.database_commit_succeeded(commit),
        }
    finally:
        ps._request = original


def main():
    ok_store={}
    ok=execute("scan-ok", request_factory(ok_store))
    fail_store={}
    partial=execute("scan-partial", request_factory(fail_store, fail_table="cak_radar_snapshots"))
    extra_store={}
    extra=execute("scan-extra", request_factory(extra_store, extra_table="cak_provider_audit"))
    result={
        "scanner_version": ps.SCANNER_VERSION,
        "success_case": ok,
        "partial_write_case": partial,
        "unexpected_extra_row_case": extra,
        "assertions": {
            "success_is_publishable": ok["publishable"] and ok["observed_status"] == "VERIFIED_COMMITTED",
            "partial_write_is_blocked": not partial["publishable"] and partial["commit_state"] != "DATABASE_FIRST_COMMITTED",
            "row_count_mismatch_is_blocked": not extra["publishable"] and extra["readback_state"] == "PARTIAL_READBACK",
        },
    }
    assert all(result["assertions"].values())
    path=Path("validation_artifacts/DATABASE_FIRST_VALIDATION_V1_5_0.json")
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
