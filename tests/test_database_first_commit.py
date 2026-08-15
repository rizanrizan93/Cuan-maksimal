from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import persistence as ps  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._payload


def ready_config() -> ps.DatabaseConfig:
    return ps.config_from_mapping({
        "CAK_DATABASE_ENABLED": "true",
        "SUPABASE_URL": "https://x.supabase.co",
        "SUPABASE_SECRET_KEY": "sb_secret_example",
    })


def radar_frame(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ticker": f"T{i:03d}.JK",
            "emir_decision_state": "EMIR_NO_EDGE_YET",
            "action": "WAIT_FOR_EDGE",
            "emir_conviction_score": 50.0 + i,
            "emir_evidence_coverage_pct": 70.0,
            "production_ready": False,
        }
        for i in range(n)
    ])


def in_memory_request(store: dict[str, list[dict]], *, inject_table: str | None = None, extra_readback_table: str | None = None):
    key_fields = {
        "cak_scan_runs": ("scan_id",),
        "cak_radar_snapshots": ("scan_id", "ticker"),
        "cak_narrative_events": ("event_id",),
        "cak_provider_audit": ("audit_id",),
        "cak_direct_evidence": ("evidence_id",),
        "cak_autonomous_evidence": ("evidence_id",),
        "cak_outcome_memory": ("outcome_id",),
    }

    def request(config, method, table, *, params=None, payload=None, **kwargs):
        scan_id = str((params or {}).get("scan_id", "eq.")).removeprefix("eq.")
        if method == "POST":
            if table == inject_table:
                raise RuntimeError("injected database write failure")
            incoming = list(payload or [])
            rows = store.setdefault(table, [])
            for item in incoming:
                keys = key_fields[table]
                rows[:] = [row for row in rows if not all(row.get(k) == item.get(k) for k in keys)]
                rows.append(dict(item))
            return FakeResponse(incoming)
        if method == "PATCH":
            updated = []
            for row in store.get(table, []):
                if row.get("scan_id") == scan_id:
                    row.update(dict(payload or {}))
                    updated.append(dict(row))
            return FakeResponse(updated)
        rows = [dict(row) for row in store.get(table, []) if row.get("scan_id") == scan_id]
        if table == extra_readback_table:
            rows.append({"scan_id": scan_id})
        return FakeResponse(rows)

    return request


def test_database_first_commit_requires_exact_write_and_readback(monkeypatch):
    store: dict[str, list[dict]] = {}
    monkeypatch.setattr(ps, "_request", in_memory_request(store))
    write, readback, commit = ps.persist_verify_commit_scan(
        ready_config(),
        scan_id="scan-commit-ok",
        as_of="2026-08-05T13:00:00+07:00",
        radar=radar_frame(3),
        events=pd.DataFrame(),
        provider_audit=pd.DataFrame([{"ticker": "T000.JK", "provider": "FIXTURE", "status": "OK"}]),
        direct_evidence=pd.DataFrame(),
        autonomous_evidence=pd.DataFrame(),
        outcomes=pd.DataFrame(),
        chunk_size=2,
    )
    assert write.iloc[0]["state"] == "WRITE_ALL_TABLES"
    assert readback.iloc[0]["state"] == "VERIFIED_ALL_TABLES"
    assert readback.iloc[0]["verification_pct"] == 100.0
    assert ps.database_commit_succeeded(commit)
    assert store["cak_scan_runs"][0]["status"] == "VERIFIED_COMMITTED"


def test_database_first_commit_blocks_partial_write(monkeypatch):
    store: dict[str, list[dict]] = {}
    monkeypatch.setattr(ps, "_request", in_memory_request(store, inject_table="cak_radar_snapshots"))
    write, readback, commit = ps.persist_verify_commit_scan(
        ready_config(),
        scan_id="scan-write-fail",
        as_of="2026-08-05T13:00:00+07:00",
        radar=radar_frame(2),
        events=pd.DataFrame(),
        provider_audit=pd.DataFrame(),
        direct_evidence=pd.DataFrame(),
    )
    assert write.iloc[0]["state"] == "WRITE_PARTIAL"
    assert readback.iloc[0]["state"] == "READBACK_SKIPPED_WRITE_INCOMPLETE"
    assert not ps.database_commit_succeeded(commit)
    assert store["cak_scan_runs"][0]["status"] == "COMMIT_FAILED"


def test_exact_readback_rejects_unexpected_extra_rows(monkeypatch):
    store: dict[str, list[dict]] = {}
    monkeypatch.setattr(ps, "_request", in_memory_request(store, extra_readback_table="cak_provider_audit"))
    write, readback, commit = ps.persist_verify_commit_scan(
        ready_config(),
        scan_id="scan-extra-row",
        as_of="2026-08-05T13:00:00+07:00",
        radar=radar_frame(1),
        events=pd.DataFrame(),
        provider_audit=pd.DataFrame([{"ticker": "T000.JK", "provider": "FIXTURE", "status": "OK"}]),
        direct_evidence=pd.DataFrame(),
    )
    assert write.iloc[0]["state"] == "WRITE_ALL_TABLES"
    assert readback.iloc[0]["state"] == "PARTIAL_READBACK"
    mismatch = readback.loc[readback["table"].eq("cak_provider_audit")].iloc[0]
    assert mismatch["state"] == "ROW_COUNT_MISMATCH"
    assert not ps.database_commit_succeeded(commit)


def test_app_uses_resumable_jobs_and_best_effort_result_persistence():
    source = (ROOT / "app.py").read_text()
    assert "create_scan_job" in source
    assert "process_next_job_step" in source
    assert "Lanjut otomatis" in source
    assert "Proses 1 checkpoint" in source
    assert "SCAN_NOT_COMMITTED" not in source
    assert "CACHE_NOT_COMMITTED" not in source
    assert "HEALTHY_EMIR_DATABASE_V9" in source
    assert "HEALTHY_EMIR_DATABASE_V8" not in source


def test_best_effort_publishes_partial_write(monkeypatch):
    store: dict[str, list[dict]] = {}
    monkeypatch.setattr(ps, "_request", in_memory_request(store, inject_table="cak_autonomous_evidence"))
    write, readback, persistence = ps.persist_verify_scan_best_effort(
        ready_config(),
        scan_id="scan-partial-publish",
        as_of="2026-08-05T13:00:00+07:00",
        radar=radar_frame(2),
        events=pd.DataFrame(),
        provider_audit=pd.DataFrame([{"ticker": "T000.JK", "provider": "FIXTURE", "status": "OK"}]),
        direct_evidence=pd.DataFrame(),
        autonomous_evidence=pd.DataFrame([{"ticker": "T000.JK", "evidence_type": "AUTO"}]),
        outcomes=pd.DataFrame(),
    )
    assert write.iloc[0]["state"] == "WRITE_PARTIAL"
    assert readback.iloc[0]["state"] == "PARTIAL_READBACK"
    assert persistence.iloc[0]["state"] == "SCAN_COMPLETED_PARTIAL_PERSISTENCE"
    assert bool(persistence.iloc[0]["publishable"]) is True
    assert ps.scan_publication_allowed(persistence)
    assert store["cak_scan_runs"][0]["status"] == "PARTIAL_PERSISTENCE"


def test_best_effort_publishes_memory_only_when_database_disabled():
    disabled = ps.config_from_mapping({})
    write, readback, persistence = ps.persist_verify_scan_best_effort(
        disabled,
        scan_id="scan-memory",
        as_of="2026-08-05T13:00:00+07:00",
        radar=radar_frame(1),
        events=pd.DataFrame(),
        provider_audit=pd.DataFrame(),
        direct_evidence=pd.DataFrame(),
    )
    assert write.iloc[0]["state"] == "DATABASE_DISABLED"
    assert readback.iloc[0]["state"] == "DATABASE_DISABLED"
    assert persistence.iloc[0]["state"] == "SCAN_COMPLETED_MEMORY_ONLY"
    assert ps.scan_publication_allowed(persistence)


def test_exact_readback_uses_postgrest_content_range_above_1000(monkeypatch):
    class CountResponse(FakeResponse):
        def __init__(self, payload, total):
            super().__init__(payload)
            self.headers = {"Content-Range": f"0-0/{total}"}

    totals = {
        "cak_scan_runs": 1,
        "cak_radar_snapshots": 150,
        "cak_narrative_events": 255,
        "cak_provider_audit": 393,
        "cak_direct_evidence": 0,
        "cak_autonomous_evidence": 1232,
        "cak_outcome_memory": 0,
    }

    def request(config, method, table, *, params=None, payload=None, **kwargs):
        assert method == "GET"
        assert kwargs.get("extra_headers", {}).get("Prefer") == "count=exact"
        assert kwargs.get("extra_headers", {}).get("Range") == "0-0"
        total = totals[table]
        return CountResponse([{"scan_id": "scan-large"}] if total else [], total)

    monkeypatch.setattr(ps, "_request", request)
    result = ps.verify_scan(
        ready_config(),
        scan_id="scan-large",
        expected_radar=150,
        expected_events=255,
        expected_provider_audit=393,
        expected_direct_evidence=0,
        expected_autonomous_evidence=1232,
        expected_outcomes=0,
    )
    assert result.iloc[0]["state"] == "VERIFIED_ALL_TABLES"
    assert result.iloc[0]["rows_verified"] == 2031
    row = result.loc[result["table"].eq("cak_autonomous_evidence")].iloc[0]
    assert row["rows_verified"] == 1232
    assert row["verification_pct"] == 100.0


def test_exact_count_parser_falls_back_for_test_doubles_without_headers():
    response = FakeResponse([{"scan_id": "x"}, {"scan_id": "x"}])
    assert ps._exact_count_from_response(response) == 2


def test_outcome_readback_accepts_post_commit_maintenance_rows(monkeypatch):
    observed = {
        "cak_scan_runs": 1,
        "cak_radar_snapshots": 1,
        "cak_narrative_events": 0,
        "cak_provider_audit": 0,
        "cak_direct_evidence": 0,
        "cak_autonomous_evidence": 0,
        "cak_outcome_memory": 180,
    }
    monkeypatch.setattr(ps, "_count_rows_for_scan", lambda _config, table, _scan_id: observed[table])

    result = ps.verify_scan(
        ready_config(),
        scan_id="scan-maintained-outcomes",
        expected_radar=1,
        expected_events=0,
        expected_provider_audit=0,
        expected_direct_evidence=0,
        expected_autonomous_evidence=0,
        expected_outcomes=0,
    )

    assert result.iloc[0]["state"] == "VERIFIED_ALL_TABLES"
    assert result.iloc[0]["rows_verified"] == 2
    assert result.iloc[0]["rows_observed"] == 182
    outcome = result.loc[result["table"].eq("cak_outcome_memory")].iloc[0]
    assert outcome["state"] == "VERIFIED_AT_LEAST_EXPECTED"
    assert outcome["rows_verified"] == 0
    assert outcome["rows_observed"] == 180


def test_outcome_readback_still_rejects_missing_expected_rows(monkeypatch):
    observed = {
        "cak_scan_runs": 1,
        "cak_radar_snapshots": 1,
        "cak_narrative_events": 0,
        "cak_provider_audit": 0,
        "cak_direct_evidence": 0,
        "cak_autonomous_evidence": 0,
        "cak_outcome_memory": 2,
    }
    monkeypatch.setattr(ps, "_count_rows_for_scan", lambda _config, table, _scan_id: observed[table])

    result = ps.verify_scan(
        ready_config(),
        scan_id="scan-missing-outcomes",
        expected_radar=1,
        expected_events=0,
        expected_provider_audit=0,
        expected_direct_evidence=0,
        expected_autonomous_evidence=0,
        expected_outcomes=3,
    )

    assert result.iloc[0]["state"] == "PARTIAL_READBACK"
    outcome = result.loc[result["table"].eq("cak_outcome_memory")].iloc[0]
    assert outcome["state"] == "ROW_COUNT_MISMATCH"
    assert outcome["rows_observed"] == 2
