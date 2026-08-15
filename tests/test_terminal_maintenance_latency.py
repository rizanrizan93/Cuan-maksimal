from pathlib import Path
from types import SimpleNamespace

import free_tier_storage


def test_v11_detaches_synchronous_terminal_housekeeping_trigger():
    migration = Path("database/migration_v11_terminal_maintenance_latency.sql").read_text(encoding="utf-8")

    assert "drop trigger if exists trg_cak_free_tier_housekeeping" in migration.lower()
    assert "on public.cak_scan_jobs" in migration.lower()


def test_outcome_maintenance_is_bounded_and_best_effort(monkeypatch):
    calls = []

    def fake_request(config, method, table, **kwargs):
        calls.append((method, table, kwargs))
        if table.endswith("cak_resolve_outcome_memory"):
            raise RuntimeError('HTTP 500: {"code":"57014","message":"canceling statement due to statement timeout"}')
        return SimpleNamespace(status_code=204)

    monkeypatch.setattr(free_tier_storage, "_request", fake_request)

    report = free_tier_storage.run_outcome_maintenance_best_effort(
        SimpleNamespace(ready=True),
        scan_id="scan-1",
        resolve_limit=50_000,
        seed_limit=5_000,
    )

    assert report["state"] == "PARTIAL"
    assert report["resolve_state"] == "FAILED_BEST_EFFORT"
    assert report["seed_state"] == "COMPLETED"
    assert calls[0][0:2] == ("POST", "rpc/cak_resolve_outcome_memory")
    assert calls[0][2]["payload"] == {"p_limit": 1000}
    assert calls[1][0:2] == ("POST", "rpc/cak_seed_outcomes_for_scan")
    assert calls[1][2]["payload"] == {"p_scan_id": "scan-1", "p_limit": 100}
    assert all(call[2]["return_rows"] is False for call in calls)
    assert all(call[2]["timeout"] == 20 for call in calls)


def test_terminal_commit_precedes_best_effort_maintenance():
    source = Path("resumable_scan.py").read_text(encoding="utf-8")

    commit_at = source.index("updated = update_scan_job_minimal")
    outcome_at = source.index("outcome_maintenance = run_outcome_maintenance_best_effort", commit_at)
    prune_at = source.index("storage_housekeeping = prune_scan_history_best_effort", outcome_at)

    assert commit_at < outcome_at < prune_at
