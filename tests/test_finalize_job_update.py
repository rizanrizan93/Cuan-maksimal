from types import SimpleNamespace

import scan_jobs


def test_minimal_job_update_avoids_returning_heavy_row(monkeypatch):
    calls = []

    def fake_request(config, method, table, **kwargs):
        calls.append((method, table, kwargs))
        return SimpleNamespace(status_code=204)

    monkeypatch.setattr(scan_jobs, "_request", fake_request)
    config = SimpleNamespace(ready=True)
    base = {"scan_id": "scan-1", "universe": [{"ticker": "MARK.JK"}], "settings": {"period": "5y"}}

    updated = scan_jobs.update_scan_job_minimal(
        config,
        "scan-1",
        {"status": "COMPLETED", "current_stage": "COMPLETED"},
        base_job=base,
    )

    assert updated["status"] == "COMPLETED"
    assert updated["universe"] == base["universe"]
    assert calls[0][0:2] == ("PATCH", "cak_scan_jobs")
    assert calls[0][2]["return_rows"] is False
    assert calls[0][2]["params"] == {"scan_id": "eq.scan-1"}
