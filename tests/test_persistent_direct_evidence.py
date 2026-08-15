import pandas as pd

import persistent_direct_evidence as pde


class DummyConfig:
    ready = True


def _disable_governed_bridge(monkeypatch):
    # These tests isolate the persistent master/legacy loader. Governed raw-table
    # consumption has its own end-to-end regression in test_governed_evidence_bridge.
    monkeypatch.setattr(pde, "load_governed_evidence", lambda *args, **kwargs: {
        "official_forward_events": pd.DataFrame(),
        "management_capital_events": pd.DataFrame(),
        "audit": pd.DataFrame(),
    })


def test_loader_keeps_only_verified_current_rows(monkeypatch):
    now = pd.Timestamp("2026-08-14T07:00:00Z")
    rows = [
        {
            "evidence_key": "a", "source_scan_id": "REF", "ticker": "MARK.JK",
            "evidence_type": "OWNERSHIP_FREE_FLOAT", "observed_at": "2026-03-27T00:00:00Z",
            "source_verified": True, "revoked": False, "freshness_policy_days": 180,
            "payload": {"free_float_pct": 19.73, "source_url": "https://issuer.example/pubex.pdf"},
        },
        {
            "evidence_key": "b", "source_scan_id": "REF", "ticker": "MARK.JK",
            "evidence_type": "OFFICIAL_FORWARD_EVENT", "observed_at": "2026-06-19T00:00:00Z",
            "source_verified": True, "revoked": False, "freshness_policy_days": 540,
            "source_url": "https://issuer.example/news",
            "payload": {
                "title": "capacity expansion", "url": "https://issuer.example/news", "source_tier": "ISSUER",
                "source_verified": True, "source_quorum_verified": True, "source_quorum_count": 2,
                "entity_match_verified": True,
            },
        },
        {
            "evidence_key": "weak", "source_scan_id": "REF", "ticker": "MARK.JK",
            "evidence_type": "OFFICIAL_FORWARD_EVENT", "observed_at": "2026-06-20T00:00:00Z",
            "source_verified": True, "revoked": False, "freshness_policy_days": 540,
            "source_url": "https://issuer.example/weak-news",
            "payload": {"title": "unconfirmed expansion", "url": "https://issuer.example/weak-news", "source_tier": "ISSUER"},
        },
        {
            "evidence_key": "c", "source_scan_id": "REF", "ticker": "OLD.JK",
            "evidence_type": "IDX_INTEGRITY_REGULATORY", "observed_at": "2025-01-01T00:00:00Z",
            "source_verified": True, "revoked": False, "freshness_policy_days": 60,
            "payload": {"listing_board": "MAIN"},
        },
    ]

    class Response:
        def json(self):
            return rows

    monkeypatch.setattr(pde, "_request", lambda *args, **kwargs: Response())
    _disable_governed_bridge(monkeypatch)
    out = pde.load_verified_direct_evidence(DummyConfig(), ["MARK.JK", "OLD.JK"], as_of=now)
    assert len(out["ownership"]) == 1
    assert float(out["ownership"].iloc[0]["free_float_pct"]) == 19.73
    assert out["ownership"].iloc[0]["persistent_evidence_store"] == "cak_persistent_direct_evidence"
    assert len(out["official_forward_events"]) == 1
    assert out["official_forward_events"].iloc[0]["persistent_evidence_key"] == "b"
    assert out["idx_integrity"].empty
    statuses = set(out["audit"]["status"])
    assert "PERSISTED_VERIFIED_CURRENT" in statuses
    assert "PERSISTED_VERIFIED_STALE" in statuses
    assert "PERSISTED_FORWARD_BLOCKED_MISSING_STRICT_LINEAGE" in statuses


def test_unverified_and_revoked_rows_are_never_promoted(monkeypatch):
    rows = [
        {
            "evidence_key": "x", "source_scan_id": "REF", "ticker": "TEST.JK",
            "evidence_type": "OWNERSHIP_FREE_FLOAT", "observed_at": "2026-08-01T00:00:00Z",
            "source_verified": False, "revoked": False, "freshness_policy_days": 180,
            "payload": {"free_float_pct": 10.0},
        },
        {
            "evidence_key": "y", "source_scan_id": "REF", "ticker": "TEST.JK",
            "evidence_type": "OWNERSHIP_FREE_FLOAT", "observed_at": "2026-08-02T00:00:00Z",
            "source_verified": True, "revoked": True, "freshness_policy_days": 180,
            "payload": {"free_float_pct": 11.0},
        },
    ]

    class Response:
        def json(self):
            return rows

    monkeypatch.setattr(pde, "_request", lambda *args, **kwargs: Response())
    _disable_governed_bridge(monkeypatch)
    out = pde.load_verified_direct_evidence(DummyConfig(), ["TEST.JK"], as_of="2026-08-14T00:00:00Z")
    assert out["ownership"].empty


def test_empty_master_does_not_fall_back_to_legacy(monkeypatch):
    calls = []

    class Response:
        def json(self):
            return []

    def fake_request(config, method, table, **kwargs):
        calls.append(table)
        return Response()

    monkeypatch.setattr(pde, "_request", fake_request)
    _disable_governed_bridge(monkeypatch)
    out = pde.load_verified_direct_evidence(DummyConfig(), ["TEST.JK"], as_of="2026-08-14T00:00:00Z")
    assert calls == ["cak_persistent_direct_evidence"]
    assert out["ownership"].empty


def test_master_schema_failure_uses_legacy_fallback(monkeypatch):
    legacy_rows = [{
        "evidence_id": "legacy-1", "scan_id": "OLD", "ticker": "TEST.JK",
        "evidence_type": "OWNERSHIP_FREE_FLOAT", "observed_at": "2026-08-01T00:00:00Z",
        "source_verified": True,
        "payload": {"free_float_pct": 12.5, "source_url": "https://issuer.example/ownership"},
    }]

    class Response:
        def json(self):
            return legacy_rows

    def fake_request(config, method, table, **kwargs):
        if table == "cak_persistent_direct_evidence":
            raise RuntimeError("table not found")
        return Response()

    monkeypatch.setattr(pde, "_request", fake_request)
    _disable_governed_bridge(monkeypatch)
    out = pde.load_verified_direct_evidence(DummyConfig(), ["TEST.JK"], as_of="2026-08-14T00:00:00Z")
    assert len(out["ownership"]) == 1
    assert out["ownership"].iloc[0]["persistent_evidence_store"] == "cak_direct_evidence"
    assert "MASTER_UNAVAILABLE_LEGACY_FALLBACK" in set(out["audit"]["status"])
