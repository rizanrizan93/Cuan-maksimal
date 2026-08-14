import pandas as pd

import persistent_direct_evidence as pde


class DummyConfig:
    ready = True


def test_loader_keeps_only_verified_current_rows(monkeypatch):
    now = pd.Timestamp("2026-08-14T07:00:00Z")
    rows = [
        {
            "evidence_id": "a", "scan_id": "REF", "ticker": "MARK.JK",
            "evidence_type": "OWNERSHIP_FREE_FLOAT", "observed_at": "2026-03-27T00:00:00Z",
            "source_verified": True,
            "payload": {"free_float_pct": 19.73, "source_url": "https://issuer.example/pubex.pdf"},
        },
        {
            "evidence_id": "b", "scan_id": "REF", "ticker": "MARK.JK",
            "evidence_type": "OFFICIAL_FORWARD_EVENT", "observed_at": "2026-06-19T00:00:00Z",
            "source_verified": True,
            "payload": {"title": "capacity expansion", "url": "https://issuer.example/news", "source_tier": "ISSUER"},
        },
        {
            "evidence_id": "c", "scan_id": "REF", "ticker": "OLD.JK",
            "evidence_type": "IDX_INTEGRITY_REGULATORY", "observed_at": "2025-01-01T00:00:00Z",
            "source_verified": True, "payload": {"listing_board": "MAIN"},
        },
    ]

    class Response:
        def json(self):
            return rows

    monkeypatch.setattr(pde, "_request", lambda *args, **kwargs: Response())
    out = pde.load_verified_direct_evidence(DummyConfig(), ["MARK.JK", "OLD.JK"], as_of=now)
    assert len(out["ownership"]) == 1
    assert float(out["ownership"].iloc[0]["free_float_pct"]) == 19.73
    assert len(out["official_forward_events"]) == 1
    assert out["idx_integrity"].empty
    assert set(out["audit"]["status"]) == {"PERSISTED_VERIFIED_CURRENT", "PERSISTED_VERIFIED_STALE"}


def test_unverified_rows_are_never_promoted(monkeypatch):
    class Response:
        def json(self):
            return [{
                "evidence_id": "x", "scan_id": "REF", "ticker": "TEST.JK",
                "evidence_type": "OWNERSHIP_FREE_FLOAT", "observed_at": "2026-08-01T00:00:00Z",
                "source_verified": False, "payload": {"free_float_pct": 10.0},
            }]

    monkeypatch.setattr(pde, "_request", lambda *args, **kwargs: Response())
    out = pde.load_verified_direct_evidence(DummyConfig(), ["TEST.JK"], as_of="2026-08-14T00:00:00Z")
    assert out["ownership"].empty
