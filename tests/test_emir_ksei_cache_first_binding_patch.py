from __future__ import annotations

import pandas as pd

import emir_ksei_cache_first_binding_patch as patch


def test_completes_cached_partial_profile(monkeypatch):
    monkeypatch.setattr(patch, "read_shared_profiles", lambda tickers=None: (
        pd.DataFrame([{
            "ticker": "APII.JK",
            "total_shares": 1075760000.0,
            "scripless_pct": 25.5283706403,
            "local_pct": 21.0478326730,
            "foreign_pct": 78.9521673270,
            "ksei_source_url": "https://web.ksei.co.id/Download/example.zip",
            "ksei_source_verified": True,
            "ksei_observed_on": "2026-08-31",
        }]),
        {"state": "SHARED_CANONICAL_KSEI", "rows": 1, "semantics": "OFFICIAL_KSEI_REGISTRATION_COMPOSITION_NOT_REGULATORY_FREE_FLOAT"},
    ))
    cached = pd.DataFrame([{
        "ticker": "APII.JK",
        "total_shares": 1075760000.0,
        "scripless_pct": float("nan"),
        "local_pct": 21.0478326730,
        "foreign_pct": 78.9521673270,
        "ksei_source_verified": True,
        "ksei_source_url": "https://web.ksei.co.id/security/APII",
    }])
    completed, audit = patch._complete_cache_first_profiles(cached, ["APII.JK"])
    row = completed.iloc[0]
    assert round(float(row["scripless_pct"]), 3) == 25.528
    assert round(float(row["local_pct"]), 3) == 21.048
    assert audit["changed"] == 1


def test_preserves_complete_cached_profile(monkeypatch):
    monkeypatch.setattr(patch, "read_shared_profiles", lambda tickers=None: (
        pd.DataFrame([{
            "ticker": "KEEP.JK", "total_shares": 1000,
            "scripless_pct": 50.0, "local_pct": 10.0, "foreign_pct": 90.0,
            "ksei_source_verified": True,
        }]),
        {"state": "SHARED_CANONICAL_KSEI", "rows": 1},
    ))
    cached = pd.DataFrame([{
        "ticker": "KEEP.JK", "total_shares": 1000,
        "scripless_pct": 75.0, "local_pct": 0.0, "foreign_pct": 100.0,
        "ksei_source_verified": True,
    }])
    completed, audit = patch._complete_cache_first_profiles(cached, ["KEEP.JK"])
    row = completed.iloc[0]
    assert float(row["scripless_pct"]) == 75.0
    assert float(row["local_pct"]) == 0.0
    assert audit["changed"] == 0


def test_requested_tickers_come_from_chunk():
    frame = pd.DataFrame([{"ticker": "OTHER.JK"}])
    assert patch._requested_from_call((object(), ["APII", "ARNA.JK"]), {}, frame) == ["APII.JK", "ARNA.JK"]
