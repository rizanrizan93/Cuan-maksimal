from __future__ import annotations

import pandas as pd

from shared_ksei_runtime_context_patch import _latest_canonical_profiles
import ksei_monthly_field_completion_patch as completion


def _canonical_rows():
    base = {
        "category": "ksei-komposisi",
        "ticker": "APII",
        "report_date": "2026-08-31",
        "source_url": "https://web.ksei.co.id/Download/BalanceposEfek20260831.zip",
        "source_verified": True,
        "validation_state": "VALID",
    }
    return [
        {**base, "holder_classification": "KSEI_SECURITY_NUMBER", "shares_held": 1075760000.0, "ownership_percentage": None},
        {**base, "holder_classification": "KSEI_SCRIPLESS_TOTAL", "shares_held": 274625252.0, "ownership_percentage": 25.5283706403},
        {**base, "holder_classification": "KSEI_LOCAL_TOTAL", "shares_held": 57806215.0, "ownership_percentage": 21.0478326730},
        {**base, "holder_classification": "KSEI_FOREIGN_TOTAL", "shares_held": 216819037.0, "ownership_percentage": 78.9521673270},
    ]


def test_latest_canonical_profiles_maps_official_ksei_context() -> None:
    frame = _latest_canonical_profiles(_canonical_rows())
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["ticker"] == "APII.JK"
    assert round(float(row["scripless_pct"]), 3) == 25.528
    assert round(float(row["local_pct"]), 3) == 21.048
    assert round(float(row["foreign_pct"]), 3) == 78.952
    assert bool(row["ksei_source_verified"])


def test_shared_context_completes_incoherent_verified_profile() -> None:
    existing = pd.DataFrame([{
        "ticker": "APII.JK",
        "total_shares": 1075760000.0,
        "scripless_pct": None,
        "local_pct": 0.0,
        "foreign_pct": 0.0,
        "ksei_source_verified": True,
    }])
    shared = _latest_canonical_profiles(_canonical_rows())
    completed, changed = completion._supplement_profiles(existing, ["APII.JK"], shared)
    row = completed.iloc[0]
    assert changed == 1
    assert row["total_shares"] == 1075760000.0
    assert round(float(row["scripless_pct"]), 3) == 25.528
    assert round(float(row["local_pct"]), 3) == 21.048
    assert round(float(row["foreign_pct"]), 3) == 78.952


def test_complete_per_security_profile_remains_authoritative() -> None:
    existing = pd.DataFrame([{
        "ticker": "APII.JK",
        "total_shares": 1075760000.0,
        "scripless_pct": 80.0,
        "local_pct": 40.0,
        "foreign_pct": 60.0,
        "ksei_source_verified": True,
    }])
    shared = _latest_canonical_profiles(_canonical_rows())
    completed, changed = completion._supplement_profiles(existing, ["APII.JK"], shared)
    row = completed.iloc[0]
    assert changed == 0
    assert row["scripless_pct"] == 80.0
    assert row["local_pct"] == 40.0
    assert row["foreign_pct"] == 60.0
    assert not any("free_float" in str(column).lower() for column in completed.columns)


def test_valid_zero_hundred_composition_is_preserved() -> None:
    existing = pd.DataFrame([{
        "ticker": "APII.JK",
        "total_shares": 1075760000.0,
        "scripless_pct": 100.0,
        "local_pct": 0.0,
        "foreign_pct": 100.0,
        "ksei_source_verified": True,
    }])
    shared = _latest_canonical_profiles(_canonical_rows())
    completed, changed = completion._supplement_profiles(existing, ["APII.JK"], shared)
    assert changed == 0
    row = completed.iloc[0]
    assert row["local_pct"] == 0.0
    assert row["foreign_pct"] == 100.0
