from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import ksei_monthly_field_completion_patch as patch


def _monthly_row(ticker: str = "TEST.JK") -> dict:
    return {
        "ticker": ticker,
        "total_shares": 1000.0,
        "scripless_pct": 80.0,
        "local_pct": 75.0,
        "foreign_pct": 25.0,
        "ksei_source_url": "https://web.ksei.co.id/Download/BalanceposEfek20260831.zip",
        "ksei_source_verified": True,
        "ksei_observed_on": "20260831",
        "ksei_monthly_local_total_shares": 600.0,
        "ksei_monthly_foreign_total_shares": 200.0,
        "ksei_monthly_institutional_shares": 650.0,
        "ksei_monthly_holding_composition_state": patch.COMPOSITION_STATE,
    }


def test_verified_profile_with_empty_composition_is_not_treated_as_complete() -> None:
    row = {
        "ticker": "TEST.JK",
        "ksei_source_verified": True,
        "total_shares": 1000.0,
        "scripless_pct": np.nan,
        "local_pct": 0.0,
        "foreign_pct": 0.0,
    }
    assert patch._composition_complete(row) is False


def test_monthly_completion_preserves_verified_share_count_and_fills_composition() -> None:
    existing = {
        "ticker": "TEST.JK",
        "ksei_source_verified": True,
        "total_shares": 1234.0,
        "scripless_pct": np.nan,
        "local_pct": 0.0,
        "foreign_pct": 0.0,
        "ksei_source_url": "https://web.ksei.co.id/per-security/TEST",
    }
    merged, changed = patch._merge_monthly_into_existing(existing, _monthly_row())

    assert changed is True
    assert float(merged["total_shares"]) == 1234.0
    assert float(merged["scripless_pct"]) == 80.0
    assert float(merged["local_pct"]) == 75.0
    assert float(merged["foreign_pct"]) == 25.0
    assert merged["ksei_source_url"] == existing["ksei_source_url"]
    assert merged["ksei_monthly_source_url"].endswith("BalanceposEfek20260831.zip")
    assert merged["ksei_monthly_holding_composition_state"] == patch.COMPOSITION_STATE


def test_legitimate_zero_hundred_composition_is_preserved() -> None:
    existing = {
        "ticker": "TEST.JK",
        "ksei_source_verified": True,
        "total_shares": 1000.0,
        "scripless_pct": 50.0,
        "local_pct": 0.0,
        "foreign_pct": 100.0,
    }
    assert patch._composition_complete(existing) is True
    merged, changed = patch._merge_monthly_into_existing(existing, _monthly_row())
    assert changed is True  # monthly metadata can still be attached
    assert float(merged["scripless_pct"]) == 50.0
    assert float(merged["local_pct"]) == 0.0
    assert float(merged["foreign_pct"]) == 100.0


def test_runtime_wrapper_supplements_incomplete_verified_rows(monkeypatch) -> None:
    def original(tickers, max_workers=2):
        return pd.DataFrame([{
            "ticker": "TEST.JK",
            "ksei_source_verified": True,
            "total_shares": 1234.0,
            "scripless_pct": np.nan,
            "local_pct": 0.0,
            "foreign_pct": 0.0,
        }]), pd.DataFrame(), pd.DataFrame()

    fake = SimpleNamespace(fetch_many_ksei_profiles=original)
    monkeypatch.setattr(
        patch.monthly,
        "fetch_monthly_profiles",
        lambda tickers: (pd.DataFrame([_monthly_row("TEST.JK")]), pd.DataFrame()),
    )
    patch._wrap_fetch_many(fake)
    profiles, _, audit = fake.fetch_many_ksei_profiles(["TEST.JK"])

    row = profiles.iloc[0]
    assert float(row["total_shares"]) == 1234.0
    assert float(row["scripless_pct"]) == 80.0
    assert float(row["local_pct"]) == 75.0
    assert float(row["foreign_pct"]) == 25.0
    assert audit.loc[audit["provider"].eq("KSEI_MONTHLY_FIELD_COMPLETION"), "items"].iloc[0] == 1


def test_runtime_release_installs_field_completion_after_monthly_fallback() -> None:
    source = open("runtime_release.py", encoding="utf-8").read()
    base = source.index('_try_optional_patch("ksei_monthly_holding_composition_patch", "install")')
    completion = source.index('_try_optional_patch("ksei_monthly_field_completion_patch", "install")')
    assert completion > base
