from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd

import ksei_monthly_holding_composition_patch as patch


def _zip_bytes() -> bytes:
    header = [
        "Date", "Code", "Type", "Sec. Num", "Price",
        "Local IS", "Local CP", "Local PF", "Local IB", "Local ID", "Local MF", "Local SC", "Local FD", "Local OT", "Total",
        "Foreign IS", "Foreign CP", "Foreign PF", "Foreign IB", "Foreign ID", "Foreign MF", "Foreign SC", "Foreign FD", "Foreign OT", "Total",
    ]
    # 1,000 issued; 800 scripless = 600 local + 200 foreign. Individual = 100 + 50.
    row = [
        "31-AUG-2026", "TEST", "EQUITY", "1000", "100",
        "50", "300", "20", "20", "100", "50", "30", "10", "20", "600",
        "20", "50", "10", "20", "50", "20", "10", "5", "15", "200",
    ]
    text = "|".join(header) + "\n" + "|".join(row) + "\n"
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("BalanceposEfek20260831.txt", text)
    return buffer.getvalue()


def test_parse_official_balancepos_archive_into_profile_semantics() -> None:
    frame = patch.parse_balancepos_zip(
        _zip_bytes(),
        source_url="https://web.ksei.co.id/Download/BalanceposEfek20260831.zip",
        observed_on="20260831",
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["ticker"] == "TEST.JK"
    assert row["ksei_source_verified"] is True or bool(row["ksei_source_verified"])
    assert round(float(row["local_pct"]), 2) == 75.00
    assert round(float(row["foreign_pct"]), 2) == 25.00
    assert round(float(row["scripless_pct"]), 2) == 80.00
    assert float(row["total_shares"]) == 1000.0
    assert float(row["ksei_monthly_institutional_shares"]) == 650.0
    assert row["ksei_monthly_holding_composition_state"] == "OFFICIAL_KSEI_SCRIPLESS_COMPOSITION_NOT_REGULATORY_FREE_FLOAT"


def test_non_equity_rows_are_not_promoted_to_stock_ownership() -> None:
    raw = _zip_bytes()
    # Parser fixture is deliberately simple; mutate EQUITY to BOND in the ZIP payload.
    from zipfile import ZipFile
    buffer = BytesIO()
    with ZipFile(BytesIO(raw)) as source, ZipFile(buffer, "w") as target:
        text = source.read(source.namelist()[0]).decode().replace("|EQUITY|", "|BOND|")
        target.writestr("BalanceposEfek20260831.txt", text)
    assert patch.parse_balancepos_zip(buffer.getvalue()).empty


def test_wrapper_preserves_verified_per_security_profile_and_fills_missing() -> None:
    def original(tickers, max_workers=2):
        profiles = pd.DataFrame([{
            "ticker": "KEEP.JK",
            "ksei_source_verified": True,
            "local_pct": 90.0,
            "foreign_pct": 10.0,
        }])
        return profiles, pd.DataFrame(), pd.DataFrame()

    fake = SimpleNamespace(fetch_many_ksei_profiles=original)
    monthly = pd.DataFrame([
        {"ticker": "KEEP.JK", "ksei_source_verified": True, "local_pct": 70.0, "ksei_monthly_holding_composition_state": "OFFICIAL_KSEI_SCRIPLESS_COMPOSITION_NOT_REGULATORY_FREE_FLOAT"},
        {"ticker": "FILL.JK", "ksei_source_verified": True, "local_pct": 60.0, "foreign_pct": 40.0, "ksei_monthly_holding_composition_state": "OFFICIAL_KSEI_SCRIPLESS_COMPOSITION_NOT_REGULATORY_FREE_FLOAT"},
    ])
    old = patch.fetch_monthly_profiles
    try:
        patch.fetch_monthly_profiles = lambda tickers: (monthly.loc[monthly["ticker"].isin(tickers)].copy(), pd.DataFrame())
        patch._wrap_fetch_many(fake)
        profiles, _, _ = fake.fetch_many_ksei_profiles(["KEEP.JK", "FILL.JK"])
    finally:
        patch.fetch_monthly_profiles = old

    keep = profiles.loc[profiles["ticker"].eq("KEEP.JK")].iloc[0]
    fill = profiles.loc[profiles["ticker"].eq("FILL.JK")].iloc[0]
    assert float(keep["local_pct"]) == 90.0
    assert float(fill["local_pct"]) == 60.0


def test_source_semantics_never_claim_regulatory_free_float() -> None:
    source = open("ksei_monthly_holding_composition_patch.py", encoding="utf-8").read()
    assert "OFFICIAL_KSEI_SCRIPLESS_COMPOSITION_NOT_REGULATORY_FREE_FLOAT" in source
    assert '"reported_free_float_pct"' not in source
    assert '"effective_free_float_pct"' not in source
