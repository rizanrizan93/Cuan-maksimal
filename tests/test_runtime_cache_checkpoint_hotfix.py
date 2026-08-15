from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import runtime_cache_checkpoint_hotfix as hotfix


def _invalid_refresh_row(ticker: str = "PRIM.JK"):
    return {
        "cache_key": f"KSEI:{ticker}",
        "ticker": ticker,
        "family": "KSEI",
        "content_sha256": "new-hash",
        "payload": {
            "profiles": [{
                "ticker": ticker,
                "company_name": "ROYAL PRIMA Tbk",
                "total_shares": None,
                "security_status": "UNRESOLVED_PROFILE_ROUTE",
                "ksei_source_verified": False,
            }],
            "actions": [],
        },
    }


def _valid_refresh_row(ticker: str = "VALID.JK"):
    return {
        "cache_key": f"KSEI:{ticker}",
        "ticker": ticker,
        "family": "KSEI",
        "content_sha256": "new-hash",
        "payload": {
            "profiles": [{
                "ticker": ticker,
                "company_name": "VALID EMITEN Tbk",
                "total_shares": 1_000_000,
                "security_status": "Active",
                "ksei_source_verified": True,
            }],
            "actions": [],
        },
    }


def test_unresolved_profile_matches_database_guard_contract():
    assert hotfix._write_would_be_guarded(_invalid_refresh_row())
    assert hotfix._ksei_profile_is_db_valid(_valid_refresh_row()["payload"])


def test_guarded_refresh_reuses_hash_valid_durable_row_and_drops_rejected_write():
    ticker = "PRIM.JK"
    old_profile = {
        "ticker": ticker,
        "company_name": "ROYAL PRIMA Tbk",
        "total_shares": None,
        "security_status": "UNRESOLVED_PROFILE_ROUTE",
        "ksei_source_verified": False,
    }
    old_action = {"ticker": ticker, "action_type": "Proxy Voting"}
    existing = {
        "ticker": ticker,
        "family": "KSEI",
        "content_sha256": "old-hash",
        "payload": {"profiles": [old_profile], "actions": [old_action]},
    }

    cache = SimpleNamespace(
        read_source_cache=lambda config, tickers, family: {ticker: existing},
        _row_hash_valid=lambda row: row is existing,
    )

    def original(config, tickers, **kwargs):
        fresh_profile = {
            "ticker": ticker,
            "company_name": "ROYAL PRIMA Tbk",
            "total_shares": None,
            "security_status": "UNRESOLVED_PROFILE_ROUTE",
            "ksei_source_verified": False,
        }
        return (
            pd.DataFrame([fresh_profile]),
            pd.DataFrame(),
            pd.DataFrame([{
                "ticker": ticker,
                "provider": "KSEI_SECURITY_PROFILE",
                "status": "REFRESHED",
                "detail": "attempt=4",
            }]),
            [_invalid_refresh_row(ticker)],
        )

    wrapped = hotfix._build_guarded_fetch(cache, original)
    profiles, actions, audit, writes = wrapped(SimpleNamespace(ready=True), [ticker])

    assert writes == []
    assert profiles.iloc[0]["ticker"] == ticker
    assert actions.iloc[0]["action_type"] == "Proxy Voting"
    assert audit.iloc[0]["status"] == "STALE_CACHE_FALLBACK_GUARD"
    assert audit.iloc[0]["provider"] == "SUPABASE_KSEI_CACHE"


def test_valid_verified_ksei_refresh_is_not_suppressed():
    ticker = "VALID.JK"
    valid = _valid_refresh_row(ticker)

    cache = SimpleNamespace(
        read_source_cache=lambda config, tickers, family: {},
        _row_hash_valid=lambda row: True,
    )

    def original(config, tickers, **kwargs):
        return (
            pd.DataFrame(valid["payload"]["profiles"]),
            pd.DataFrame(),
            pd.DataFrame([{"ticker": ticker, "status": "REFRESHED"}]),
            [valid],
        )

    wrapped = hotfix._build_guarded_fetch(cache, original)
    _, _, audit, writes = wrapped(SimpleNamespace(ready=True), [ticker])

    assert len(writes) == 1
    assert writes[0]["ticker"] == ticker
    assert audit.iloc[0]["status"] == "REFRESHED"
