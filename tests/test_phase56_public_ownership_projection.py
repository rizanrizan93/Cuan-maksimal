from __future__ import annotations

from phase56_public_ownership_projection import _OwnershipContextCache, _rows_to_context, merge_public_context


def test_public_row_is_context_not_ksei_or_free_float() -> None:
    context = _rows_to_context([{
        "ticker": "BBCA",
        "source_period": "2026-09-01",
        "observed_on": "2026-09-04",
        "insiders_held_pct": 60.8,
        "institutions_held_pct": 19.0,
        "institutions_float_held_pct": 48.5,
        "institutions_count": 374,
        "coverage_pct": 100,
        "provenance_state": "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI",
    }])["BBCA"]
    assert context["ownership_public_context_coverage_pct"] == 100
    assert context["ownership_public_institutions_float_held_pct"] == 48.5
    assert "reported_free_float_pct" not in context
    assert "effective_free_float_pct" not in context
    assert "ownership_score" not in context
    assert "ownership_coverage_pct" not in context


def test_merge_preserves_direct_ownership_and_integrity_semantics() -> None:
    base = {
        "BBCA": {
            "ownership_score": 77.0,
            "ownership_coverage_pct": 62.5,
            "reported_free_float_pct": 42.0,
            "effective_free_float_pct": 40.5,
            "ownership_provenance_state": "DIRECT_SOURCE_VERIFIED",
        }
    }
    public = {
        "BBCA": {
            "ownership_public_institutions_held_pct": 19.0,
            "ownership_public_context_coverage_pct": 100.0,
            "ownership_public_context_provenance_state": "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI",
        }
    }
    merged = merge_public_context(base, public)["BBCA"]
    assert merged["ownership_score"] == 77.0
    assert merged["ownership_coverage_pct"] == 62.5
    assert merged["reported_free_float_pct"] == 42.0
    assert merged["effective_free_float_pct"] == 40.5
    assert merged["ownership_provenance_state"] == "DIRECT_SOURCE_VERIFIED"
    assert merged["ownership_public_institutions_held_pct"] == 19.0


def test_context_only_ticker_does_not_gain_score_or_free_float() -> None:
    public = {
        "MARK": {
            "ownership_public_insiders_held_pct": 70.0,
            "ownership_public_context_coverage_pct": 100.0,
            "ownership_public_context_provenance_state": "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI",
        }
    }
    merged = merge_public_context({}, public)["MARK"]
    assert "ownership_score" not in merged
    assert "ownership_coverage_pct" not in merged
    assert "reported_free_float_pct" not in merged
    assert "effective_free_float_pct" not in merged


def test_cache_refreshes_after_success_ttl_without_calling_each_scan() -> None:
    now = [0.0]
    calls = []

    def loader():
        calls.append(len(calls) + 1)
        value = 19.0 if len(calls) == 1 else 20.0
        return {
            "BBCA": {
                "ownership_public_institutions_held_pct": value,
                "ownership_public_context_coverage_pct": 100.0,
            }
        }, {"state": "PUBLIC_OWNERSHIP_LOADED", "tickers": 1}

    cache = _OwnershipContextCache(loader, ttl_seconds=100.0, retry_seconds=10.0, clock=lambda: now[0])
    first, _ = cache.get()
    assert first["BBCA"]["ownership_public_institutions_held_pct"] == 19.0
    now[0] = 99.0
    second, _ = cache.get()
    assert second["BBCA"]["ownership_public_institutions_held_pct"] == 19.0
    assert len(calls) == 1
    now[0] = 101.0
    third, _ = cache.get()
    assert third["BBCA"]["ownership_public_institutions_held_pct"] == 20.0
    assert len(calls) == 2


def test_cache_keeps_last_good_context_when_refresh_fails_and_retries_bounded() -> None:
    now = [0.0]
    calls = []

    def loader():
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return {
                "BBCA": {
                    "ownership_public_institutions_held_pct": 19.0,
                    "ownership_public_context_coverage_pct": 100.0,
                }
            }, {"state": "PUBLIC_OWNERSHIP_LOADED", "tickers": 1}
        if len(calls) == 2:
            return {}, {"state": "PUBLIC_OWNERSHIP_HTTP_503", "tickers": 0}
        return {
            "BBCA": {
                "ownership_public_institutions_held_pct": 20.0,
                "ownership_public_context_coverage_pct": 100.0,
            }
        }, {"state": "PUBLIC_OWNERSHIP_LOADED", "tickers": 1}

    cache = _OwnershipContextCache(loader, ttl_seconds=100.0, retry_seconds=10.0, clock=lambda: now[0])
    first, _ = cache.get()
    assert first["BBCA"]["ownership_public_institutions_held_pct"] == 19.0
    now[0] = 101.0
    stale, meta = cache.get()
    assert stale["BBCA"]["ownership_public_institutions_held_pct"] == 19.0
    assert meta["fallback_state"] == "LAST_KNOWN_GOOD_PUBLIC_OWNERSHIP_CONTEXT"
    now[0] = 109.0
    still_stale, _ = cache.get()
    assert still_stale["BBCA"]["ownership_public_institutions_held_pct"] == 19.0
    assert len(calls) == 2
    now[0] = 112.0
    refreshed, _ = cache.get()
    assert refreshed["BBCA"]["ownership_public_institutions_held_pct"] == 20.0
    assert len(calls) == 3
