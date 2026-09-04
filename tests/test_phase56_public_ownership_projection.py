from __future__ import annotations

from phase56_public_ownership_projection import _rows_to_context, merge_public_context


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
