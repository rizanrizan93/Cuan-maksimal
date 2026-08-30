from __future__ import annotations

import pandas as pd
import pytest

from evidence_governance import apply_three_rank_contract
from future_fundamental import calculate_future_fundamental


def _context() -> dict[str, object]:
    return {
        "narrative": {"top_down_catalyst_score": 70.0, "industry_translation_score": 70.0, "narrative_coverage_pct": 90.0},
        "fundamental": {
            "fundamental_coverage_pct": 90.0,
            "fundamental_cashflow_quality_state": "CASHFLOW_POSITIVE_CONVERTING",
            "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK",
            "fundamental_conversion_score": 75.0,
            "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED",
            "fundamental_period_freshness_state": "CURRENT",
            "fundamental_observed_at": "2026-01-01T00:00:00Z",
        },
        "ownership": {},
        "sector": {"sector_leadership_score": 65.0, "sector_context_coverage_pct": 90.0},
    }


def _event(**overrides) -> dict[str, object]:
    event = {
        "ticker": "TEST.JK",
        "published_at": "2026-06-19T00:00:00Z",
        "title": "TEST capacity expansion and new customer contract",
        "summary": "Production capacity expansion supports a new customer contract.",
        "category": "CAPACITY_AND_BACKLOG",
        "materiality_score": 95.0,
        "financial_bridge_score": 95.0,
        "url": "https://example.com/research",
        "source_verified": False,
        "source_tier": "PUBLIC_RESEARCH",
        "forward_research_only": True,
    }
    event.update(overrides)
    return event


@pytest.mark.parametrize(
    ("events", "provenance", "authorized"),
    [
        ([_event()], "PUBLIC_RESEARCH", False),
        ([_event(source_verified=True, source_tier="ISSUER", forward_research_only=False)], "DIRECT_OR_OFFICIAL", True),
        ([_event(), _event(title="Issuer confirms capacity project", source_verified=True, source_tier="OFFICIAL", forward_research_only=False, url="https://issuer.example/disclosure")], "DIRECT_OR_OFFICIAL", True),
        ([_event(source_tier="", forward_research_only=False, url="")], "INFERRED", False),
        ([], "MISSING", False),
    ],
)
def test_forward_provenance_and_authorization_boundary(events, provenance, authorized) -> None:
    out = calculate_future_fundamental(
        ticker="TEST.JK",
        events=pd.DataFrame(events),
        as_of="2026-08-14T00:00:00Z",
        **_context(),
    )
    assert out["future_forward_provenance_state"] == provenance
    assert bool(out["future_direct_forward_authorization_eligible"]) is authorized
    if provenance == "PUBLIC_RESEARCH":
        assert out["future_fundamental_score"] > 0
        assert out["future_public_research_forward_event_count"] >= 1
        assert out["future_official_forward_event_count"] == 0
        normalized = apply_three_rank_contract(pd.DataFrame([{
            "ticker": "TEST.JK",
            "emir_decision_state": "EMIR_WAIT_NARRATIVE",
            "emir_conviction_score": out["future_fundamental_score"],
            **out,
        }]))
        assert normalized.iloc[0]["future_forward_provenance_state"] == "PUBLIC_RESEARCH"
        assert bool(normalized.iloc[0]["future_direct_forward_authorization_eligible"]) is False


def test_verified_nonofficial_provenance_does_not_become_direct_official() -> None:
    out = calculate_future_fundamental(
        ticker="TEST.JK",
        events=pd.DataFrame([_event(source_verified=True, source_tier="VERIFIED_MEDIA", forward_research_only=False)]),
        as_of="2026-08-14T00:00:00Z",
        **_context(),
    )
    assert out["future_forward_provenance_state"] == "VERIFIED"
    assert out["future_direct_forward_authorization_eligible"] is False
