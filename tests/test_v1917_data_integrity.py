import pandas as pd

from data_providers import assess_benchmark_freshness
from future_fundamental import calculate_future_fundamental


def _future_case(level: str):
    if level == "high":
        narrative = {
            "top_down_catalyst_score": 92.0,
            "industry_translation_score": 88.0,
            "narrative_coverage_pct": 100.0,
            "issuer_alignment_score": 90.0,
            "issuer_alignment_coverage_pct": 100.0,
        }
        ownership = {"ownership_score": 88.0, "ownership_coverage_pct": 100.0}
        sector = {"sector_leadership_score": 90.0, "sector_context_coverage_pct": 100.0}
        fundamental = {
            "fundamental_coverage_pct": 100.0,
            "fundamental_cashflow_quality_state": "CASHFLOW_POSITIVE_CONVERTING",
            "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK",
            "current_ratio": 1.8,
            "cash_to_debt_ratio": 1.2,
            "fundamental_conversion_score": 90.0,
            "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED",
            "fundamental_period_freshness_state": "CURRENT",
        }
    else:
        narrative = {
            "top_down_catalyst_score": 52.0,
            "industry_translation_score": 48.0,
            "narrative_coverage_pct": 100.0,
            "issuer_alignment_score": 50.0,
            "issuer_alignment_coverage_pct": 100.0,
        }
        ownership = {"ownership_score": 48.0, "ownership_coverage_pct": 100.0}
        sector = {"sector_leadership_score": 50.0, "sector_context_coverage_pct": 100.0}
        fundamental = {
            "fundamental_coverage_pct": 100.0,
            "fundamental_cashflow_quality_state": "OCF_NEGATIVE",
            "fundamental_leverage_risk_state": "HIGH_LEVERAGE",
            "current_ratio": 0.9,
            "cash_to_debt_ratio": 0.3,
            "fundamental_conversion_score": 50.0,
            "fundamental_growth_consistency_state": "QUARTER_AND_YTD_WEAK",
            "fundamental_period_freshness_state": "CURRENT",
        }
    return calculate_future_fundamental(
        ticker=f"{level.upper()}.JK",
        events=pd.DataFrame(),
        narrative=narrative,
        fundamental=fundamental,
        ownership=ownership,
        sector=sector,
        as_of="2026-08-14T00:00:00Z",
    )


def test_missing_forward_evidence_penalty_preserves_cross_sectional_discrimination():
    high = _future_case("high")
    low = _future_case("low")

    assert "NO_FORWARD_PROJECT_OR_CONTRACT_EVIDENCE" in high["future_fundamental_risk_flags"]
    assert "NO_FORWARD_PROJECT_OR_CONTRACT_EVIDENCE" in low["future_fundamental_risk_flags"]
    assert high["future_fundamental_score"] > low["future_fundamental_score"]
    assert high["future_fundamental_score"] != 55.0
    assert low["future_fundamental_score"] != 55.0
    assert high["future_fundamental_version"] == "1.0.2-forward-event-integrity"


def test_benchmark_freshness_accepts_index_only_fast_cache_references():
    session = pd.Timestamp("2026-08-13")
    benchmark = pd.DataFrame(
        {"Open": [8000.0], "High": [8050.0], "Low": [7950.0], "Close": [8025.0], "Volume": [1_000_000.0]},
        index=[session],
    )
    references = {
        f"T{i:03d}.JK": pd.DataFrame(index=[session])
        for i in range(20)
    }

    result = assess_benchmark_freshness(benchmark, references, min_universe_count=20)

    assert result["benchmark_freshness_state"] == "CURRENT_RELATIVE_TO_UNIVERSE"
    assert result["benchmark_usable"] is True
    assert result["universe_reference_count"] == 20
    assert result["universe_reference_date"] == "2026-08-13"
