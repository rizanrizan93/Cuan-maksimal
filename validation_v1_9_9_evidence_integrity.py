from __future__ import annotations

import pandas as pd

from autonomous_enrichment import recalibrate_cached_fundamental_snapshot
from idx_official_fundamentals import idx_instance_urls
from narrative_flow_engine import score_narrative_events
from top3_dashboard import calculate_next_leader_score


def check_database_first_field_reconciliation() -> None:
    payload = {
        "ticker": "TEST.JK",
        "revenue_ttm": 1_000.0,
        "net_income_ttm": 100.0,
        "operating_cash_flow_ttm": 120.0,
        "free_cash_flow_proxy_ttm": 80.0,
        "equity_latest": 500.0,
        "debt_latest": 100.0,
        "cash_latest": 70.0,
        "total_assets_latest": 900.0,
        "total_liabilities_latest": 400.0,
        "current_assets_latest": 250.0,
        "current_liabilities_latest": 125.0,
        "revenue_growth_pct": 20.0,
        "earnings_growth_pct": 25.0,
        "revenue_growth_yoy_pct": 20.0,
        "earnings_growth_yoy_pct": 25.0,
        "net_margin_ttm_pct": 10.0,
        "roe_ttm_pct": 20.0,
        "roa_ttm_pct": 11.1,
        "growth_basis_state": "YOY_PRIMARY",
        "fundamental_period_alignment_state": "ALIGNED",
        "fundamental_period_freshness_state": "CURRENT_QUARTERLY_PERIOD",
        "fundamental_growth_consistency_score": 90.0,
        "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED",
        "fundamental_official_source_coverage_pct": 0.0,
        "fundamental_cache_schema_version": "4",
    }
    out = recalibrate_cached_fundamental_snapshot(payload)
    assert out["fundamental_cashflow_state"] == "OCF_FCF_TTM_AVAILABLE"
    assert out["ocf_conversion_ratio"] == 1.2
    assert out["fundamental_coverage_pct"] >= 85.0
    assert out["fundamental_database_enrichment_state"] == "DATABASE_FIRST_FIELD_LEVEL_RECONCILIATION"
    assert out["fundamental_official_source_coverage_pct"] == 0.0


def check_business_momentum_survives_dashboard_projection() -> None:
    row = {
        "fundamental_conversion_score": 72.0,
        "fundamental_coverage_pct": 90.0,
        "fundamental_data_quality_score": 86.0,
        "fundamental_cashflow_state": "OCF_FCF_TTM_AVAILABLE",
        "fundamental_state": "FUTURE_FUNDAMENTAL_SUPPORTIVE",
        "fundamental_period_freshness_state": "CURRENT_QUARTERLY_PERIOD",
        "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED",
        "fundamental_ytd_quarters_count": 2,
        "revenue_growth_ytd_yoy_pct": 24.0,
        "earnings_growth_ytd_yoy_pct": 31.0,
        "story_runway_score": 70.0,
        "financial_conversion_score": 68.0,
        "issuer_alignment_score": 65.0,
        "sector_leadership_score": 65.0,
        "smart_money_score": 55.0,
        "broker_inventory_score": 50.0,
        "market_structure_score": 60.0,
        "liquidity_score": 70.0,
        "ownership_score": 50.0,
        "distribution_score": 20.0,
    }
    result = calculate_next_leader_score(row)
    assert result["next_leader_business_momentum_score"] > 70.0
    assert result["next_leader_business_momentum_basis"] == "YTD_CONFIRMED_BUSINESS_MOMENTUM"
    assert result["next_leader_eligible"] is True


def check_current_and_legacy_idx_sources() -> None:
    urls = idx_instance_urls("TEST.JK", 2026, "TW1")
    assert urls[0].startswith("https://www.idx.id/")
    assert any(url.startswith("https://www.idx.co.id/") for url in urls)

    events = pd.DataFrame([{
        "ticker": "TEST.JK",
        "published_at": "2026-08-09T00:00:00Z",
        "title": "TEST mengumumkan ekspansi kapasitas dan pertumbuhan pendapatan",
        "publisher": "IDX",
        "url": "https://www.idx.id/id/perusahaan-tercatat/keterbukaan-informasi",
        "source_verified": False,
        "source_tier": "PUBLIC",
    }])
    scored = score_narrative_events(events, as_of="2026-08-10T00:00:00Z")
    assert scored["narrative_official_source_count"] == 1
    assert scored["narrative_verified_source_count"] == 1
    assert scored["narrative_source_provenance_state"] == "VERIFIED_OFFICIAL_SOURCE"

    spoofed = events.copy()
    spoofed.loc[0, "url"] = "https://example.com/article?source=idx.id"
    spoofed_score = score_narrative_events(spoofed, as_of="2026-08-10T00:00:00Z")
    assert spoofed_score["narrative_official_source_count"] == 0
    assert spoofed_score["narrative_verified_source_count"] == 0


if __name__ == "__main__":
    checks = [
        check_database_first_field_reconciliation,
        check_business_momentum_survives_dashboard_projection,
        check_current_and_legacy_idx_sources,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("VALIDATION_V1_9_9_EVIDENCE_INTEGRITY=PASS")
