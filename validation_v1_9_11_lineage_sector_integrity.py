from __future__ import annotations

import pandas as pd

from autonomous_enrichment import reconcile_fundamental_snapshot
from top3_dashboard import calculate_next_leader_score, _reason_lines


def main() -> None:
    proxy = {
        "ticker":"BISI.JK", "fundamental_latest_period":"2026-06-30",
        "fundamental_income_period":"2026-06-30", "fundamental_conversion_score":70,
        "fundamental_coverage_pct":80, "fundamental_data_quality_score":75,
        "revenue_growth_yoy_pct":81, "earnings_growth_yoy_pct":4494,
        "fundamental_growth_quality_state":"EARNINGS_BASE_EFFECT_EXTREME | EARNINGS_SMALL_BASE",
    }
    official = {
        "ticker":"BISI.JK", "idx_official_source_verified":True,
        "idx_official_period_end":"2026-03-31", "idx_official_source_url":"https://idx.id/TW1/BISI/instance.zip",
        "idx_official_coverage_pct":90, "idx_official_revenue_growth_yoy_pct":22,
        "idx_official_earnings_growth_yoy_pct":58,
    }
    rec = reconcile_fundamental_snapshot(proxy, official, now=pd.Timestamp("2026-08-10"))
    assert rec["fundamental_latest_period"] == "2026-06-30"
    assert rec["fundamental_cross_source_state"] == "PROXY_NEWER_THAN_OFFICIAL"
    assert rec["fundamental_official_source_coverage_pct"] == 0.0
    assert rec["fundamental_official_crosscheck_period"] == "2026-03-31"
    assert rec["fundamental_official_source_url"] == ""

    bisi = {**proxy, "sector":"Consumer Non-Cyclicals", "story_runway_score":70, "financial_conversion_score":70,
            "issuer_alignment_score":70, "sector_leadership_score":70, "smart_money_score":70, "broker_inventory_score":70,
            "market_structure_score":70, "liquidity_score":70, "ownership_score":70, "distribution_score":20}
    score = calculate_next_leader_score(bisi)
    assert score["next_leader_business_momentum_score"] <= 68.0
    assert "SMALL_EARNINGS_BASE_NORMALIZED" in score["next_leader_quality_flags"]

    dmas = {**bisi, "ticker":"DMAS.JK", "sector":"Property & Real Estate", "fundamental_growth_quality_state":"NORMAL_GROWTH_BASE",
            "revenue_growth_yoy_pct":194, "earnings_growth_yoy_pct":175}
    prop = calculate_next_leader_score(dmas)
    assert prop["next_leader_business_momentum_score"] <= 90.0
    assert prop["next_leader_sector_model_state"] == "PROPERTY_LUMPY_RECOGNITION_NORMALIZED"

    reasons = _reason_lines({"narrative_score":55,"dashboard_flow_score":70,"dashboard_silent_accum_score":70,
                             "market_structure_score":70,"dashboard_momentum_score":40,"fundamental_conversion_score":50})
    assert reasons[-1][0] == "warning" and reasons[-2][0] == "warning"
    assert any(state == "developing" for state, _ in reasons)
    print("PASS v1.9.11 lineage/sector integrity")


if __name__ == "__main__":
    main()
