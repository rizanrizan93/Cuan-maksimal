from __future__ import annotations

import pandas as pd

from top3_dashboard import (
    calculate_dashboard_scores,
    enrich_dashboard_scores,
    render_top3_dashboard_html,
    select_top3,
)


def sample_row(ticker: str, state: str, conviction: float, deep: str = "DEEP_REVIEWED") -> dict:
    return {
        "ticker": ticker,
        "company_name": f"Company {ticker}",
        "sector": "ENERGY",
        "emir_decision_state": state,
        "emir_conviction_score": conviction,
        "emir_evidence_coverage_pct": 75,
        "deep_review_state": deep,
        "narrative_score": 80,
        "broker_inventory_score": 75,
        "smart_money_score": 78,
        "absorption_score": 72,
        "close_acceptance20_pct": 70,
        "holder_persistence_score": 73,
        "inventory_dryness_score": 69,
        "inventory_dryness_multiyear_score": 71,
        "inventory_cycle_score": 72,
        "distribution_score": 20,
        "trend_score": 76,
        "markup_quality_score": 71,
        "relative_strength20_pct": 10,
        "market_structure_score": 79,
        "fundamental_conversion_score": 74,
        "liquidity_score": 82,
        "last_price": 1000,
        "entry_low": 970,
        "entry_high": 1000,
        "trigger": 1020,
        "stop_loss": 940,
        "tp1": 1080,
        "tp2": 1150,
        "rr_tp2": 3,
        "rr_tp2_at_entry_high": 1.5,
        "breakout_entry": 1020,
        "breakout_stop_loss": 980,
        "breakout_tp2": 1140,
        "breakout_rr_tp2": 3.0,
        "accumulation_days20": 5,
        "absorption_days20": 3,
        "distribution_days20": 1,
        "broker_inventory_evidence_type": "OHLCV_PROXY",
        "emir_lifecycle": "INVENTORY_COLLECTION",
        "sector_rrg_state": "IMPROVING",
        "execution_capacity_state": "EXECUTION_CAPACITY_OK",
        "why_now": "Narrative dan flow mulai konvergen.",
        "risk_flags": "DIRECT_BID_OFFER_TRIGGER_MISSING_PROXY_USED",
    }


def test_final_score_is_existing_emir_conviction_score():
    scores = calculate_dashboard_scores(sample_row("AAA.JK", "EMIR_AUTO_EOD_READY", 83.7))
    assert scores["emir_final_score"] == 83.7
    assert 0 <= scores["dashboard_flow_score"] <= 100
    assert 0 <= scores["dashboard_silent_accum_score"] <= 100


def test_top3_excludes_blocked_and_prefers_deep_review():
    rows = [
        sample_row("AAA.JK", "EMIR_AUTO_EOD_READY", 80),
        sample_row("BBB.JK", "EMIR_WATCH_INVENTORY_COLLECTION", 90),
        sample_row("CCC.JK", "EMIR_DATA_INTEGRITY_BLOCK", 99),
        sample_row("DDD.JK", "EMIR_NO_EDGE_YET", 95, deep="RADAR_ONLY"),
        sample_row("EEE.JK", "EMIR_WAIT_MONEY_FLOW", 78),
    ]
    enriched = enrich_dashboard_scores(pd.DataFrame(rows))
    top3 = select_top3(enriched)
    assert list(top3["ticker"]) == ["AAA.JK", "BBB.JK", "EEE.JK"]
    assert "CCC.JK" not in set(top3["ticker"])


def test_html_labels_proxy_and_does_not_fabricate_broker_codes():
    enriched = enrich_dashboard_scores(pd.DataFrame([sample_row("AAA.JK", "EMIR_AUTO_EOD_READY", 84)]))
    top3 = select_top3(enriched)
    html = render_top3_dashboard_html(top3, scan_id="scan-1", as_of="2026-08-06", market_regime="SELECTIVE")
    assert "OHLCV PROXY — BUKAN IDENTITAS BROKER" in html
    assert "TOP 3" in html
    assert "FINAL SCORE" in html
    assert "XL" not in html and "YU" not in html


def test_html_escapes_untrusted_text():
    row = sample_row("AAA.JK", "EMIR_NO_EDGE_YET", 50)
    row["company_name"] = "<script>alert(1)</script>"
    top3 = select_top3(enrich_dashboard_scores(pd.DataFrame([row])))
    html = render_top3_dashboard_html(top3)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_watch_inventory_state_outranks_no_edge_even_with_lower_score():
    rows = [
        sample_row("WATCH.JK", "EMIR_WATCH_INVENTORY_COLLECTION", 62),
        sample_row("NOEDGE.JK", "EMIR_NO_EDGE_YET", 90),
    ]
    top = select_top3(enrich_dashboard_scores(pd.DataFrame(rows)), limit=2)
    assert list(top["ticker"]) == ["WATCH.JK", "NOEDGE.JK"]


def test_dashboard_displays_separate_accumulation_and_breakout_rr():
    top = select_top3(enrich_dashboard_scores(pd.DataFrame([sample_row("AAA.JK", "EMIR_AUTO_EOD_READY", 84)])))
    html = render_top3_dashboard_html(top)
    assert "ACCUM RR @ high" in html
    assert "BREAKOUT RR" in html
    assert "Risk : Reward" not in html
