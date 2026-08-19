from __future__ import annotations

import pandas as pd

import execution_research_top3_runtime_patch as patch


def test_research_top3_uses_raw_research_order_even_with_hard_blocks():
    frame = pd.DataFrame([
        {
            "ticker": "PKPK.JK",
            "raw_research_rank": 1,
            "raw_research_score": 90.0,
            "emir_final_score": 70.0,
            "real_money_gate_class": "HARD_BLOCK",
        },
        {
            "ticker": "SPTO.JK",
            "raw_research_rank": 2,
            "raw_research_score": 85.0,
            "emir_final_score": 68.0,
            "real_money_gate_class": "HARD_BLOCK",
        },
        {
            "ticker": "MARK.JK",
            "raw_research_rank": 3,
            "raw_research_score": 80.0,
            "emir_final_score": 65.0,
            "real_money_gate_class": "WAIT_TIMING",
        },
    ])
    out = patch.select_research_top3(frame, 3)
    assert out["ticker"].tolist() == ["PKPK.JK", "SPTO.JK", "MARK.JK"]
    assert out["selection_lane"].eq("RAW_RESEARCH_TOP3").all()


def test_research_top3_falls_back_to_raw_score_when_rank_is_missing():
    frame = pd.DataFrame([
        {"ticker": "AAA.JK", "raw_research_score": 60.0, "emir_final_score": 65.0},
        {"ticker": "BBB.JK", "raw_research_score": 80.0, "emir_final_score": 55.0},
        {"ticker": "CCC.JK", "raw_research_score": 70.0, "emir_final_score": 75.0},
    ])
    out = patch.select_research_top3(frame, 3)
    assert out["ticker"].tolist() == ["BBB.JK", "CCC.JK", "AAA.JK"]


def test_research_top3_falls_back_when_rank_column_exists_but_is_all_na():
    frame = pd.DataFrame([
        {"ticker": "AAA.JK", "raw_research_rank": pd.NA, "raw_research_score": pd.NA, "emir_final_score": 62.0, "emir_evidence_coverage_pct": 80.0},
        {"ticker": "BBB.JK", "raw_research_rank": pd.NA, "raw_research_score": pd.NA, "emir_final_score": 84.0, "emir_evidence_coverage_pct": 50.0},
        {"ticker": "CCC.JK", "raw_research_rank": pd.NA, "raw_research_score": pd.NA, "emir_final_score": 74.0, "emir_evidence_coverage_pct": 90.0},
    ])
    out = patch.select_research_top3(frame, 3)
    assert out["ticker"].tolist() == ["BBB.JK", "CCC.JK", "AAA.JK"]
    assert out["selection_lane"].eq("RAW_RESEARCH_TOP3").all()
    assert out["selection_rank"].tolist() == [1, 2, 3]


def test_install_routes_both_dashboard_namespaces(monkeypatch):
    import top3_dashboard
    import top3_dashboard_legacy

    monkeypatch.setattr(top3_dashboard, "select_top3", object(), raising=False)
    monkeypatch.setattr(top3_dashboard_legacy, "select_top3", object(), raising=False)
    result = patch.install()

    assert result["research_display_lane"] == "RAW_RESEARCH_TOP3"
    assert top3_dashboard.select_top3 is patch.select_research_top3
    assert top3_dashboard_legacy.select_top3 is patch.select_research_top3
