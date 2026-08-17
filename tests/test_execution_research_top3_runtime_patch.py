import pandas as pd

from execution_research_top3_runtime_patch import install


def test_install_routes_dashboard_top3_to_research_lane(monkeypatch):
    import top3_dashboard
    from top3_lane_patch import select_research_top3

    monkeypatch.setattr(top3_dashboard, "select_top3", object(), raising=False)
    result = install()

    assert result["research_display_lane"] == "RAW_RESEARCH_TOP3"
    assert top3_dashboard.select_top3 is select_research_top3


def test_research_display_keeps_three_candidates_even_when_execution_is_empty():
    from top3_lane_patch import select_research_top3, select_execution_top3

    frame = pd.DataFrame([
        {
            "ticker": "PKPK.JK",
            "raw_research_rank": 1,
            "raw_research_score": 90,
            "real_money_entry_candidate": False,
            "real_money_hard_block_count": 1,
            "real_money_authorization_tier": "HARD_BLOCKED",
            "real_money_gate_class": "HARD_BLOCK",
        },
        {
            "ticker": "SPTO.JK",
            "raw_research_rank": 2,
            "raw_research_score": 85,
            "real_money_entry_candidate": False,
            "real_money_hard_block_count": 1,
            "real_money_authorization_tier": "HARD_BLOCKED",
            "real_money_gate_class": "HARD_BLOCK",
        },
        {
            "ticker": "MARK.JK",
            "raw_research_rank": 3,
            "raw_research_score": 80,
            "real_money_entry_candidate": False,
            "real_money_hard_block_count": 0,
            "real_money_authorization_tier": "WAIT_TIMING",
            "real_money_gate_class": "WAIT_TIMING",
        },
    ])

    research = select_research_top3(frame, 3)
    execution = select_execution_top3(frame, 3)

    assert research["ticker"].tolist() == ["PKPK.JK", "SPTO.JK", "MARK.JK"]
    assert research["selection_lane"].eq("RAW_RESEARCH_TOP3").all()
    assert execution.empty
