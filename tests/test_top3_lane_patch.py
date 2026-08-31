import pandas as pd
import pytest

from top3_lane_patch import (
    _wrap_renderer,
    select_execution_top3,
    select_guarded_top3,
    select_production_watch_top3,
    select_research_top3,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ticker": "PKPK.JK",
            "emir_conviction_score": 90.0,
            "next_leader_score": 80.0,
            "real_money_candidate_score": 92.0,
            "real_money_candidate": False,
            "real_money_entry_candidate": False,
            "real_money_ready": False,
            "real_money_hard_block_count": 1,
            "real_money_authorization_tier": "HARD_BLOCKED",
            "real_money_gate_class": "HARD_BLOCK",
            "emir_decision_state": "EMIR_DATA_INTEGRITY_BLOCK",
            "emir_evidence_coverage_pct": 95.0,
            "real_money_rr_score": 70.0,
        },
        {
            "ticker": "SPTO.JK",
            "emir_conviction_score": 85.0,
            "next_leader_score": 75.0,
            "real_money_candidate_score": 88.0,
            "real_money_candidate": False,
            "real_money_entry_candidate": False,
            "real_money_ready": False,
            "real_money_hard_block_count": 1,
            "real_money_authorization_tier": "HARD_BLOCKED",
            "real_money_gate_class": "HARD_BLOCK",
            "emir_decision_state": "EMIR_REJECT_IDX_INTEGRITY",
            "emir_evidence_coverage_pct": 90.0,
            "real_money_rr_score": 65.0,
        },
        {
            "ticker": "MARK.JK",
            "emir_conviction_score": 75.0,
            "next_leader_score": 90.0,
            "real_money_candidate_score": 80.0,
            "real_money_candidate": True,
            "real_money_entry_candidate": False,
            "real_money_ready": False,
            "real_money_hard_block_count": 0,
            "real_money_authorization_tier": "WAIT_TIMING",
            "real_money_gate_class": "WAIT_TIMING",
            "emir_decision_state": "EMIR_WAIT_REACCUMULATION",
            "emir_evidence_coverage_pct": 92.0,
            "real_money_rr_score": 75.0,
        },
        {
            "ticker": "DMAS.JK",
            "emir_conviction_score": 72.0,
            "next_leader_score": 88.0,
            "real_money_candidate_score": 78.0,
            "real_money_candidate": True,
            "real_money_entry_candidate": False,
            "real_money_ready": False,
            "real_money_hard_block_count": 0,
            "real_money_authorization_tier": "WAIT_TIMING",
            "real_money_gate_class": "WAIT_TIMING",
            "emir_decision_state": "EMIR_WAIT_NARRATIVE",
            "emir_evidence_coverage_pct": 91.0,
            "real_money_rr_score": 72.0,
        },
        {
            "ticker": "POWR.JK",
            "emir_conviction_score": 70.0,
            "next_leader_score": 86.0,
            "real_money_candidate_score": 76.0,
            "real_money_candidate": True,
            "real_money_entry_candidate": False,
            "real_money_ready": False,
            "real_money_hard_block_count": 0,
            "real_money_authorization_tier": "WAIT_TIMING",
            "real_money_gate_class": "WAIT_TIMING",
            "emir_decision_state": "EMIR_WAIT_MONEY_FLOW",
            "emir_evidence_coverage_pct": 90.0,
            "real_money_rr_score": 70.0,
        },
        {
            "ticker": "OMED.JK",
            "emir_conviction_score": 74.0,
            "next_leader_score": 82.0,
            "real_money_candidate_score": 79.0,
            "real_money_candidate": True,
            "real_money_entry_candidate": True,
            "real_money_ready": True,
            "real_money_hard_block_count": 0,
            "real_money_authorization_tier": "DIRECT_VERIFIED_READY",
            "real_money_gate_class": "DIRECT_VERIFIED_READY",
            "real_money_gate_state": "REAL_MONEY_DIRECT_VERIFIED_READY",
            "entry_authorization_state": "SCANNER_AUTHORIZED_DIRECT_VERIFIED",
            "real_money_block_reasons": "NONE",
            "emir_decision_state": "EMIR_READY_WITH_PRECISE_TRIGGER",
            "execution_geometry_valid": True,
            "execution_entry_reference": 100.0,
            "execution_stop_loss": 90.0,
            "execution_tp1": 116.0,
            "execution_tp2": 125.0,
            "preferred_execution_path": "ACCUMULATION_PULLBACK",
            "execution_rr_tp1": 1.6,
            "execution_min_rr_pass": True,
            "emir_evidence_coverage_pct": 91.0,
            "real_money_rr_score": 62.0,
        },
    ])


def test_raw_research_lane_can_contain_hard_blocked_but_labels_it_research_only():
    out = select_research_top3(_frame(), 3)
    assert out["ticker"].tolist()[:2] == ["PKPK.JK", "SPTO.JK"]
    assert set(out["selection_lane"]) == {"RAW_RESEARCH_TOP3"}
    assert "Research priority only" in out.iloc[0]["selection_lane_note"]


def test_guarded_lane_excludes_genuine_hard_blockers_and_backfills_from_full_radar():
    out = select_guarded_top3(_frame(), 3)
    assert out["ticker"].tolist() == ["MARK.JK", "DMAS.JK", "POWR.JK"]
    assert "PKPK.JK" not in out["ticker"].tolist()
    assert "SPTO.JK" not in out["ticker"].tolist()


def test_production_watch_separates_wait_timing_from_execution():
    out = select_production_watch_top3(_frame(), 3)
    assert out["ticker"].tolist() == ["MARK.JK", "DMAS.JK", "POWR.JK"]
    assert "OMED.JK" not in out["ticker"].tolist()


def test_execution_lane_returns_only_omed_and_never_backfills_hard_block_or_wait():
    out = select_execution_top3(_frame(), 3)
    assert out["ticker"].tolist() == ["OMED.JK"]
    assert set(out["selection_lane"]) == {"EXECUTION_TOP3_AUTHORIZED_DIRECT"}


def test_execution_lane_can_return_less_than_three():
    out = select_execution_top3(_frame(), 3)
    assert len(out) == 1


def test_lane_banner_is_injected_inside_existing_dashboard_wrap():
    source = pd.DataFrame([
        {
            "ticker": ticker,
            "selection_lane": "RAW_RESEARCH_TOP3",
            "selection_lane_note": "Research priority only.",
        }
        for ticker in ("BBRI.JK", "ADRO.JK", "BBCA.JK")
    ])
    frozen = source.copy(deep=True)
    original = (
        '<style>.es-wrap{color:white}</style>'
        '<div class="es-wrap"><h1>TOP 3 EMIR-STYLE SCANNER</h1>'
        '<section class="es-card rank1">BBRI</section>'
        '<section class="es-card rank2">ADRO</section>'
        '<section class="es-card rank3">BBCA</section></div>'
    )
    owner = type("Dashboard", (), {"render_top3_dashboard_html": staticmethod(lambda frame: original)})

    _wrap_renderer(owner)
    _wrap_renderer(owner)
    html = owner.render_top3_dashboard_html(source)

    assert html.count("RAW_RESEARCH_TOP3") == 1
    assert html.count('class="es-lane-banner"') == 1
    assert html.count('class="es-card ') == 3
    assert all(ticker in html for ticker in ("BBRI", "ADRO", "BBCA"))
    assert html.index("<style>") < html.index('<div class="es-wrap">')
    assert html.index('<div class="es-wrap">') < html.index('class="es-lane-banner"')
    assert html.index('class="es-lane-banner"') < html.index('class="es-card rank1"')
    pd.testing.assert_frame_equal(source, frozen)


def test_lane_banner_fails_soft_when_renderer_contract_is_absent():
    source = pd.DataFrame([{
        "ticker": "BBRI.JK",
        "selection_lane": "RAW_RESEARCH_TOP3",
        "selection_lane_note": "Research priority only.",
    }])
    original = '<section class="custom-dashboard">BBRI</section>'
    owner = type("Dashboard", (), {"render_top3_dashboard_html": staticmethod(lambda frame: original)})

    _wrap_renderer(owner)

    assert owner.render_top3_dashboard_html(source) == original


@pytest.mark.parametrize("result", [None, 17])
def test_lane_banner_returns_non_string_renderer_output_unchanged(result):
    source = pd.DataFrame([{"ticker": "BBRI.JK", "selection_lane": "RAW_RESEARCH_TOP3"}])
    owner = type(
        "Dashboard",
        (),
        {"render_top3_dashboard_html": staticmethod(lambda frame: result)},
    )

    _wrap_renderer(owner)

    assert owner.render_top3_dashboard_html(source) is result


def test_lane_banner_preserves_arbitrary_renderer_object_identity():
    source = pd.DataFrame([{"ticker": "BBRI.JK", "selection_lane": "RAW_RESEARCH_TOP3"}])
    result = object()
    owner = type(
        "Dashboard",
        (),
        {"render_top3_dashboard_html": staticmethod(lambda frame: result)},
    )

    _wrap_renderer(owner)

    assert owner.render_top3_dashboard_html(source) is result


def test_lane_banner_does_not_swallow_original_renderer_exception():
    source = pd.DataFrame([{"ticker": "BBRI.JK", "selection_lane": "RAW_RESEARCH_TOP3"}])

    def broken_renderer(frame):
        raise RuntimeError("renderer failed")

    owner = type("Dashboard", (), {"render_top3_dashboard_html": staticmethod(broken_renderer)})
    _wrap_renderer(owner)

    with pytest.raises(RuntimeError, match="renderer failed"):
        owner.render_top3_dashboard_html(source)


def test_lane_banner_empty_top3_behavior_is_unchanged():
    source = pd.DataFrame(columns=["ticker", "selection_lane"])
    original = '<div class="es-wrap">empty renderer output</div>'
    owner = type("Dashboard", (), {"render_top3_dashboard_html": staticmethod(lambda frame: original)})

    _wrap_renderer(owner)

    assert owner.render_top3_dashboard_html(source) == original


def test_lane_banner_does_not_mutate_valid_frozen_snapshot():
    import final_decision as decision

    source = pd.DataFrame([
        {
            "ticker": "BBRI.JK",
            "selection_lane": "RAW_RESEARCH_TOP3",
            "raw_research_rank": 1,
            "guarded_decision_priority_rank": 1,
            "production_real_money_rank": float("nan"),
            "decision_snapshot_state": decision.FINAL_DECISION_STATE,
            "decision_snapshot_version": decision.FINAL_DECISION_VERSION,
        }
    ])
    source["decision_snapshot_fingerprint"] = decision._fingerprint(source)
    assert decision.is_final_decision_snapshot(source) is True
    frozen = source.copy(deep=True)
    original = '<style></style><div class="es-wrap"><section class="es-card rank1">BBRI</section></div>'
    owner = type("Dashboard", (), {"render_top3_dashboard_html": staticmethod(lambda frame: original)})

    _wrap_renderer(owner)
    owner.render_top3_dashboard_html(source)

    pd.testing.assert_frame_equal(source, frozen)
    assert decision.is_final_decision_snapshot(source) is True
