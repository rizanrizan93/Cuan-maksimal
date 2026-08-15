from __future__ import annotations

import pandas as pd

import future_fundamental_ui_patch as ui


def _row(ticker: str, **extra):
    row = {
        "ticker": ticker,
        "last_price": 100.0,
        "emir_final_score": 60.0,
        "emir_decision_state": "EMIR_NO_EDGE_YET",
        "future_fundamental_score": float("nan"),
        "future_fundamental_coverage_pct": 0.0,
    }
    row.update(extra)
    return row


def test_missing_future_score_shows_collection_state():
    import top3_dashboard

    ui.install()
    top = pd.DataFrame([
        _row("AAA.JK", dashboard_rank=1, forward_collection_state="FORWARD_CHECK_COMPLETED_NO_MATERIAL_EVENT"),
        _row("BBB.JK", dashboard_rank=2, forward_collection_state="MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND"),
        _row("CCC.JK", dashboard_rank=3, future_fundamental_score=80.0, future_fundamental_coverage_pct=90.0),
    ])
    html = top3_dashboard.render_top3_dashboard_html(top)
    assert ">CHECKED</b>" in html
    assert ">RESEARCH</b>" in html
    assert html.count(">PENDING</b>") == 0


def test_evidence_pending_is_explicit_not_blank():
    import top3_dashboard

    ui.install()
    top = pd.DataFrame([_row("AAA.JK", dashboard_rank=1, future_fundamental_state="FUTURE_FUNDAMENTAL_EVIDENCE_PENDING")])
    html = top3_dashboard.render_top3_dashboard_html(top)
    assert ">PENDING</b>" in html
    assert "does not invent a neutral score" in html
