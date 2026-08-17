from __future__ import annotations

"""Restore the dashboard's Execution Research Top 3 semantics.

The dashboard tab is explicitly research-oriented. It must show the same
research candidates as before, while the guarded and real-money selectors
remain available separately for decision/execution use.
"""


def install() -> dict[str, str]:
    import top3_dashboard
    from top3_lane_patch import select_research_top3

    # app.py imports select_top3 after runtime patches are installed. Point
    # that legacy dashboard selector at the research lane so the existing
    # "Execution Research Top 3" tab is populated again.
    top3_dashboard.select_top3 = select_research_top3
    return {
        "research_display_lane": "RAW_RESEARCH_TOP3",
        "guarded_lane": "GUARDED_DECISION_TOP3",
        "execution_lane": "EXECUTION_TOP3_MANUAL_OR_DIRECT",
    }


__all__ = ["install"]
