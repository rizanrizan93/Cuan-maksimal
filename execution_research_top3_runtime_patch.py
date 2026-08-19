from __future__ import annotations

"""Deterministic runtime routing for Emir's Execution Research Top 3 tab.

The tab is research-only. It must remain populated from the raw research lane
without depending on the guarded/real-money selector or on a recomputed three-rank
contract. Real-money execution selection remains in its dedicated selector.
"""

import numpy as np
import pandas as pd

PATCH_VERSION = "2.0.0-deterministic-research-top3"


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def select_research_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Return raw research priorities and never use execution gates as a filter."""
    if not isinstance(radar, pd.DataFrame) or radar.empty:
        return radar.copy() if isinstance(radar, pd.DataFrame) else pd.DataFrame()

    local = radar.copy()
    raw_rank = _num(local, "raw_research_rank")
    raw_score = _num(local, "raw_research_score")
    evidence_coverage = _num(local, "emir_evidence_coverage_pct", 0.0).fillna(0.0)

    if raw_rank.notna().any():
        local["_research_rank"] = raw_rank
        local["_research_score"] = raw_score
        local = local.sort_values(
            ["_research_rank", "_research_score"],
            ascending=[True, False],
            na_position="last",
            kind="stable",
        )
    else:
        fallback_score = raw_score
        if not fallback_score.notna().any():
            fallback_score = _num(local, "emir_final_score")
        local["_research_rank"] = fallback_score.rank(method="first", ascending=False, na_option="bottom")
        local["_research_score"] = fallback_score
        local["_research_evidence_coverage"] = evidence_coverage
        local = local.sort_values(
            ["_research_score", "_research_evidence_coverage"],
            ascending=[False, False],
            na_position="last",
            kind="stable",
        )

    local = (
        local.head(max(0, int(limit)))
        .drop(columns=["_research_rank", "_research_score", "_research_evidence_coverage"], errors="ignore")
        .reset_index(drop=True)
    )
    if local.empty:
        return local

    local.insert(0, "selection_rank", range(1, len(local) + 1))
    local.insert(1, "selection_lane", "RAW_RESEARCH_TOP3")
    local.insert(
        2,
        "selection_lane_note",
        "Research priority only; hard blockers may appear. This table is not capital authorization.",
    )
    local["execution_research_top3_patch_version"] = PATCH_VERSION
    return local


def install() -> dict[str, str]:
    import top3_dashboard
    import top3_dashboard_legacy

    # Patch both module namespaces used by the app and legacy dashboard. Do not
    # depend on top3_lane_patch's rank-contract reconstruction for this research
    # display path; the research tab must remain populated from persisted radar.
    for owner in (top3_dashboard, top3_dashboard_legacy):
        setattr(owner, "select_research_top3", select_research_top3)
        setattr(owner, "select_top3", select_research_top3)

    return {
        "patch_version": PATCH_VERSION,
        "research_display_lane": "RAW_RESEARCH_TOP3",
        "guarded_lane": "GUARDED_DECISION_TOP3",
        "execution_lane": "EXECUTION_TOP3_MANUAL_OR_DIRECT",
        "policy": "RESEARCH_DISPLAY_CANNOT_BE_EMPTIED_BY_REAL_MONEY_GATE",
    }


__all__ = ["PATCH_VERSION", "select_research_top3", "install"]
