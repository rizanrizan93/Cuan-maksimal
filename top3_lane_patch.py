from __future__ import annotations

"""Explicit Top-3 lane contract for Emir.

The legacy "Execution Research Top 3" selector mixed research priority with
execution semantics. This module preserves raw research, introduces a guarded
decision lane that excludes genuine hard blockers, and hardens the real-money
selector so it never backfills WAIT/HARD_BLOCK rows just to reach three names.
"""

from functools import wraps
from html import escape
from typing import Any

import numpy as np
import pandas as pd

from evidence_governance import (
    apply_three_rank_contract,
    production_authorization_mask,
    production_hard_block_mask,
)

TOP3_LANE_CONTRACT_VERSION = "2.0.0-emir-fail-closed-production-lane"


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().upper() in {
        "1", "TRUE", "YES", "Y", "ON", "READY", "VERIFIED",
    }


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _ensure_rank_contract(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    frozen = (
        "decision_snapshot_state" in frame.columns
        and frame["decision_snapshot_state"].eq("FINAL_DECISION_FROZEN").all()
        and {
            "raw_research_rank",
            "guarded_decision_priority_rank",
            "production_real_money_rank",
        }.issubset(frame.columns)
    )
    if frozen:
        return frame.copy(deep=True)
    # Recompute even when persisted rank columns exist. A stale broad production
    # rank must never bypass the current fail-closed authorization contract.
    return apply_three_rank_contract(frame)


def _hard_block_mask(frame: pd.DataFrame) -> pd.Series:
    return production_hard_block_mask(frame)


def _stamp_lane(
    frame: pd.DataFrame,
    *,
    lane: str,
    note: str,
    rank_column: str,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    if out.empty:
        return out
    out.insert(0, "selection_rank", range(1, len(out) + 1))
    out.insert(1, "selection_lane", lane)
    out.insert(2, "selection_lane_note", note)
    if rank_column in out.columns:
        out.insert(3, "selection_source_rank", out[rank_column])
    out["top3_lane_contract_version"] = TOP3_LANE_CONTRACT_VERSION
    return out


def select_research_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Raw research lane.

    Hard blockers may appear because this lane answers "what deserves research",
    not "what can receive capital". The output labels that distinction explicitly.
    """
    local = _ensure_rank_contract(radar)
    if local.empty:
        return local
    rank = _numeric(local, "raw_research_rank")
    local = local.loc[rank.notna()].copy()
    if local.empty:
        return local
    local["_rank"] = _numeric(local, "raw_research_rank")
    local["_score"] = _numeric(local, "raw_research_score")
    local = local.sort_values(
        ["_rank", "_score"], ascending=[True, False], na_position="last", kind="stable"
    ).head(max(0, int(limit)))
    local = local.drop(columns=["_rank", "_score"])
    return _stamp_lane(
        local,
        lane="RAW_RESEARCH_TOP3",
        note="Research priority only; hard blockers may appear and must never be read as execution authorization.",
        rank_column="raw_research_rank",
    )


def select_guarded_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Guarded decision lane, excluding genuine hard blockers.

    This is the backward-compatible replacement for the old select_top3 used by
    the dashboard. It is still NOT an order-authorization table.
    """
    local = _ensure_rank_contract(radar)
    if local.empty:
        return local
    rank = _numeric(local, "guarded_decision_priority_rank")
    eligible = rank.notna() & ~_hard_block_mask(local)
    local = local.loc[eligible].copy()
    if local.empty:
        return local
    local["_rank"] = _numeric(local, "guarded_decision_priority_rank")
    local["_score"] = _numeric(local, "guarded_decision_priority_score")
    local["_evidence"] = _numeric(local, "emir_evidence_coverage_pct")
    local = local.sort_values(
        ["_rank", "_score", "_evidence"],
        ascending=[True, False, False],
        na_position="last",
        kind="stable",
    ).head(max(0, int(limit)))
    local = local.drop(columns=["_rank", "_score", "_evidence"])
    return _stamp_lane(
        local,
        lane="GUARDED_DECISION_TOP3",
        note="Genuine hard blockers excluded. This lane is decision priority, not capital authorization; use Real Money Gate for execution.",
        rank_column="guarded_decision_priority_rank",
    )


def select_execution_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Fully authorized production candidates; never backfill WAIT/HARD_BLOCK."""
    local = _ensure_rank_contract(radar)
    if local.empty:
        return local
    eligible = production_authorization_mask(local)
    local = local.loc[eligible].copy()
    if local.empty:
        return local

    local["_ready"] = local.get(
        "real_money_ready", pd.Series(False, index=local.index)
    ).map(_truthy).map({True: 0, False: 1})
    local["_production_rank"] = _numeric(local, "production_real_money_rank")
    local["_score"] = _numeric(local, "real_money_candidate_score")
    local["_rr"] = _numeric(local, "real_money_rr_score")
    local = local.sort_values(
        ["_ready", "_production_rank", "_score", "_rr"],
        ascending=[True, True, False, False],
        na_position="last",
        kind="stable",
    ).head(max(0, int(limit)))
    local = local.drop(columns=["_ready", "_production_rank", "_score", "_rr"])
    return _stamp_lane(
        local,
        lane="EXECUTION_TOP3_AUTHORIZED_DIRECT",
        note="Only fail-closed, direct-verified production authorizations. WAIT, manual-confirmation, proxy-only, invalid-geometry, and HARD_BLOCK rows are never backfilled.",
        rank_column="production_real_money_rank",
    )


def select_production_watch_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Quality-qualified real-money candidates still waiting for timing."""
    local = _ensure_rank_contract(radar)
    if local.empty:
        return local
    candidate = local.get(
        "real_money_candidate", pd.Series(False, index=local.index)
    ).map(_truthy)
    entry = local.get(
        "real_money_entry_candidate", pd.Series(False, index=local.index)
    ).map(_truthy)
    eligible = candidate & ~entry & ~_hard_block_mask(local)
    local = local.loc[eligible].copy()
    if local.empty:
        return local
    local["_production_rank"] = _numeric(local, "production_real_money_rank")
    local["_score"] = _numeric(local, "real_money_candidate_score")
    local = local.sort_values(
        ["_production_rank", "_score"],
        ascending=[True, False],
        na_position="last",
        kind="stable",
    ).head(max(0, int(limit)))
    local = local.drop(columns=["_production_rank", "_score"])
    return _stamp_lane(
        local,
        lane="PRODUCTION_WATCH_TOP3_WAIT_TIMING",
        note="Passed quality/hard-risk screen but timing is not actionable. Watchlist only; not an execution Top 3.",
        rank_column="production_real_money_rank",
    )


def _wrap_renderer(owner: Any) -> None:
    original = getattr(owner, "render_top3_dashboard_html", None)
    if not callable(original) or getattr(original, "__explicit_top3_lane_banner_v1__", False):
        return

    @wraps(original)
    def wrapped(top3: pd.DataFrame, *args: Any, **kwargs: Any) -> str:
        html = original(top3, *args, **kwargs)
        if not isinstance(top3, pd.DataFrame) or top3.empty:
            return html
        lane = escape(str(top3.iloc[0].get("selection_lane") or "GUARDED_DECISION_TOP3"))
        note = escape(str(top3.iloc[0].get("selection_lane_note") or ""))
        banner = (
            '<div style="margin:0 0 12px;padding:10px 12px;border:1px solid #35556a;'
            'border-radius:8px;background:#0a1a24;color:#d8edf7;font-family:system-ui">'
            f'<strong>{lane}</strong><br><span style="font-size:12px;color:#9fc1d2">{note}</span>'
            '</div>'
        )
        return banner + html

    wrapped.__explicit_top3_lane_banner_v1__ = True
    setattr(owner, "render_top3_dashboard_html", wrapped)


def install() -> dict[str, str]:
    import top3_dashboard
    import top3_dashboard_legacy

    for owner in (top3_dashboard_legacy, top3_dashboard):
        # Keep a raw research selector available for explicit UI use later.
        setattr(owner, "select_research_top3", select_research_top3)
        setattr(owner, "select_guarded_top3", select_guarded_top3)
        setattr(owner, "select_production_watch_top3", select_production_watch_top3)

        # Backward-compatible names used by the current app become safe lanes.
        setattr(owner, "select_top3", select_guarded_top3)
        setattr(owner, "select_real_money_top3", select_execution_top3)

    _wrap_renderer(top3_dashboard)
    return {
        "patch_version": TOP3_LANE_CONTRACT_VERSION,
        "research_lane": "RAW_RESEARCH_TOP3_MAY_INCLUDE_HARD_BLOCKS_RESEARCH_ONLY",
        "guarded_lane": "GUARDED_DECISION_TOP3_EXCLUDES_GENUINE_HARD_BLOCKERS",
        "execution_lane": "FAIL_CLOSED_DIRECT_AUTHORIZATION_ONLY_NO_BACKFILL",
        "production_watch_lane": "QUALITY_CANDIDATE_WAIT_TIMING_ONLY",
    }


__all__ = [
    "TOP3_LANE_CONTRACT_VERSION",
    "select_research_top3",
    "select_guarded_top3",
    "select_production_watch_top3",
    "select_execution_top3",
    "install",
]
