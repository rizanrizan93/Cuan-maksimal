from __future__ import annotations

"""Canonical Emir enrich -> finalize -> rank -> freeze decision contract.

Decision-critical callers use this module directly.  Runtime patch installers
may keep compatibility names on dashboard modules, but patch/import order is no
longer part of persistence, UI, export, or execution-selection semantics.
"""

import hashlib
import json
from typing import Any, Mapping

import pandas as pd

import public_idx_broker_flow
import top3_dashboard
import top3_dashboard_legacy
import zapi_flow_enrichment
from evidence_governance import apply_three_rank_contract
from execution_research_top3_runtime_patch import select_research_top3 as _select_research_top3
from top3_lane_patch import (
    select_execution_top3 as _select_execution_top3,
    select_guarded_top3 as _select_guarded_top3,
    select_production_watch_top3 as _select_production_watch_top3,
)
from zapi_post_calibration import enrich_emir_shadow, refine_emir_proxy_authorization_tier
from zapi_runtime_patch import (
    _adjust_cost_confidence,
    _apply_foreign_shock_guard,
    _apply_gate_audit,
)
from release_contract import SCANNER_RELEASE_VERSION


FINAL_DECISION_VERSION = "1.0.0-emir-p1-enrich-finalize-rank-freeze"
SCANNER_VERSION = SCANNER_RELEASE_VERSION
FINAL_DECISION_STATE = "FINAL_DECISION_FROZEN"
FINAL_PIPELINE_STATE = "ENRICHED_RECOMPUTED_GATED_RANKED_FROZEN"

_FINGERPRINT_FIELDS = (
    # Identity/session.
    "ticker",
    "symbol",
    "scan_id",
    "scan_run_id",
    "decision_snapshot_version",
    "completed_session",
    "effective_session",
    "as_of",
    "scan_as_of",
    # Canonical state, scores, ranks, and lane/contract metadata.
    "emir_decision_state",
    "emir_action_state",
    "emir_conviction_score",
    "emir_final_score",
    "raw_research_score",
    "guarded_decision_priority_score",
    "production_real_money_score",
    "next_leader_score",
    "smart_money_score",
    "dashboard_flow_score",
    "dashboard_silent_accum_score",
    "real_money_candidate_score",
    "real_money_rr_score",
    "raw_research_rank",
    "guarded_decision_priority_rank",
    "production_real_money_rank",
    "next_leader_universe_rank",
    "dashboard_universe_rank",
    "ranking_contract_state",
    "ranking_contract_version",
    "selection_lane",
    "production_tier",
    "real_money_authorization_tier",
    # Authorization and decision gates.
    "production_ready",
    "production_authorization_pass",
    "real_money_authorization_pass",
    "execution_authorized",
    "real_money_candidate",
    "real_money_entry_candidate",
    "real_money_ready",
    "auto_eod_ready",
    "real_money_gate_state",
    "real_money_gate_class",
    "real_money_authorization_tier_note",
    "real_money_proxy_authorization_eligible",
    "entry_authorization_state",
    "entry_eligibility_state",
    "idx_integrity_ready",
    # Blockers, guards, and audit outcomes.
    "real_money_hard_block_count",
    "real_money_manual_condition_count",
    "real_money_evidence_gap_count",
    "real_money_block_reasons",
    "real_money_evidence_gap_flags",
    "real_money_manual_conditions",
    "real_money_material_risk_flags",
    "real_money_gate_explanation",
    "zapi_foreign_shock_state",
    "zapi_foreign_shock_severity",
    "zapi_execution_flow_guard_state",
    "zapi_execution_flow_guard_reason",
    "zapi_pre_guard_real_money_gate_state",
    "zapi_pre_guard_real_money_ready",
    "risk_flags",
    "execution_capacity_state",
    # Canonical execution plan and backward-compatible plan aliases.
    "preferred_execution_path",
    "execution_state",
    "execution_geometry_state",
    "execution_entry_low",
    "execution_entry_high",
    "execution_entry_reference",
    "execution_trigger",
    "execution_stop_loss",
    "execution_tp1",
    "execution_tp2",
    "execution_rr_tp1",
    "execution_rr_tp2",
    "execution_geometry_valid",
    "execution_min_rr_pass",
    "execution_targets_structural",
    "entry_low",
    "entry_high",
    "stop_loss",
    "tp1",
    "tp2",
    "rr_tp1",
    "rr_tp2",
    # Decision-material evidence provenance, freshness, and coverage.
    "future_forward_provenance_state",
    "future_direct_forward_authorization_eligible",
    "future_direct_forward_lineage_verified",
    "future_source_quorum_verified",
    "future_verified_forward_event_count",
    "future_official_forward_event_count",
    "future_public_research_forward_event_count",
    "future_direct_forward_visibility_score",
    "future_direct_forward_visibility_coverage_pct",
    "real_money_fundamental_evidence_tier",
    "real_money_narrative_evidence_tier",
    "broker_summary_provenance_state",
    "broker_inventory_evidence_type",
    "broker_flow_provenance",
    "broker_flow_coverage_pct",
    "broker_enrichment_compatibility_state",
    "zapi_foreign_flow_coverage_pct",
    "zapi_emir_flow_basis",
    "zapi_enrichment_compatibility_state",
    "fundamental_period_freshness_state",
    "fundamental_availability_state",
    "idx_integrity_provenance_state",
    "orderbook_provenance_state",
)
_FINGERPRINT_BOOLEAN_FIELDS = frozenset({
    "production_ready", "production_authorization_pass", "real_money_authorization_pass",
    "execution_authorized", "real_money_candidate", "real_money_entry_candidate",
    "real_money_ready", "auto_eod_ready", "idx_integrity_ready",
    "execution_geometry_valid", "execution_min_rr_pass", "execution_targets_structural",
    "future_direct_forward_authorization_eligible",
    "future_direct_forward_lineage_verified", "future_source_quorum_verified",
    "real_money_proxy_authorization_eligible", "zapi_pre_guard_real_money_ready",
})
_FINGERPRINT_TIMESTAMP_FIELDS = frozenset({"as_of", "scan_as_of", "completed_session", "effective_session"})


def _row_identity(row: Mapping[str, Any]) -> str:
    value = row.get("ticker")
    if value is None or pd.isna(value) or not str(value).strip():
        value = row.get("symbol")
    identity = str(value or "").strip().upper()
    return identity.removesuffix(".JK")


def _is_numeric_material_field(column: str) -> bool:
    return bool(
        column.endswith(("_score", "_rank", "_pct", "_count"))
        or column.startswith(("execution_entry", "execution_stop", "execution_tp", "execution_rr"))
        or column in {"entry_low", "entry_high", "stop_loss", "tp1", "tp2", "rr_tp1", "rr_tp2"}
    )


def _canonical_value(value: Any, column: str = "") -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if column in _FINGERPRINT_BOOLEAN_FIELDS:
        text = str(value).strip().upper()
        if text in {"TRUE", "YES", "Y", "ON", "READY", "VERIFIED", "PASS", "VALID"}:
            return True
        if text in {"FALSE", "NO", "N", "OFF", "NOT_READY", "UNVERIFIED", "FAIL", "INVALID"}:
            return False
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return bool(float(number)) if pd.notna(number) else False
    if column in _FINGERPRINT_TIMESTAMP_FIELDS:
        stamp = pd.to_datetime(value, errors="coerce", utc=True)
        return None if pd.isna(stamp) else stamp.isoformat()
    if _is_numeric_material_field(column):
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return None if pd.isna(number) else format(float(number), ".17g")
    return str(value).strip()


def _fingerprint(frame: pd.DataFrame) -> str:
    columns = [column for column in _FINGERPRINT_FIELDS if column in frame.columns]
    if not columns:
        raise ValueError("FINAL_DECISION_FINGERPRINT_HAS_NO_MATERIAL_FIELDS")
    records: list[tuple[str, dict[str, Any]]] = []
    identities: set[str] = set()
    for _, series in frame.loc[:, columns].iterrows():
        row = series.to_dict()
        identity = _row_identity(row)
        if not identity:
            raise ValueError("FINAL_DECISION_ROW_IDENTITY_MISSING")
        if identity in identities:
            raise ValueError(f"FINAL_DECISION_DUPLICATE_ROW_IDENTITY:{identity}")
        identities.add(identity)
        record = {column: _canonical_value(row.get(column), column) for column in columns}
        record["ticker"] = identity
        records.append((identity, record))
    canonical = [record for _, record in sorted(records, key=lambda item: item[0])]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_final_decision_snapshot(frame: pd.DataFrame) -> bool:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return False
    required = {
        "decision_snapshot_state",
        "decision_snapshot_version",
        "decision_snapshot_fingerprint",
        "raw_research_rank",
        "guarded_decision_priority_rank",
        "production_real_money_rank",
    }
    if not required.issubset(frame.columns):
        return False
    structurally_frozen = bool(
        frame["decision_snapshot_state"].eq(FINAL_DECISION_STATE).all()
        and frame["decision_snapshot_version"].eq(FINAL_DECISION_VERSION).all()
        and frame["decision_snapshot_fingerprint"].nunique(dropna=False) == 1
    )
    if not structurally_frozen:
        return False
    expected = str(frame["decision_snapshot_fingerprint"].iloc[0])
    try:
        observed = _fingerprint(frame)
    except (TypeError, ValueError):
        return False
    return bool(expected and expected == observed)


def _recompute_real_money(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    scores = out.apply(
        lambda row: pd.Series(top3_dashboard_legacy.calculate_real_money_candidate_score(row)),
        axis=1,
    )
    for column in scores.columns:
        out[column] = scores[column]
    return out


def finalize_decision_snapshot(
    radar: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return one reusable final decision snapshot.

    A frozen persisted snapshot is copied, never enriched or ranked again.  New
    input follows exactly one material-enrichment pass and one ranking pass.
    """
    if not isinstance(radar, pd.DataFrame):
        return pd.DataFrame()
    if radar.empty:
        return radar.copy(deep=True)
    if is_final_decision_snapshot(radar):
        return radar.copy(deep=True)

    out = radar.copy(deep=True)
    stale_columns = [
        "decision_snapshot_state",
        "decision_snapshot_version",
        "decision_snapshot_fingerprint",
        "decision_pipeline_state",
        "decision_enrichment_pass_count",
        "decision_ranking_pass_count",
        "raw_research_rank",
        "guarded_decision_priority_rank",
        "production_real_money_rank",
    ]
    out = out.drop(columns=stale_columns, errors="ignore")

    # Material enrichment is explicit and ordered.  Each provider derives from
    # persisted base fields, so even direct repeat calls remain idempotent.
    out = zapi_flow_enrichment.enrich_emir_radar(out)
    out = public_idx_broker_flow.enrich_emir_broker(out)
    out = top3_dashboard._canonical_enrich_dashboard_scores(out, frames or {})
    out = zapi_flow_enrichment.blend_emir_dashboard_output(out)

    # Recompute dependent execution scores and gates only after enrichment.
    out = _recompute_real_money(out)
    out = _adjust_cost_confidence(out)
    out = _apply_foreign_shock_guard(out)
    out = _apply_gate_audit(out)
    out = refine_emir_proxy_authorization_tier(out)
    out = enrich_emir_shadow(out)

    # Lane ranks are the last calculated decision fields.
    out = apply_three_rank_contract(out)
    out["decision_snapshot_state"] = FINAL_DECISION_STATE
    out["decision_snapshot_version"] = FINAL_DECISION_VERSION
    out["decision_pipeline_state"] = FINAL_PIPELINE_STATE
    out["decision_enrichment_pass_count"] = 1
    out["decision_ranking_pass_count"] = 1
    fingerprint = _fingerprint(out)
    out["decision_snapshot_fingerprint"] = fingerprint
    return out.copy(deep=True)


def select_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    return _select_research_top3(finalize_decision_snapshot(radar), limit)


def select_research_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    return _select_research_top3(finalize_decision_snapshot(radar), limit)


def select_guarded_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    return _select_guarded_top3(finalize_decision_snapshot(radar), limit)


def select_production_watch_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    return _select_production_watch_top3(finalize_decision_snapshot(radar), limit)


def select_real_money_top3(radar: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    return _select_execution_top3(finalize_decision_snapshot(radar), limit)


def select_next_leaders(radar: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    return top3_dashboard_legacy.select_next_leaders(finalize_decision_snapshot(radar), limit)


def export_decision_snapshot(radar: pd.DataFrame) -> pd.DataFrame:
    """Return a detached export view without mutating or re-finalizing state."""
    return finalize_decision_snapshot(radar).copy(deep=True)


def render_top3_dashboard_html(top3: pd.DataFrame, **kwargs: Any) -> str:
    """Rendering is deliberately read-only and never invokes enrichment."""
    return top3_dashboard.render_top3_dashboard_html(top3.copy(deep=True), **kwargs)


__all__ = [
    "FINAL_DECISION_STATE",
    "FINAL_DECISION_VERSION",
    "FINAL_PIPELINE_STATE",
    "SCANNER_VERSION",
    "export_decision_snapshot",
    "finalize_decision_snapshot",
    "is_final_decision_snapshot",
    "render_top3_dashboard_html",
    "select_guarded_top3",
    "select_next_leaders",
    "select_production_watch_top3",
    "select_real_money_top3",
    "select_research_top3",
    "select_top3",
]
