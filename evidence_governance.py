from __future__ import annotations

"""Evidence, ranking, provider-cache and walk-forward integrity for Emir scanner.

All gates are fail-closed. Proxy evidence cannot satisfy official-forward or
real-money contracts, and OOS calibration cannot activate on unresolved labels.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
import math
import time

import numpy as np
import pandas as pd

EVIDENCE_GOVERNANCE_VERSION = "1.1.0"

CANONICAL_DECISION_COLUMN = "emir_decision_state"
LEGACY_DECISION_COLUMN = "emir_action_state"
ACTIONABLE_PRODUCTION_DECISION_STATES = frozenset({
    "EMIR_READY_WITH_PRECISE_TRIGGER",
})
HARD_BLOCK_DECISION_STATES = frozenset({
    "EMIR_DATA_INTEGRITY_BLOCK",
    "EMIR_REJECT_IDX_INTEGRITY",
    "EMIR_REJECT_SMART_MONEY_DISTRIBUTION",
    "EMIR_AVOID_RETAIL_EUPHORIA",
    "EMIR_CALIBRATION_REJECTED",
})
def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().upper() in {"1", "TRUE", "YES", "Y", "PASS", "VALID", "VERIFIED", "READY"}


def is_https_url(value: Any) -> bool:
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    return parsed.scheme.lower() == "https" and bool(parsed.hostname)


def validate_official_evidence(
    *,
    source_url: Any,
    evidence_date: Any,
    entity_match_verified: Any,
    source_verified: Any,
    source_urls: Sequence[str] | None = None,
    quorum_required: bool = False,
    min_quorum: int = 2,
) -> dict[str, Any]:
    urls = [str(url).strip() for url in (source_urls or []) if str(url or "").strip()]
    primary = str(source_url or "").strip()
    if primary and primary not in urls:
        urls.insert(0, primary)
    distinct_https = list(dict.fromkeys(url for url in urls if is_https_url(url)))
    stamp = pd.to_datetime(evidence_date, errors="coerce", utc=True)
    date_ok = bool(pd.notna(stamp) and stamp <= pd.Timestamp.now(tz="UTC") + pd.to_timedelta(1, unit="D"))
    source_ok = _truthy(source_verified)
    entity_ok = _truthy(entity_match_verified)
    https_ok = is_https_url(primary)
    quorum_count = len(distinct_https)
    quorum_ok = quorum_count >= max(1, int(min_quorum)) if quorum_required else quorum_count >= 1
    valid = bool(source_ok and entity_ok and https_ok and date_ok and quorum_ok)
    failures = []
    if not source_ok: failures.append("SOURCE_NOT_VERIFIED")
    if not https_ok: failures.append("HTTPS_SOURCE_MISSING")
    if not entity_ok: failures.append("ENTITY_MATCH_NOT_VERIFIED")
    if not date_ok: failures.append("EVIDENCE_DATE_MISSING_OR_INVALID")
    if not quorum_ok: failures.append("SOURCE_QUORUM_NOT_MET")
    return {
        "evidence_production_valid": valid,
        "source_https_verified": https_ok,
        "entity_match_verified": entity_ok,
        "evidence_date_verified": date_ok,
        "source_quorum_count": quorum_count,
        "source_quorum_verified": quorum_ok,
        "evidence_validation_state": "VERIFIED_DIRECT_EVIDENCE" if valid else "EVIDENCE_INCOMPLETE_FAIL_CLOSED",
        "evidence_validation_reasons": " | ".join(failures) or "NONE",
    }


def _rank(score: pd.Series, eligible: pd.Series) -> pd.Series:
    score = pd.to_numeric(score, errors="coerce")
    eligible = eligible.fillna(False).astype(bool)
    out = pd.Series(pd.NA, index=score.index, dtype="Int64")
    order = score.loc[eligible & score.notna()].sort_values(ascending=False, kind="stable").index
    out.loc[order] = np.arange(1, len(order) + 1)
    return out


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[column].fillna("").astype(str).str.strip().str.upper()


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_truthy).fillna(False).astype(bool)


def canonical_decision_state(frame: pd.DataFrame) -> pd.Series:
    """Return only the canonical decision state; legacy state never controls gates."""
    if not isinstance(frame, pd.DataFrame):
        return pd.Series(dtype=str)
    return _text(frame, CANONICAL_DECISION_COLUMN)


def production_hard_block_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify explicit hard blockers without treating missing metadata as approval."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(False, index=getattr(frame, "index", None), dtype=bool)
    hard_count = pd.to_numeric(
        frame.get("real_money_hard_block_count", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0).gt(0.0)
    tier = _text(frame, "real_money_authorization_tier").eq("HARD_BLOCKED")
    gate_class = _text(frame, "real_money_gate_class").eq("HARD_BLOCK")
    blocked_decision = canonical_decision_state(frame).isin(HARD_BLOCK_DECISION_STATES)
    return hard_count | tier | gate_class | blocked_decision


def _valid_execution_geometry_mask(frame: pd.DataFrame) -> pd.Series:
    explicit = _flag(frame, "execution_geometry_valid")
    entry = pd.to_numeric(
        frame.get("execution_entry_reference", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    stop = pd.to_numeric(
        frame.get("execution_stop_loss", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    tp1 = pd.to_numeric(
        frame.get("execution_tp1", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    tp2 = pd.to_numeric(
        frame.get("execution_tp2", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    finite = entry.notna() & stop.notna() & tp1.notna() & tp2.notna()
    return explicit & finite & stop.lt(entry) & entry.lt(tp1) & tp1.lt(tp2)


def _minimum_rr_mask(frame: pd.DataFrame) -> pd.Series:
    explicit = _flag(frame, "execution_min_rr_pass")
    path = _text(frame, "preferred_execution_path")
    rr = pd.to_numeric(
        frame.get("execution_rr_tp1", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    required = pd.Series(np.nan, index=frame.index, dtype=float)
    required.loc[path.eq("ACCUMULATION_PULLBACK")] = 1.5
    required.loc[path.eq("BREAKOUT_RETEST")] = 1.8
    return explicit & rr.notna() & required.notna() & rr.ge(required)


def production_authorization_mask(frame: pd.DataFrame) -> pd.Series:
    """Fail-closed authorization shared by production ranking and execution lanes.

    Research and manual-confirmation candidates remain visible elsewhere, but a
    production row must be direct-verified and fully executable at this snapshot.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(False, index=getattr(frame, "index", None), dtype=bool)
    decision_ok = canonical_decision_state(frame).isin(ACTIONABLE_PRODUCTION_DECISION_STATES)
    candidate_ok = _flag(frame, "real_money_candidate") & _flag(frame, "real_money_entry_candidate")
    authorization_ok = (
        _flag(frame, "real_money_ready")
        & _text(frame, "real_money_gate_state").eq("REAL_MONEY_DIRECT_VERIFIED_READY")
        & _text(frame, "entry_authorization_state").eq("SCANNER_AUTHORIZED_DIRECT_VERIFIED")
    )
    return (
        decision_ok
        & candidate_ok
        & authorization_ok
        & ~production_hard_block_mask(frame)
        & _valid_execution_geometry_mask(frame)
        & _minimum_rr_mask(frame)
    ).fillna(False).astype(bool)


def apply_three_rank_contract(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    raw_col = next((c for c in ("emir_conviction_score", "emir_final_score", "next_leader_quality_pre_confidence", "next_leader_score") if c in out.columns), None)
    guarded_col = next((c for c in ("next_leader_score", "emir_final_score", "emir_conviction_score") if c in out.columns), raw_col)
    production_col = next((c for c in ("real_money_candidate_score", "emir_final_score", guarded_col) if c and c in out.columns), guarded_col)
    raw = pd.to_numeric(out.get(raw_col, pd.Series(np.nan, index=out.index)), errors="coerce")
    guarded = pd.to_numeric(out.get(guarded_col, raw), errors="coerce")
    production = pd.to_numeric(out.get(production_col, guarded), errors="coerce")
    canonical_state = canonical_decision_state(out)
    # Compatibility is output-only: the legacy field mirrors the canonical
    # state and can never override it when a stale persisted value disagrees.
    out[LEGACY_DECISION_COLUMN] = canonical_state
    blocked = production_hard_block_mask(out)
    research_ok = raw.notna()
    guarded_ok = guarded.notna() & ~blocked
    production_ok = production_authorization_mask(out)
    out["production_authorization_pass"] = production_ok
    out["real_money_authorization_pass"] = production_ok
    out["execution_authorized"] = production_ok
    out["production_ready"] = production_ok
    out["raw_research_score"] = raw
    out["guarded_decision_priority_score"] = guarded
    out["production_real_money_score"] = production
    out["raw_research_rank"] = _rank(raw, research_ok)
    out["guarded_decision_priority_rank"] = _rank(guarded, guarded_ok)
    out["production_real_money_rank"] = _rank(production, production_ok)
    out["ranking_contract_state"] = "THREE_RANK_CONTRACT_V2_FAIL_CLOSED_AUTHORIZATION"
    return out


def select_enrichment_shortlist(frame: pd.DataFrame, *, limit: int = 24) -> list[str]:
    if frame is None or frame.empty or "ticker" not in frame.columns or limit <= 0:
        return []
    local = apply_three_rank_contract(frame)
    base = pd.to_numeric(local["guarded_decision_priority_score"], errors="coerce").fillna(-1e9)
    percentile = base.rank(pct=True, method="average").fillna(0.0)
    gaps = pd.Series(0.0, index=local.index)
    for column in ("future_direct_forward_lineage_verified", "future_source_quorum_verified", "ownership_source_verified"):
        if column in local.columns:
            gaps += (~local[column].map(_truthy)).astype(float)
    local["_enrichment_priority"] = base + 4.0 * gaps * percentile
    return local.sort_values(["_enrichment_priority", "ticker"], ascending=[False, True], kind="stable").head(int(limit))["ticker"].astype(str).tolist()


@dataclass
class ProviderNegativeCache:
    max_entries: int = 4096
    ttl_seconds: Mapping[str, int] = field(default_factory=lambda: {
        "NOT_FOUND": 86400, "AUTH": 86400, "PARSE": 21600, "RATE_LIMIT": 3600,
        "TIMEOUT": 1800, "SERVER": 900, "EMPTY": 1800, "OTHER": 900,
    })
    _entries: dict[tuple[str, str, str], tuple[float, str]] = field(default_factory=dict)

    def _key(self, provider: Any, family: Any, cache_key: Any) -> tuple[str, str, str]:
        return (str(provider or "UNKNOWN").upper(), str(family or "UNKNOWN").upper(), str(cache_key or "").upper())

    def should_skip(self, provider: Any, family: Any, cache_key: Any) -> bool:
        key = self._key(provider, family, cache_key)
        entry = self._entries.get(key)
        if not entry:
            return False
        if time.monotonic() >= entry[0]:
            self._entries.pop(key, None)
            return False
        return True

    def record_failure(self, provider: Any, family: Any, cache_key: Any, failure_class: str = "OTHER") -> None:
        failure = str(failure_class or "OTHER").upper()
        ttl = int(self.ttl_seconds.get(failure, self.ttl_seconds["OTHER"]))
        self._entries[self._key(provider, family, cache_key)] = (time.monotonic() + max(60, ttl), failure)
        if len(self._entries) > self.max_entries:
            for key, _ in sorted(self._entries.items(), key=lambda item: item[1][0])[: len(self._entries) - self.max_entries]:
                self._entries.pop(key, None)

    def record_success(self, provider: Any, family: Any, cache_key: Any) -> None:
        self._entries.pop(self._key(provider, family, cache_key), None)


DEFAULT_GUARDRAIL_PARAMETERS = {
    "anti_chase_penalty": 3.0,
    "technical_floor": 50.0,
    "technical_slope": 0.08,
    "flow_floor": 40.0,
    "flow_slope": 0.06,
    "distribution_floor": 45.0,
    "distribution_slope": 0.08,
    "distribution_block_penalty": 8.0,
}


def _series(frame: pd.DataFrame, names: Sequence[str], default: float = np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _guarded_score(frame: pd.DataFrame, p: Mapping[str, float]) -> pd.Series:
    raw = _series(frame, ("emir_conviction_score", "emir_final_score", "raw_research_score"))
    technical = _series(frame, ("technical_execution_score", "market_structure_score", "trend_score"))
    flow = _series(frame, ("smart_money_score", "dashboard_flow_score", "broker_inventory_score"))
    distribution = _series(frame, ("distribution_score",), 0.0)
    anti = frame.get("anti_chase_gate", pd.Series(False, index=frame.index)).map(_truthy)
    block = frame.get("distribution_block", pd.Series(False, index=frame.index)).map(_truthy)
    penalty = anti.astype(float) * float(p["anti_chase_penalty"])
    penalty += (float(p["technical_floor"]) - technical).clip(lower=0).fillna(0) * float(p["technical_slope"])
    penalty += (float(p["flow_floor"]) - flow).clip(lower=0).fillna(0) * float(p["flow_slope"])
    penalty += (distribution - float(p["distribution_floor"])).clip(lower=0).fillna(0) * float(p["distribution_slope"])
    penalty += block.astype(float) * float(p["distribution_block_penalty"])
    return raw - penalty


def _objective(frame: pd.DataFrame, p: Mapping[str, float], return_col: str) -> float:
    score = _guarded_score(frame, p)
    ret = pd.to_numeric(frame[return_col], errors="coerce")
    local = pd.DataFrame({"score": score, "ret": ret}).dropna()
    if len(local) < 12:
        return -1e9
    selected = local.sort_values("score", ascending=False, kind="stable").head(max(10, min(30, int(math.ceil(len(local) * 0.20)))))
    return float(selected["ret"].median()) + 4.0 * float((selected["ret"] > 0).mean() - 0.5) + 0.35 * min(0.0, float(selected["ret"].quantile(0.10)))


def calibrate_guardrails_walk_forward(
    outcomes: pd.DataFrame,
    *,
    default_parameters: Mapping[str, float] | None = None,
    return_col: str = "forward_return_20d",
    min_rows: int = 120,
    min_signal_dates: int = 8,
) -> dict[str, Any]:
    defaults = dict(default_parameters or DEFAULT_GUARDRAIL_PARAMETERS)
    if outcomes is None or outcomes.empty or "signal_date" not in outcomes.columns or return_col not in outcomes.columns:
        return {"calibration_state": "INSUFFICIENT_MATURE_OOS_EVIDENCE_KEEP_BASELINE", "active": False, "parameters": defaults, "sample_count": 0, "fold_count": 0}
    local = outcomes.copy()
    local["signal_date"] = pd.to_datetime(local["signal_date"], errors="coerce", utc=True)
    local[return_col] = pd.to_numeric(local[return_col], errors="coerce")
    if "outcome_verified" in local.columns:
        local = local[local["outcome_verified"].map(_truthy)]
    local = local.dropna(subset=["signal_date", return_col])
    dates = list(pd.Index(local["signal_date"].dt.date.unique()).sort_values())
    if len(local) < int(min_rows) or len(dates) < int(min_signal_dates):
        return {"calibration_state": "INSUFFICIENT_MATURE_OOS_EVIDENCE_KEEP_BASELINE", "active": False, "parameters": defaults, "sample_count": int(len(local)), "distinct_signal_dates": len(dates), "fold_count": 0}
    split_points = sorted(set(int(x) for x in np.linspace(max(4, len(dates)//2), len(dates)-1, num=min(4, max(1, len(dates)//2)), dtype=int) if int(x) < len(dates)))
    folds = []
    chosen = []
    for split in split_points:
        train = local[local["signal_date"].dt.date.isin(set(dates[:split]))]
        valid = local[local["signal_date"].dt.date.eq(dates[split])]
        if len(train) < 40 or len(valid) < 5:
            continue
        best_scale, best_train = 1.0, -1e9
        for scale in (0.75, 1.0, 1.25):
            candidate = dict(defaults)
            for key in ("anti_chase_penalty", "technical_slope", "flow_slope", "distribution_slope", "distribution_block_penalty"):
                candidate[key] = float(defaults[key]) * scale
            objective = _objective(train, candidate, return_col)
            if objective > best_train:
                best_scale, best_train = float(scale), objective
        candidate = dict(defaults)
        for key in ("anti_chase_penalty", "technical_slope", "flow_slope", "distribution_slope", "distribution_block_penalty"):
            candidate[key] = float(defaults[key]) * best_scale
        oos, baseline = _objective(valid, candidate, return_col), _objective(valid, defaults, return_col)
        folds.append({"validation_date": str(dates[split]), "chosen_scale": best_scale, "oos_objective": oos, "baseline_objective": baseline, "oos_lift": oos-baseline})
        chosen.append(best_scale)
    if len(folds) < 2:
        return {"calibration_state": "INSUFFICIENT_WALK_FORWARD_FOLDS_KEEP_BASELINE", "active": False, "parameters": defaults, "sample_count": int(len(local)), "distinct_signal_dates": len(dates), "fold_count": len(folds), "folds": folds}
    median_lift = float(np.median([fold["oos_lift"] for fold in folds]))
    positive_rate = float(np.mean([fold["oos_lift"] >= 0 for fold in folds]))
    scale = float(np.median(chosen))
    active = bool(median_lift > 0 and positive_rate >= 0.60 and scale != 1.0)
    selected = dict(defaults)
    if active:
        for key in ("anti_chase_penalty", "technical_slope", "flow_slope", "distribution_slope", "distribution_block_penalty"):
            selected[key] = float(defaults[key]) * scale
    return {
        "calibration_state": "OOS_WALK_FORWARD_ACTIVE" if active else "OOS_NO_STABLE_LIFT_KEEP_BASELINE",
        "active": active, "parameters": selected, "sample_count": int(len(local)), "distinct_signal_dates": len(dates),
        "fold_count": len(folds), "median_oos_lift": median_lift, "positive_fold_rate": positive_rate,
        "selected_penalty_scale": scale if active else 1.0, "folds": folds,
    }


__all__ = [
    "ACTIONABLE_PRODUCTION_DECISION_STATES", "CANONICAL_DECISION_COLUMN",
    "DEFAULT_GUARDRAIL_PARAMETERS", "EVIDENCE_GOVERNANCE_VERSION",
    "HARD_BLOCK_DECISION_STATES", "LEGACY_DECISION_COLUMN", "ProviderNegativeCache",
    "apply_three_rank_contract", "calibrate_guardrails_walk_forward",
    "canonical_decision_state", "is_https_url", "production_authorization_mask",
    "production_hard_block_mask", "select_enrichment_shortlist", "validate_official_evidence",
]
