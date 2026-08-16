from __future__ import annotations

"""Runtime hook: bounded ZAPI foreign-flow confirmation for Emir scanner."""

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from zapi_flow_enrichment import (
    ZAPI_FLOW_ENRICHMENT_VERSION,
    blend_emir_dashboard_output,
    enrich_emir_radar,
)

PATCH_VERSION = "1.0.0-emir-zapi-flow"


def _recompute_real_money(owner: Any, frame: pd.DataFrame) -> pd.DataFrame:
    calculator = getattr(owner, "calculate_real_money_candidate_score", None)
    if not callable(calculator) or frame.empty:
        return frame
    out = frame.copy()
    try:
        scores = out.apply(lambda row: pd.Series(calculator(row)), axis=1)
        for column in scores.columns:
            out[column] = scores[column]
    except Exception:
        return frame
    return out


def _adjust_cost_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "smart_money_cost_confidence_pct" not in frame.columns:
        return frame
    out = frame.copy()
    deltas: list[float] = []
    for idx, row in out.iterrows():
        evidence = str(row.get("smart_money_cost_evidence_type") or "").upper()
        if "DIRECT_BROKER" in evidence:
            deltas.append(0.0)
            continue
        score = pd.to_numeric(pd.Series([row.get("zapi_smart_money_confirmation_score")]), errors="coerce").iloc[0]
        cov = pd.to_numeric(pd.Series([row.get("zapi_foreign_flow_coverage_pct")]), errors="coerce").iloc[0]
        base = pd.to_numeric(pd.Series([row.get("smart_money_cost_confidence_pct")]), errors="coerce").iloc[0]
        if not (np.isfinite(score) and np.isfinite(cov) and np.isfinite(base) and cov > 0):
            deltas.append(0.0)
            continue
        directional = float(np.clip((score - 50.0) / 50.0, -1.0, 1.0))
        delta = float(np.clip(8.0 * directional * cov / 100.0, -6.0, 8.0))
        out.at[idx, "smart_money_cost_confidence_pct"] = round(float(np.clip(base + delta, 0.0, 75.0)), 1)
        deltas.append(delta)
    out["zapi_smart_money_cost_confidence_delta"] = deltas
    return out


def _wrap_dashboard_scores(owner: Any) -> None:
    original = getattr(owner, "enrich_dashboard_scores", None)
    if not callable(original) or getattr(original, "__zapi_flow_confirmation_v1__", False):
        return

    @wraps(original)
    def wrapped(radar: pd.DataFrame, *args: Any, **kwargs: Any):
        enriched = radar
        try:
            if isinstance(radar, pd.DataFrame) and not radar.empty:
                enriched = enrich_emir_radar(radar)
        except Exception:
            enriched = radar
        out = original(enriched, *args, **kwargs)
        if isinstance(out, pd.DataFrame) and not out.empty:
            try:
                out = blend_emir_dashboard_output(out)
                out = _recompute_real_money(owner, out)
                out = _adjust_cost_confidence(out)
            except Exception:
                pass
        return out

    wrapped.__zapi_flow_confirmation_v1__ = True
    setattr(owner, "enrich_dashboard_scores", wrapped)


def install() -> dict[str, str]:
    import top3_dashboard_legacy
    import top3_dashboard

    # Install before runtime_integrity_patch. The existing ranking-contract wrapper
    # then remains outermost and ranks the already-ZAPI-confirmed result.
    _wrap_dashboard_scores(top3_dashboard_legacy)
    _wrap_dashboard_scores(top3_dashboard)
    return {
        "patch_version": PATCH_VERSION,
        "zapi_version": ZAPI_FLOW_ENRICHMENT_VERSION,
        "conviction_policy": "BOUNDED_PLUS_MINUS_2_5_POINT_FOREIGN_FLOW_CONFIRMATION",
        "smart_money_policy": "MAX_30_PERCENT_CONFIRMATION_WEIGHT_COVERAGE_AWARE",
        "smc_policy": "PRICE_STRUCTURE_PRIMARY_ZAPI_FLOW_CONFIRMATION_ONLY",
        "cost_policy": "ZAPI_MAY_ADJUST_PROXY_CONFIDENCE_NOT_COST_PRICE_DIRECT_BROKER_WINS",
        "identity_policy": "FOREIGN_FLOW_IS_NOT_BROKER_OR_BENEFICIAL_OWNER_IDENTITY",
    }


__all__ = ["PATCH_VERSION", "install"]
