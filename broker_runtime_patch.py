from __future__ import annotations

from functools import wraps
from typing import Any

import pandas as pd

from public_idx_broker_flow import PUBLIC_CACHE_URL, VERSION, enrich_emir_broker

PATCH_VERSION = "1.1.1-emir-owned-public-idx-broker-cache"


def _wrap(owner: Any) -> None:
    original = getattr(owner, "enrich_dashboard_scores", None)
    if not callable(original) or getattr(original, "__public_idx_broker_v1__", False):
        return

    @wraps(original)
    def wrapped(radar: pd.DataFrame, *args: Any, **kwargs: Any):
        enriched = radar
        try:
            if isinstance(radar, pd.DataFrame) and not radar.empty:
                enriched = enrich_emir_broker(radar)
        except Exception:
            enriched = radar
        return original(enriched, *args, **kwargs)

    wrapped.__public_idx_broker_v1__ = True
    setattr(owner, "enrich_dashboard_scores", wrapped)


def install() -> dict[str, str]:
    import top3_dashboard_legacy
    import top3_dashboard
    from final_decision import finalize_decision_snapshot
    # Keep historical names callable while routing all decision calculation to
    # the same explicit enrich/finalize/rank/freeze implementation.
    top3_dashboard_legacy.enrich_dashboard_scores = finalize_decision_snapshot
    top3_dashboard.enrich_dashboard_scores = finalize_decision_snapshot
    return {
        "patch_version": PATCH_VERSION,
        "broker_flow_version": VERSION,
        "policy": "CANONICAL_IDX_PUBLIC_PARTICIPANT_FLOW_CONFIRMATION_NOT_BENEFICIAL_OWNER_IDENTITY",
        "max_confirmation_weight_pct": "20",
        "cache_policy": "EMIR_OWNED_PUBLIC_PARTICIPANT_CACHE_FAIL_CLOSED",
        "canonical_source": PUBLIC_CACHE_URL,
    }


__all__ = ["PATCH_VERSION", "install"]
