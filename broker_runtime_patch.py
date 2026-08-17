from __future__ import annotations

from functools import wraps
from typing import Any

import pandas as pd

from public_idx_broker_flow import VERSION, enrich_emir_broker

PATCH_VERSION = "1.0.0-emir-public-idx-broker-runtime"


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
    _wrap(top3_dashboard_legacy)
    _wrap(top3_dashboard)
    return {
        "patch_version": PATCH_VERSION,
        "broker_flow_version": VERSION,
        "policy": "OFFICIAL_IDX_PUBLIC_PARTICIPANT_FLOW_CONFIRMATION_NOT_BENEFICIAL_OWNER_IDENTITY",
        "max_confirmation_weight_pct": "20",
        "cache_policy": "SHARED_30D_ROLLING_GITHUB_CACHE_FAIL_SOFT",
    }


__all__ = ["PATCH_VERSION", "install"]
