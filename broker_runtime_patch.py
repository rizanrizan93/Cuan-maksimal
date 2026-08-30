from __future__ import annotations

from functools import wraps
from io import BytesIO
import gzip
from typing import Any

import pandas as pd
import requests

from public_idx_broker_flow import VERSION, enrich_emir_broker

PATCH_VERSION = "1.1.0-emir-public-idx-broker-canonical-bridge"
CANONICAL_PUBLIC_PARTICIPANT_URL = "https://raw.githubusercontent.com/rizanrizan93/idx-flow-scanner/main/data/cache/idx_public_participant_30d.csv.gz"


def _normalize_canonical_participant_cache(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    required = {"ticker", "trade_date", "participant", "buy_value", "sell_value", "buy_volume", "sell_volume"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    out["broker_code"] = out["participant"].astype(str).str.strip().str.upper()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.removesuffix(".JK")
    for column in ("buy_value", "sell_value", "buy_volume", "sell_volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["buy_avg"] = out["buy_value"].div(out["buy_volume"].replace(0.0, pd.NA))
    out["sell_avg"] = out["sell_value"].div(out["sell_volume"].replace(0.0, pd.NA))
    out["net_value"] = out["buy_value"] - out["sell_value"]
    out["net_volume"] = out["buy_volume"] - out["sell_volume"]
    out["gross_value"] = out["buy_value"] + out["sell_value"]
    out["source"] = "IDX_OFFICIAL_PUBLIC_TRADE_DETAIL_PARTICIPANT_FLOW"
    out["source_verified"] = True
    out["provenance_state"] = "VERIFIED_IDX_PUBLIC_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER"
    out["side"] = out["net_value"].map(lambda x: "TOP_NET_BUYER" if x > 0 else ("TOP_NET_SELLER" if x < 0 else "NEUTRAL"))
    out["net_rank"] = out.groupby(["trade_date", "ticker", "side"])["net_value"].rank(method="first", ascending=False)
    return out.dropna(subset=["ticker", "trade_date", "broker_code"]).reset_index(drop=True)


def _install_canonical_cache_bridge() -> None:
    import public_idx_broker_flow as broker_module
    original = getattr(broker_module, "load_public_cache", None)
    if not callable(original) or getattr(original, "__canonical_idx_participant_bridge_v1__", False):
        return

    @wraps(original)
    def wrapped() -> pd.DataFrame:
        try:
            response = requests.get(
                CANONICAL_PUBLIC_PARTICIPANT_URL,
                timeout=12,
                headers={"User-Agent": "IDX-Scanner-Broker-Bridge/1.0"},
            )
            response.raise_for_status()
            canonical = _normalize_canonical_participant_cache(pd.read_csv(BytesIO(gzip.decompress(response.content))))
            if not canonical.empty:
                return canonical
        except Exception:
            pass
        return original()

    wrapped.__canonical_idx_participant_bridge_v1__ = True
    broker_module.load_public_cache = wrapped


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
    _install_canonical_cache_bridge()
    # Keep historical names callable while routing all decision calculation to
    # the same explicit enrich/finalize/rank/freeze implementation.
    top3_dashboard_legacy.enrich_dashboard_scores = finalize_decision_snapshot
    top3_dashboard.enrich_dashboard_scores = finalize_decision_snapshot
    return {
        "patch_version": PATCH_VERSION,
        "broker_flow_version": VERSION,
        "policy": "CANONICAL_IDX_PUBLIC_PARTICIPANT_FLOW_CONFIRMATION_NOT_BENEFICIAL_OWNER_IDENTITY",
        "max_confirmation_weight_pct": "20",
        "cache_policy": "IDX_FLOW_SCANNER_PUBLIC_PARTICIPANT_CACHE_PRIMARY_PASTICUAN_CACHE_FAIL_SOFT_FALLBACK",
        "canonical_source": CANONICAL_PUBLIC_PARTICIPANT_URL,
    }


__all__ = ["PATCH_VERSION", "install"]
