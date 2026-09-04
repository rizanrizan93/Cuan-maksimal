from __future__ import annotations

"""Public read-only Phase 5.6 factual projection fallback for EMIR.

The source table exposes scanner-neutral public-source facts only. It never
contains score, rank, gate, recommendation, entry, stop, target, or Future
Fundamental outputs. Private Shared Hub access remains the preferred route.
"""

import os
from typing import Any, Iterable, Mapping

import requests

from shared_fundamental_runtime import bare_ticker

PATCH_VERSION = "1.0.0-phase5.6-public-projection"
PUBLIC_PROJECTION_URL = os.getenv(
    "PHASE56_PUBLIC_FUNDAMENTAL_URL",
    "https://mbtsvflwszcgdtijdgas.supabase.co/rest/v1/phase56_public_fundamental_snapshots",
).strip()
# Supabase publishable keys are intentionally safe for client-side use. This
# key can be rotated independently; an environment override is preferred when set.
PUBLIC_PROJECTION_KEY = os.getenv(
    "PHASE56_PUBLIC_FUNDAMENTAL_KEY",
    "sb_publishable_Oz548O0F21T8UZgZc9Xcew_mMCRFl2B",
).strip()
REQUEST_TIMEOUT_SECONDS = 10


def _rows_to_bundle(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        ticker = bare_ticker(row.get("ticker"))
        proxy = row.get("proxy_metrics") if isinstance(row.get("proxy_metrics"), Mapping) else {}
        official = row.get("official_metrics") if isinstance(row.get("official_metrics"), Mapping) else {}
        families = row.get("source_families") if isinstance(row.get("source_families"), list) else []
        if not ticker or (not proxy and not official):
            continue
        bundle[ticker] = {
            "ticker": ticker,
            "proxy_metrics": dict(proxy),
            "proxy_period_end": row.get("proxy_period_end"),
            "proxy_observed_at": row.get("proxy_observed_at"),
            "official_metrics": dict(official),
            "official_period_end": row.get("official_period_end"),
            "official_observed_at": row.get("official_observed_at"),
            "official_coverage_pct": row.get("official_coverage_pct") or 0.0,
            "source_families": [str(item) for item in families if str(item or "").strip()],
        }
    return bundle


def fetch_public_bundle(tickers: Iterable[str], *, session: requests.Session | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    symbols = list(dict.fromkeys(bare_ticker(value) for value in tickers if bare_ticker(value)))
    if not symbols or not PUBLIC_PROJECTION_URL or not PUBLIC_PROJECTION_KEY:
        return {}, {"state": "PUBLIC_PROJECTION_ENVIRONMENT_BLOCKED", "rows": 0, "tickers": 0}
    http = session or requests.Session()
    rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(symbols), 80):
            chunk = symbols[start:start + 80]
            quoted = ",".join(f'"{symbol}"' for symbol in chunk)
            response = http.get(
                PUBLIC_PROJECTION_URL,
                params={
                    "select": "ticker,proxy_period_end,proxy_observed_at,official_period_end,official_observed_at,proxy_metrics,official_metrics,source_families,official_coverage_pct,source_state,refreshed_at",
                    "ticker": f"in.({quoted})",
                    "limit": 1000,
                },
                headers={"apikey": PUBLIC_PROJECTION_KEY, "Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                return {}, {"state": f"PUBLIC_PROJECTION_HTTP_{response.status_code}", "rows": len(rows), "tickers": 0}
            payload = response.json()
            if not isinstance(payload, list):
                return {}, {"state": "PUBLIC_PROJECTION_PARSE_FAILURE", "rows": len(rows), "tickers": 0}
            rows.extend(dict(item) for item in payload if isinstance(item, Mapping))
    except requests.Timeout:
        return {}, {"state": "PUBLIC_PROJECTION_TIMEOUT", "rows": len(rows), "tickers": 0}
    except requests.ConnectionError:
        return {}, {"state": "PUBLIC_PROJECTION_CONNECTION_ERROR", "rows": len(rows), "tickers": 0}
    except Exception as exc:
        return {}, {"state": f"PUBLIC_PROJECTION_ERROR:{type(exc).__name__}", "rows": len(rows), "tickers": 0}
    bundle = _rows_to_bundle(rows)
    return bundle, {"state": "PUBLIC_PROJECTION_LOADED", "rows": len(rows), "tickers": len(bundle)}


def install() -> None:
    import shared_fundamental_runtime_patch as runtime_patch

    if getattr(runtime_patch, "_phase56_public_projection_patch", "") == PATCH_VERSION:
        return
    original = runtime_patch._read_shared

    def read_shared_with_public_fallback(tickers: Iterable[str]):
        proxy, official, meta = original(tickers)
        if not proxy.empty or not official.empty:
            return proxy, official, meta
        bundle, public_meta = fetch_public_bundle(tickers)
        if not bundle:
            return proxy, official, {"private_state": meta.get("state"), **public_meta}
        public_proxy, public_official = runtime_patch._bundle_frames(bundle)
        return public_proxy, public_official, {"private_state": meta.get("state"), **public_meta}

    runtime_patch._read_shared = read_shared_with_public_fallback
    runtime_patch._phase56_public_projection_patch = PATCH_VERSION


__all__ = [
    "PATCH_VERSION",
    "PUBLIC_PROJECTION_URL",
    "fetch_public_bundle",
    "install",
    "_rows_to_bundle",
]
