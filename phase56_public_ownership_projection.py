from __future__ import annotations

"""Read-only Phase 5.6 public ownership-concentration context for EMIR.

The source exposes scanner-neutral public-provider facts only. These fields are
strictly context: they do not become KSEI evidence, free float, beneficial
ownership, ownership score, coverage score, broker/bandar identity, or an IDX
integrity clearance.
"""

import os
from typing import Any, Iterable, Mapping

import requests

from shared_fundamental_runtime import bare_ticker
from phase56_public_fundamental_projection import PUBLIC_PROJECTION_KEY

PATCH_VERSION = "1.0.0-phase5.6-public-ownership-context"
PUBLIC_OWNERSHIP_URL = os.getenv(
    "PHASE56_PUBLIC_OWNERSHIP_URL",
    "https://mbtsvflwszcgdtijdgas.supabase.co/rest/v1/phase56_public_ownership_snapshots",
).strip()
PUBLIC_OWNERSHIP_KEY = os.getenv("PHASE56_PUBLIC_OWNERSHIP_KEY", PUBLIC_PROJECTION_KEY).strip()
REQUEST_TIMEOUT_SECONDS = 10
CONTEXT_FIELDS = (
    "ownership_public_insiders_held_pct",
    "ownership_public_institutions_held_pct",
    "ownership_public_institutions_float_held_pct",
    "ownership_public_institutions_count",
    "ownership_public_context_coverage_pct",
    "ownership_public_source_period",
    "ownership_public_observed_on",
    "ownership_public_context_provenance_state",
)


def _rows_to_context(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        ticker = bare_ticker(row.get("ticker"))
        if not ticker:
            continue
        output[ticker] = {
            "ownership_public_insiders_held_pct": row.get("insiders_held_pct"),
            "ownership_public_institutions_held_pct": row.get("institutions_held_pct"),
            "ownership_public_institutions_float_held_pct": row.get("institutions_float_held_pct"),
            "ownership_public_institutions_count": row.get("institutions_count"),
            "ownership_public_context_coverage_pct": row.get("coverage_pct") or 0.0,
            "ownership_public_source_period": row.get("source_period"),
            "ownership_public_observed_on": row.get("observed_on"),
            "ownership_public_context_provenance_state": str(row.get("provenance_state") or "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI"),
        }
    return output


def fetch_public_ownership_context(
    tickers: Iterable[str] | None = None,
    *,
    session: requests.Session | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    symbols = list(dict.fromkeys(bare_ticker(value) for value in (tickers or []) if bare_ticker(value)))
    if not PUBLIC_OWNERSHIP_URL or not PUBLIC_OWNERSHIP_KEY:
        return {}, {"state": "PUBLIC_OWNERSHIP_ENVIRONMENT_BLOCKED", "rows": 0, "tickers": 0}
    http = session or requests.Session()
    rows: list[dict[str, Any]] = []
    chunks: list[list[str] | None] = [None] if not symbols else [symbols[start:start + 80] for start in range(0, len(symbols), 80)]
    try:
        for chunk in chunks:
            params: dict[str, Any] = {
                "select": "ticker,source_period,observed_on,insiders_held_pct,institutions_held_pct,institutions_float_held_pct,institutions_count,coverage_pct,source_authority,official_verified,provenance_state,source_state,refreshed_at",
                "limit": 1000,
            }
            if chunk:
                quoted = ",".join(f'"{symbol}"' for symbol in chunk)
                params["ticker"] = f"in.({quoted})"
            response = http.get(
                PUBLIC_OWNERSHIP_URL,
                params=params,
                headers={"apikey": PUBLIC_OWNERSHIP_KEY, "Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                return {}, {"state": f"PUBLIC_OWNERSHIP_HTTP_{response.status_code}", "rows": len(rows), "tickers": 0}
            payload = response.json()
            if not isinstance(payload, list):
                return {}, {"state": "PUBLIC_OWNERSHIP_PARSE_FAILURE", "rows": len(rows), "tickers": 0}
            rows.extend(dict(item) for item in payload if isinstance(item, Mapping))
    except requests.Timeout:
        return {}, {"state": "PUBLIC_OWNERSHIP_TIMEOUT", "rows": len(rows), "tickers": 0}
    except requests.ConnectionError:
        return {}, {"state": "PUBLIC_OWNERSHIP_CONNECTION_ERROR", "rows": len(rows), "tickers": 0}
    except Exception as exc:
        return {}, {"state": f"PUBLIC_OWNERSHIP_ERROR:{type(exc).__name__}", "rows": len(rows), "tickers": 0}
    context = _rows_to_context(rows)
    return context, {"state": "PUBLIC_OWNERSHIP_LOADED", "rows": len(rows), "tickers": len(context)}


def merge_public_context(
    base_map: Mapping[str, Mapping[str, Any]] | None,
    public_map: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    output = {str(ticker): dict(payload or {}) for ticker, payload in dict(base_map or {}).items()}
    for ticker, context in dict(public_map or {}).items():
        merged = dict(output.get(str(ticker)) or {})
        for field in CONTEXT_FIELDS:
            value = dict(context or {}).get(field)
            if value is not None and not (isinstance(value, str) and not value.strip()):
                merged[field] = value
        output[str(ticker)] = merged
    return output


def install() -> None:
    import resumable_scan as scan

    if getattr(scan, "_phase56_public_ownership_patch", "") == PATCH_VERSION:
        return

    original_ksei_maps = scan.ksei_profiles_to_maps
    original_build_profile = scan.build_emir_profile
    cached_context: dict[str, dict[str, Any]] | None = None

    def ksei_maps_with_public_context(ksei_profiles, ksei_actions, as_of=None):
        nonlocal cached_context
        ownership_map, integrity_map = original_ksei_maps(ksei_profiles, ksei_actions, as_of=as_of)
        if cached_context is None:
            cached_context, _meta = fetch_public_ownership_context()
        return merge_public_context(ownership_map, cached_context), integrity_map

    def build_profile_with_public_context(*args, **kwargs):
        result = original_build_profile(*args, **kwargs)
        ownership = kwargs.get("ownership") if isinstance(kwargs.get("ownership"), Mapping) else {}
        if isinstance(result, Mapping):
            result = dict(result)
            for field in CONTEXT_FIELDS:
                if field in ownership:
                    result[field] = ownership[field]
        return result

    scan.ksei_profiles_to_maps = ksei_maps_with_public_context
    scan.build_emir_profile = build_profile_with_public_context
    scan._phase56_public_ownership_patch = PATCH_VERSION


__all__ = [
    "CONTEXT_FIELDS",
    "PATCH_VERSION",
    "PUBLIC_OWNERSHIP_URL",
    "fetch_public_ownership_context",
    "install",
    "merge_public_context",
    "_rows_to_context",
]
