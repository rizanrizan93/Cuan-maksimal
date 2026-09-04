from __future__ import annotations

"""Read-only Phase 5.6 public ownership-concentration context for EMIR.

The source exposes scanner-neutral public-provider facts only. These fields are
strictly context: they do not become KSEI evidence, free float, beneficial
ownership, ownership score, coverage score, broker/bandar identity, or an IDX
integrity clearance.
"""

import os
import time
from typing import Any, Callable, Iterable, Mapping

import requests

from shared_fundamental_runtime import bare_ticker
from phase56_public_fundamental_projection import PUBLIC_PROJECTION_KEY

PATCH_VERSION = "1.1.0-phase5.6-public-ownership-context-ttl"
PUBLIC_OWNERSHIP_URL = os.getenv(
    "PHASE56_PUBLIC_OWNERSHIP_URL",
    "https://mbtsvflwszcgdtijdgas.supabase.co/rest/v1/phase56_public_ownership_snapshots",
).strip()
PUBLIC_OWNERSHIP_KEY = os.getenv("PHASE56_PUBLIC_OWNERSHIP_KEY", PUBLIC_PROJECTION_KEY).strip()
REQUEST_TIMEOUT_SECONDS = 10
CACHE_TTL_SECONDS = 6 * 60 * 60
CACHE_RETRY_SECONDS = 15 * 60
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


class _OwnershipContextCache:
    """Bounded cache that refreshes long-lived Streamlit workers safely.

    A successful public-projection read is kept for six hours. If a refresh
    fails, the last known-good context is retained and the endpoint is retried
    after 15 minutes instead of being called on every scan.
    """

    def __init__(
        self,
        loader: Callable[[], tuple[dict[str, dict[str, Any]], dict[str, Any]]],
        *,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        retry_seconds: float = CACHE_RETRY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.loader = loader
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.retry_seconds = max(1.0, float(retry_seconds))
        self.clock = clock
        self.context: dict[str, dict[str, Any]] = {}
        self.last_success_at: float | None = None
        self.last_attempt_at: float | None = None
        self.last_meta: dict[str, Any] = {"state": "PUBLIC_OWNERSHIP_NOT_LOADED", "rows": 0, "tickers": 0}

    def _due(self, now: float) -> bool:
        if self.last_attempt_at is None:
            return True
        if self.last_success_at is None:
            return now - self.last_attempt_at >= self.retry_seconds
        if self.last_attempt_at > self.last_success_at:
            return now - self.last_attempt_at >= self.retry_seconds
        return now - self.last_success_at >= self.ttl_seconds

    def get(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        now = float(self.clock())
        if not self._due(now):
            return self.context, dict(self.last_meta)
        self.last_attempt_at = now
        fresh, meta = self.loader()
        self.last_meta = dict(meta or {})
        if str(self.last_meta.get("state") or "") == "PUBLIC_OWNERSHIP_LOADED" and fresh:
            self.context = {ticker: dict(payload) for ticker, payload in fresh.items()}
            self.last_success_at = now
        elif self.context:
            self.last_meta["fallback_state"] = "LAST_KNOWN_GOOD_PUBLIC_OWNERSHIP_CONTEXT"
            self.last_meta["tickers"] = len(self.context)
        return self.context, dict(self.last_meta)


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
    context_cache = _OwnershipContextCache(fetch_public_ownership_context)

    def ksei_maps_with_public_context(ksei_profiles, ksei_actions, as_of=None):
        ownership_map, integrity_map = original_ksei_maps(ksei_profiles, ksei_actions, as_of=as_of)
        public_context, _meta = context_cache.get()
        return merge_public_context(ownership_map, public_context), integrity_map

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
    "CACHE_RETRY_SECONDS",
    "CACHE_TTL_SECONDS",
    "CONTEXT_FIELDS",
    "PATCH_VERSION",
    "PUBLIC_OWNERSHIP_URL",
    "_OwnershipContextCache",
    "fetch_public_ownership_context",
    "install",
    "merge_public_context",
    "_rows_to_context",
]
