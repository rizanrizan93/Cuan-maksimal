from __future__ import annotations

"""Mirror canonical Phase 5.6 shared facts into native EMIR cache rows.

This is a cache mirror only. It does not transfer score/rank/gate/Future Fundamental.
Fresh compatible EMIR rows are preserved; only stale/missing/lagged rows can be replaced.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import os
from typing import Any, Mapping

import pandas as pd

import persistent_cache as pc
import persistence
from shared_fundamental_runtime import SharedFundamentalRuntime
from shared_fundamental_runtime_patch import _official_good, _official_payload, _proxy_payload, _shared_good


def _local_config() -> persistence.DatabaseConfig:
    mapping = dict(os.environ)
    mapping["CAK_DATABASE_ENABLED"] = "true"
    mapping.setdefault("CAK_DATABASE_SCHEMA", "public")
    return persistence.config_from_mapping(mapping)


def _latest_radar_tickers(config: persistence.DatabaseConfig) -> list[str]:
    response = persistence._request(
        config,
        "GET",
        "cak_radar_snapshots",
        params={"select": "ticker,as_of", "order": "as_of.desc", "limit": "1200"},
        timeout=12,
    )
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        return []
    stamps = [pd.to_datetime(row.get("as_of"), errors="coerce", utc=True) for row in rows]
    latest = max((stamp for stamp in stamps if pd.notna(stamp)), default=pd.NaT)
    if pd.isna(latest):
        return []
    tickers: list[str] = []
    for row, stamp in zip(rows, stamps):
        if pd.notna(stamp) and stamp == latest:
            symbol = pc.normalize_ticker(row.get("ticker"))
            if symbol and symbol not in tickers:
                tickers.append(symbol)
    return tickers


def _upsert_cache(config: persistence.DatabaseConfig, rows: list[dict[str, Any]]) -> int:
    written = 0
    for start in range(0, len(rows), 150):
        batch = rows[start:start + 150]
        response = persistence._request(
            config,
            "POST",
            "cak_source_cache",
            params={"on_conflict": "cache_key"},
            payload=batch,
            timeout=25,
            return_rows=True,
        )
        payload = response.json()
        written += len(payload) if isinstance(payload, list) else len(batch)
    return written


def _safe_local(row: Mapping[str, Any] | None, *, official: bool, now: Any) -> bool:
    if not isinstance(row, Mapping) or not pc._row_hash_valid(row) or not pc._source_row_fresh(row, now):
        return False
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    if official:
        return bool(payload.get("idx_official_source_verified"))
    return bool(pc._fundamental_payload_compatible(payload) and not pc._fundamental_payload_reporting_lagged(payload, now))


def main() -> int:
    now = pd.Timestamp.now(tz="UTC")
    local = _local_config()
    if not local.ready:
        raise SystemExit("EMIR operational Supabase credentials are unavailable")
    tickers = _latest_radar_tickers(local)
    if not tickers:
        raise SystemExit("No latest EMIR radar tickers found")

    shared = SharedFundamentalRuntime("EMIR")
    bundle, shared_meta = shared.read_bundle(tickers)
    if not bundle:
        raise SystemExit(f"Shared Hub returned no bundle: {shared_meta}")

    local_f = pc.read_source_cache(local, tickers, "FUNDAMENTAL")
    local_o = pc.read_source_cache(local, tickers, "IDX_FUNDAMENTAL")
    writes: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for symbol in tickers:
        bare = symbol[:-3] if symbol.endswith(".JK") else symbol
        item = bundle.get(bare)
        if not isinstance(item, Mapping):
            stats["shared_missing"] += 1
            continue

        if _safe_local(local_f.get(symbol), official=False, now=now):
            stats["fundamental_preserved_fresh"] += 1
        else:
            payload = _proxy_payload(item)
            if payload and _shared_good(payload, pc, now):
                payload["ticker"] = symbol
                writes.append(pc.build_source_cache_row(
                    symbol,
                    "FUNDAMENTAL",
                    payload,
                    provider="SHARED_EVIDENCE_HUB_MIRROR",
                    status="OK",
                    checked_at=now,
                    ttl_hours=pc.FUNDAMENTAL_CACHE_TTL_HOURS,
                    latest_observed_at=payload.get("fundamental_observed_at"),
                    last_scan_id="PHASE5_6_SHARED_MIRROR",
                ))
                stats["fundamental_mirrored"] += 1
            else:
                stats["fundamental_shared_insufficient"] += 1

        if _safe_local(local_o.get(symbol), official=True, now=now):
            stats["official_preserved_fresh"] += 1
        else:
            payload = _official_payload(item)
            if payload and _official_good(payload, pc, now):
                payload["ticker"] = symbol
                writes.append(pc.build_source_cache_row(
                    symbol,
                    "IDX_FUNDAMENTAL",
                    payload,
                    provider="SHARED_EVIDENCE_HUB_OFFICIAL_MIRROR",
                    status="OK",
                    checked_at=now,
                    ttl_hours=pc.IDX_OFFICIAL_FUNDAMENTAL_TTL_HOURS,
                    latest_observed_at=payload.get("idx_official_observed_at"),
                    last_scan_id="PHASE5_6_SHARED_MIRROR",
                ))
                stats["official_mirrored"] += 1
            else:
                stats["official_shared_insufficient"] += 1

    written = _upsert_cache(local, writes) if writes else 0
    summary = {
        "latest_radar_tickers": len(tickers),
        "shared_bundle_tickers": len(bundle),
        "cache_rows_prepared": len(writes),
        "cache_rows_written": written,
        "stats": dict(stats),
        "shared_state": shared_meta.get("state"),
        "policy": "FACTS_ONLY_NATIVE_CACHE_MIRROR_NO_SCORE_RANK_GATE_FUTURE_FUNDAMENTAL",
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    print("EMIR_SHARED_MIRROR=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
