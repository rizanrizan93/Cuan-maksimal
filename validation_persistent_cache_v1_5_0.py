from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import persistent_cache as pc
from data_providers import FetchResult
from persistence import DatabaseConfig


def synthetic_frame(seed: int, bars: int = 760, end: str = "2026-08-03") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=end, periods=bars)
    close = 500 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, bars)))
    open_ = close * (1 + rng.normal(0, 0.002, bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, bars))
    volume = rng.integers(500_000, 8_000_000, bars).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)


def main() -> None:
    tickers = [f"T{i:03d}.JK" for i in range(400)]
    frames = {ticker: synthetic_frame(i + 1) for i, ticker in enumerate(tickers)}
    provider_audit = pd.DataFrame([
        {"ticker": ticker, "provider": "FIXTURE_PROVIDER", "status": "OK", "detail": ""}
        for ticker in tickers
    ])
    config = DatabaseConfig(True, "https://fixture.supabase.co", "sb_secret_fixture", key_type="SECRET")
    original_read = pc.read_ohlcv_cache
    original_fetch = pc.fetch_many_ohlcv
    original_window = pc.fetch_ohlcv_window
    original_post = pc._post_payload_in_chunks
    original_get = pc._get_in_chunks
    try:
        cold_provider_calls = {"count": 0}
        pc.read_ohlcv_cache = lambda *_: {}
        def cold_fetch(*args, **kwargs):
            cold_provider_calls["count"] += len(tickers)
            return frames, provider_audit
        pc.fetch_many_ohlcv = cold_fetch
        start = time.perf_counter()
        cold_frames, cold_audit, cache_rows = pc.fetch_ohlcv_cache_first(
            config, tickers, now="2026-08-03T18:00:00+07:00", completed_only=False, last_scan_id="cold"
        )
        cold_elapsed = time.perf_counter() - start

        cache_map = {row["ticker"]: row for row in cache_rows}
        payload_bytes = sum(len(json.dumps(row["payload"], separators=(",", ":"))) for row in cache_rows)

        warm_provider_calls = {"count": 0}
        pc.read_ohlcv_cache = lambda *_: cache_map
        def forbidden_fetch(*args, **kwargs):
            warm_provider_calls["count"] += 1
            raise AssertionError("Warm cache unexpectedly called full provider")
        pc.fetch_many_ohlcv = forbidden_fetch
        start = time.perf_counter()
        warm_frames, warm_audit, warm_writes = pc.fetch_ohlcv_cache_first(
            config, tickers, now="2026-08-03T20:00:00+07:00", completed_only=False, last_scan_id="warm"
        )
        warm_elapsed = time.perf_counter() - start

        equivalence = all(
            len(cold_frames[ticker]) == len(warm_frames[ticker])
            and np.isclose(cold_frames[ticker]["Close"].iloc[-1], warm_frames[ticker]["Close"].iloc[-1])
            for ticker in tickers
        )

        stale_map = {ticker: dict(row, checked_at="2026-08-01T00:00:00Z") for ticker, row in list(cache_map.items())[:40]}
        pc.read_ohlcv_cache = lambda *_: stale_map
        incremental_calls = {"count": 0}
        def incremental_fetch(ticker, **kwargs):
            incremental_calls["count"] += 1
            old = frames[ticker]
            next_date = old.index[-1] + pd.offsets.BDay(1)
            last = old.iloc[-1]
            tail = pd.DataFrame({
                "Open": [float(last["Close"])], "High": [float(last["Close"]) * 1.01],
                "Low": [float(last["Close"]) * 0.99], "Close": [float(last["Close"]) * 1.005],
                "Volume": [float(last["Volume"])],
            }, index=[next_date])
            return FetchResult(ticker, tail, "FIXTURE_INCREMENTAL", "OK")
        pc.fetch_ohlcv_window = incremental_fetch
        incremental_frames, incremental_audit, incremental_writes = pc.fetch_ohlcv_cache_first(
            config, list(stale_map), now="2026-08-05T18:00:00+07:00", completed_only=False, last_scan_id="incremental"
        )

        memory_store = {
            "cak_ohlcv_cache": {},
            "cak_source_cache": {},
        }
        def fake_post(config, table, conflict, payload, chunk_size, return_rows=True):
            for row in payload:
                memory_store[table][row[conflict]] = row
            return len(payload)
        def fake_get(config, table, key, values, select="*"):
            return [memory_store[table][value] for value in values if value in memory_store[table]]
        pc._post_payload_in_chunks = fake_post
        pc._get_in_chunks = fake_get
        cache_write, cache_verify = pc.persist_verify_cache_bundle(
            config, scan_id="cache-validation", ohlcv_rows=cache_rows, source_rows=[]
        )

        result = {
            "scanner_version": "1.5.0-persistent-cache-incremental-refresh",
            "tickers": 400,
            "bars_per_ticker": 760,
            "cold_cache_rows_created": len(cache_rows),
            "cold_provider_ticker_calls": cold_provider_calls["count"],
            "cold_elapsed_cpu_seconds_network_excluded": round(cold_elapsed, 3),
            "warm_cache_hits": int(warm_audit["status"].eq("CACHE_HIT").sum()),
            "warm_provider_calls": warm_provider_calls["count"],
            "warm_cache_writes": len(warm_writes),
            "warm_elapsed_cpu_seconds_network_excluded": round(warm_elapsed, 3),
            "cold_warm_data_equivalent": equivalence,
            "compact_ohlcv_payload_mb": round(payload_bytes / 1024 / 1024, 2),
            "incremental_tickers": len(stale_map),
            "incremental_provider_calls": incremental_calls["count"],
            "incremental_refresh_rows": len(incremental_writes),
            "incremental_all_added_one_bar": all(len(incremental_frames[ticker]) == 761 for ticker in stale_map),
            "cache_write_state": str(cache_write.iloc[0]["state"]),
            "cache_readback_state": str(cache_verify.iloc[0]["state"]),
            "cache_hash_verification_pct": 100.0 if pc.cache_commit_succeeded(cache_verify) else 0.0,
            "assertions": {
                "400_warm_hits": int(warm_audit["status"].eq("CACHE_HIT").sum()) == 400,
                "zero_warm_provider_calls": warm_provider_calls["count"] == 0,
                "cold_warm_equivalent": equivalence,
                "incremental_tail_only": incremental_calls["count"] == 40 and len(incremental_writes) == 40,
                "cache_exact_hash_readback": pc.cache_commit_succeeded(cache_verify),
            },
            "timing_note": "CPU times exclude internet latency. The validated speed mechanism is provider-call elimination on warm scans and bounded tail refresh on stale OHLCV.",
        }
        assert all(result["assertions"].values()), result
        path = Path("validation_artifacts/VALIDATION_PERSISTENT_CACHE_V1_5_0.json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        pc.read_ohlcv_cache = original_read
        pc.fetch_many_ohlcv = original_fetch
        pc.fetch_ohlcv_window = original_window
        pc._post_payload_in_chunks = original_post
        pc._get_in_chunks = original_get


if __name__ == "__main__":
    main()
