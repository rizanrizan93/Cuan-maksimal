from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import persistent_cache as pc  # noqa: E402
from data_providers import FetchResult  # noqa: E402
from persistence import DatabaseConfig  # noqa: E402


def frame(start: str = "2026-07-01", periods: int = 20) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({
        "Open": range(100, 100 + periods),
        "High": range(102, 102 + periods),
        "Low": range(99, 99 + periods),
        "Close": range(101, 101 + periods),
        "Volume": [1_000_000 + i * 1000 for i in range(periods)],
    }, index=idx)


def config() -> DatabaseConfig:
    return DatabaseConfig(True, "https://x.supabase.co", "sb_secret_test", key_type="SECRET")


def test_ohlcv_payload_roundtrip_is_finite_and_sorted():
    original = frame(periods=12)
    payload = pc.frame_to_payload(original)
    restored = pc.payload_to_frame(payload)
    assert len(restored) == 12
    assert restored.index.is_monotonic_increasing
    assert restored.iloc[-1]["Close"] == original.iloc[-1]["Close"]


def test_fresh_ohlcv_cache_hit_avoids_provider(monkeypatch):
    cached_frame = frame("2026-07-15", 15)
    row = pc.build_ohlcv_cache_row(
        "ADMR.JK", cached_frame, period="5y", provider="YAHOO", checked_at="2026-08-05T12:00:00Z"
    )
    monkeypatch.setattr(pc, "read_ohlcv_cache", lambda *_: {"ADMR.JK": row})
    monkeypatch.setattr(pc, "fetch_many_ohlcv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("provider called")))
    frames, audit, writes = pc.fetch_ohlcv_cache_first(
        config(), ["ADMR"], now="2026-08-05T13:00:00Z", completed_only=False
    )
    assert len(frames["ADMR.JK"]) == 15
    assert audit.iloc[0]["status"] == "CACHE_HIT"
    assert writes == []


def test_stale_ohlcv_cache_incremental_merge(monkeypatch):
    cached_frame = frame("2026-06-01", 20)
    row = pc.build_ohlcv_cache_row(
        "ADMR.JK", cached_frame, period="5y", provider="YAHOO", checked_at="2026-06-30T00:00:00Z"
    )
    tail = frame(cached_frame.index[-3].date().isoformat(), 8)
    monkeypatch.setattr(pc, "read_ohlcv_cache", lambda *_: {"ADMR.JK": row})
    monkeypatch.setattr(pc, "fetch_ohlcv_window", lambda *a, **k: FetchResult("ADMR.JK", tail, "YAHOO_CHART_INCREMENTAL", "OK"))
    frames, audit, writes = pc.fetch_ohlcv_cache_first(
        config(), ["ADMR"], now="2026-08-05T18:00:00+07:00", completed_only=False
    )
    assert len(frames["ADMR.JK"]) > len(cached_frame)
    assert audit.iloc[0]["status"] == "INCREMENTAL_REFRESH"
    assert len(writes) == 1
    assert writes[0]["last_session_date"] == frames["ADMR.JK"].index.max().date().isoformat()


def test_stale_ohlcv_provider_failure_uses_cache(monkeypatch):
    cached_frame = frame("2026-07-10", 15)
    row = pc.build_ohlcv_cache_row(
        "ADMR.JK", cached_frame, period="5y", provider="YAHOO", checked_at="2026-07-20T00:00:00Z"
    )
    monkeypatch.setattr(pc, "read_ohlcv_cache", lambda *_: {"ADMR.JK": row})
    monkeypatch.setattr(pc, "fetch_ohlcv_window", lambda *a, **k: FetchResult("ADMR.JK", pd.DataFrame(), "NONE", "ERROR", "429"))
    frames, audit, writes = pc.fetch_ohlcv_cache_first(
        config(), ["ADMR"], now="2026-08-05T18:00:00+07:00", completed_only=False
    )
    assert not frames["ADMR.JK"].empty
    assert audit.iloc[0]["status"] == "STALE_CACHE_FALLBACK"
    assert writes == []


def test_fundamental_cache_hit_avoids_provider(monkeypatch):
    row = pc.build_source_cache_row(
        "ELSA.JK", "FUNDAMENTAL", {
            "ticker": "ELSA.JK", "fundamental_cache_schema_version": "4", "fundamental_coverage_pct": 70.0,
            "revenue_growth_qoq_pct": 5.0, "revenue_growth_yoy_pct": 9.0,
            "earnings_growth_qoq_pct": 8.0, "earnings_growth_yoy_pct": 29.0,
            "roe_ttm_pct": 12.0, "roa_ttm_pct": 7.0,
            "interest_bearing_debt_to_equity": 0.2, "total_liabilities_to_equity": 0.5,
            "net_debt_to_equity": -0.1, "current_ratio": 1.5, "cash_to_debt_ratio": 2.0,
                "fundamental_period_alignment_state": "ALIGNED", "fundamental_cashflow_state": "OCF_FCF_TTM_AVAILABLE",
                "fundamental_data_quality_score": 80.0, "fundamental_score_cap": 88.0,
                "fundamental_growth_consistency_state": "QUARTER_AND_YTD_CONFIRMED", "fundamental_growth_consistency_score": 100.0,
                "revenue_growth_ytd_yoy_pct": 9.0, "earnings_growth_ytd_yoy_pct": 29.0,
        },
        provider="YFINANCE", status="OK", checked_at="2026-08-05T10:00:00Z", ttl_hours=168,
    )
    monkeypatch.setattr(pc, "read_source_cache", lambda *_: {"ELSA.JK": row})
    monkeypatch.setattr(pc, "fetch_many_fundamentals", lambda *a, **k: (_ for _ in ()).throw(AssertionError("provider called")))
    snapshots, audit, writes = pc.fetch_fundamental_cache_first(
        config(), ["ELSA"], now="2026-08-05T12:00:00Z"
    )
    assert snapshots.iloc[0]["fundamental_coverage_pct"] == 70.0
    assert audit.iloc[0]["status"] == "CACHE_HIT"
    assert writes == []


def test_cache_bundle_requires_exact_hash_readback(monkeypatch):
    ohlcv = pc.build_ohlcv_cache_row("ADMR", frame(), period="5y", provider="YAHOO", checked_at="2026-08-05T00:00:00Z")
    source = pc.build_source_cache_row("ADMR", "NEWS", [], provider="NEWS", status="NO_ITEMS", checked_at="2026-08-05T00:00:00Z", ttl_hours=2)
    monkeypatch.setattr(pc, "_post_payload_in_chunks", lambda config, table, conflict, payload, chunk_size, return_rows=True: len(payload))

    def exact_get(config, table, key, values, select="*"):
        rows = [ohlcv] if table == "cak_ohlcv_cache" else [source]
        return [{key: row[key], "content_sha256": row["content_sha256"]} for row in rows]

    monkeypatch.setattr(pc, "_get_in_chunks", exact_get)
    write, verify = pc.persist_verify_cache_bundle(config(), scan_id="scan", ohlcv_rows=[ohlcv], source_rows=[source])
    assert write.iloc[0]["state"] == "CACHE_WRITE_ALL"
    assert verify.iloc[0]["state"] == "CACHE_DATABASE_COMMITTED"
    assert pc.cache_commit_succeeded(verify)


def test_cache_bundle_rejects_hash_mismatch(monkeypatch):
    ohlcv = pc.build_ohlcv_cache_row("ADMR", frame(), period="5y", provider="YAHOO", checked_at="2026-08-05T00:00:00Z")
    monkeypatch.setattr(pc, "_post_payload_in_chunks", lambda config, table, conflict, payload, chunk_size, return_rows=True: len(payload))
    monkeypatch.setattr(pc, "_get_in_chunks", lambda *a, **k: [{"ticker": "ADMR.JK", "content_sha256": "wrong"}] if a[1] == "cak_ohlcv_cache" else [])
    _, verify = pc.persist_verify_cache_bundle(config(), scan_id="scan", ohlcv_rows=[ohlcv], source_rows=[])
    assert verify.iloc[0]["state"] == "CACHE_DATABASE_NOT_COMMITTED"
    assert not pc.cache_commit_succeeded(verify)


def test_app_contains_resumable_cache_checkpoint_flow():
    source = (ROOT / "app.py").read_text()
    assert "resumable chunked scan" in source
    assert "Ticker per checkpoint" in source
    assert "KSEI untuk target deep review" in source
    assert "process_next_job_step" in source
    assert "CACHE_NOT_COMMITTED" not in source


def test_corrupt_cache_hash_forces_cold_refresh(monkeypatch):
    cached_frame = frame("2026-07-15", 15)
    row = pc.build_ohlcv_cache_row(
        "ADMR.JK", cached_frame, period="5y", provider="YAHOO", checked_at="2026-08-05T12:00:00Z"
    )
    row["content_sha256"] = "corrupt"
    fresh = frame("2026-07-20", 15)
    monkeypatch.setattr(pc, "read_ohlcv_cache", lambda *_: {"ADMR.JK": row})
    monkeypatch.setattr(
        pc,
        "fetch_many_ohlcv",
        lambda *a, **k: (
            {"ADMR.JK": fresh},
            pd.DataFrame([{"ticker": "ADMR.JK", "provider": "YAHOO_CHART_DIRECT", "status": "OK", "detail": ""}]),
        ),
    )
    frames, audit, writes = pc.fetch_ohlcv_cache_first(
        config(), ["ADMR"], now="2026-08-05T13:00:00Z", completed_only=False
    )
    assert audit.iloc[0]["status"] == "COLD_REFRESH"
    assert len(frames["ADMR.JK"]) == len(fresh)
    assert len(writes) == 1


def test_hash_is_stable_across_jsonb_integer_float_roundtrip():
    assert pc._hash_payload([["2026-08-03", 1, 2.0, 3, 4.0, 1000000]]) == pc._hash_payload([["2026-08-03", 1.0, 2, 3.0, 4, 1000000.0]])


def test_cache_read_error_becomes_cache_miss(monkeypatch):
    monkeypatch.setattr(pc, "_get_in_chunks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db unavailable")))
    assert pc.read_ohlcv_cache(config(), ["ADMR"]) == {}
    assert pc.read_source_cache(config(), ["ADMR"], "NEWS") == {}


def test_mixed_cache_fetches_only_missing_ticker(monkeypatch):
    cached_frame = frame("2026-07-15", 15)
    row = pc.build_ohlcv_cache_row(
        "ADMR.JK", cached_frame, period="5y", provider="YAHOO", checked_at="2026-08-05T12:00:00Z"
    )
    monkeypatch.setattr(pc, "read_ohlcv_cache", lambda *_: {"ADMR.JK": row})
    calls = []
    fresh = frame("2026-07-20", 15)
    def fetch_missing(symbols, **kwargs):
        calls.extend(symbols)
        return {"ELSA.JK": fresh}, pd.DataFrame([{
            "ticker": "ELSA.JK", "provider": "YAHOO_CHART_DIRECT", "status": "OK", "detail": ""
        }])
    monkeypatch.setattr(pc, "fetch_many_ohlcv", fetch_missing)
    frames, audit, writes = pc.fetch_ohlcv_cache_first(
        config(), ["ADMR", "ELSA"], now="2026-08-05T13:00:00Z", completed_only=False
    )
    assert calls == ["ELSA.JK"]
    assert set(frames) == {"ADMR.JK", "ELSA.JK"}
    assert set(audit["status"]) == {"CACHE_HIT", "COLD_REFRESH"}
    assert len(writes) == 1


def test_cache_database_disabled_does_not_raise():
    disabled = DatabaseConfig(False, "", "")
    write, verify = pc.persist_verify_cache_bundle(
        disabled, scan_id="scan", ohlcv_rows=[{"ticker": "ADMR.JK"}], source_rows=[]
    )
    assert write.iloc[0]["state"] == "CACHE_DATABASE_DISABLED"
    assert verify.iloc[0]["state"] == "CACHE_PERSISTENCE_SKIPPED"
    assert pc.cache_persistence_state(verify) == "CACHE_MEMORY_ONLY"
