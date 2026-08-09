from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_providers as dp  # noqa: E402
import persistence as ps  # noqa: E402


def frame(n: int = 260) -> pd.DataFrame:
    idx = pd.bdate_range("2025-08-01", periods=n)
    close = np.linspace(500, 700, n)
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 2_000_000.0),
        },
        index=idx,
    )


def test_429_direct_falls_back_to_multiindex_yfinance(monkeypatch):
    def fail_direct(*args, **kwargs):
        raise RuntimeError("429 Too Many Requests")

    base = frame()
    multi = base.copy()
    multi.columns = pd.MultiIndex.from_tuples([(col, "TEST.JK") for col in base.columns])
    fake_yf = SimpleNamespace(download=lambda *args, **kwargs: multi)
    monkeypatch.setattr(dp, "yahoo_chart_direct", fail_direct)
    monkeypatch.setattr(dp, "yf", fake_yf)

    result = dp.fetch_ohlcv("TEST", completed_only=False)
    assert result.status == "OK"
    assert result.provider == "YFINANCE"
    assert len(result.frame) == len(base)
    assert "429" in result.detail


def test_both_ohlcv_providers_fail_closed(monkeypatch):
    monkeypatch.setattr(dp, "yahoo_chart_direct", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("429")))
    monkeypatch.setattr(dp, "yf", SimpleNamespace(download=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fallback down"))))
    monkeypatch.setattr(dp, "ksei_price_history", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ksei down")))
    result = dp.fetch_ohlcv("TEST", completed_only=False)
    assert result.status == "ERROR"
    assert result.provider == "NONE"
    assert result.frame.empty
    assert "fallback down" in result.detail
    assert "ksei down" in result.detail




KSEI_PRICE_HTML = """
<html><body>
<table><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th><th>Value</th><th>Freq</th></tr>
<tr><td>31 Jul 2026</td><td>8,000</td><td>8,100</td><td>7,950</td><td>8,050</td><td>1,234</td><td>993,000,000</td><td>500</td></tr>
<tr><td>30 Jul 2026</td><td>7,900</td><td>8,000</td><td>7,850</td><td>7,950</td><td>2,000</td><td>1,590,000,000</td><td>700</td></tr>
</table>
</body></html>
"""


def test_ksei_price_parser_converts_lots_to_shares():
    parsed = dp.parse_ksei_price_history_html(KSEI_PRICE_HTML)
    assert len(parsed) == 2
    assert parsed.loc[pd.Timestamp("2026-07-31"), "Volume"] == 123_400
    assert parsed.loc[pd.Timestamp("2026-07-31"), "Close"] == 8050


def test_fetch_ohlcv_uses_ksei_as_third_provider(monkeypatch):
    monkeypatch.setattr(dp, "yahoo_chart_direct", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("429")))
    monkeypatch.setattr(dp, "yf", SimpleNamespace(download=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("yfinance down"))))
    monkeypatch.setattr(dp, "ksei_price_history", lambda *args, **kwargs: frame())
    result = dp.fetch_ohlcv("TEST", completed_only=False)
    assert result.status == "OK"
    assert result.provider == "KSEI_PRICE_HISTORY"
    assert len(result.frame) == 260
    assert "yfinance down" in result.detail


def test_fetch_many_marks_provider_failure_without_crashing(monkeypatch):
    def fake_fetch(ticker, *args, **kwargs):
        symbol = dp.normalize_ticker(ticker)
        if symbol.startswith("BAD"):
            return dp.FetchResult(symbol, pd.DataFrame(), "NONE", "ERROR", "injected")
        return dp.FetchResult(symbol, frame(), "FIXTURE", "OK", "")

    monkeypatch.setattr(dp, "fetch_ohlcv", fake_fetch)
    frames, audit = dp.fetch_many_ohlcv(["GOOD", "BAD"], completed_only=False, max_workers=2)
    assert "GOOD.JK" in frames
    assert "BAD.JK" not in frames
    bad = audit.loc[audit["ticker"].eq("BAD.JK")].iloc[0]
    assert bad["quality_state"] == "PROVIDER_FAILED"


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


def ready_config() -> ps.DatabaseConfig:
    return ps.config_from_mapping(
        {
            "CAK_DATABASE_ENABLED": "true",
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SECRET_KEY": "sb_secret_test",
        }
    )


def minimal_radar() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA.JK",
                "emir_decision_state": "EMIR_EVIDENCE_PENDING",
                "action": "COMPLETE_DIRECT_EVIDENCE",
                "emir_conviction_score": 50.0,
                "emir_evidence_coverage_pct": 40.0,
                "production_ready": False,
            }
        ]
    )


def test_database_all_tables_write_and_readback(monkeypatch):
    stored: dict[str, list[dict]] = {}

    def fake_request(config, method, table, *, params=None, payload=None, **kwargs):
        if method == "POST":
            incoming = list(payload or [])
            existing = stored.setdefault(table, [])
            # Test fixture only needs deterministic upsert for the supplied primary keys.
            existing.extend(incoming)
            return FakeResponse(incoming)
        scan_id = str((params or {}).get("scan_id", "eq.")).removeprefix("eq.")
        if method == "PATCH":
            updated = []
            for row in stored.get(table, []):
                if row.get("scan_id") == scan_id:
                    row.update(dict(payload or {}))
                    updated.append(dict(row))
            return FakeResponse(updated)
        rows = [row for row in stored.get(table, []) if row.get("scan_id") == scan_id]
        return FakeResponse(rows)

    monkeypatch.setattr(ps, "_request", fake_request)
    outcomes = pd.DataFrame(
        [
            {
                "ticker": "AAA.JK",
                "signal_date": "2026-08-01",
                "horizon_days": 20,
                "outcome_verified": True,
                "return_pct": 5.0,
            }
        ]
    )
    report = ps.persist_scan(
        ready_config(),
        scan_id="scan-x",
        as_of="2026-08-03",
        radar=minimal_radar(),
        events=pd.DataFrame([{
            "ticker": "AAA.JK", "published_at": "2026-08-02", "title": "Verified event",
            "publisher": "Issuer", "url": "https://issuer.example/event",
        }]),
        provider_audit=pd.DataFrame([{"ticker": "AAA.JK", "provider": "FIXTURE", "status": "OK"}]),
        direct_evidence=pd.DataFrame([{
            "ticker": "AAA.JK", "evidence_type": "IDX_INTEGRITY_REGULATORY",
            "observed_at": "2026-08-02", "source_verified": True,
        }]),
        autonomous_evidence=pd.DataFrame([{
            "ticker": "AAA.JK", "evidence_type": "BROKER_INVENTORY_OHLCV_PROXY",
            "observed_at": "2026-08-02", "source_verified": False,
        }]),
        outcomes=outcomes,
        mode="EMIR_DEEP_REVIEW",
    )
    assert report.iloc[0]["state"] == "WRITE_ALL_TABLES"
    assert set(stored) == {
        "cak_scan_runs",
        "cak_radar_snapshots",
        "cak_narrative_events",
        "cak_provider_audit",
        "cak_direct_evidence",
        "cak_autonomous_evidence",
        "cak_outcome_memory",
    }
    verification = ps.verify_scan(
        ready_config(),
        scan_id="scan-x",
        expected_radar=1,
        expected_events=1,
        expected_provider_audit=1,
        expected_direct_evidence=1,
        expected_autonomous_evidence=1,
        expected_outcomes=1,
    )
    assert verification.iloc[0]["state"] == "VERIFIED_ALL_TABLES"
    assert verification.iloc[0]["verification_pct"] == 100.0


def test_database_partial_write_is_visible(monkeypatch):
    def fake_request(config, method, table, *, params=None, payload=None, **kwargs):
        if method == "POST" and table == "cak_radar_snapshots":
            raise RuntimeError("injected RLS failure")
        return FakeResponse(list(payload or []))

    monkeypatch.setattr(ps, "_request", fake_request)
    report = ps.persist_scan(
        ready_config(),
        scan_id="scan-y",
        as_of="2026-08-03",
        radar=minimal_radar(),
        events=pd.DataFrame(),
        provider_audit=pd.DataFrame(),
        direct_evidence=pd.DataFrame(),
        outcomes=pd.DataFrame(),
        mode="EMIR_DEEP_REVIEW",
    )
    assert report.iloc[0]["state"] == "WRITE_PARTIAL"
    failed = report.loc[report["table"].eq("cak_radar_snapshots")].iloc[0]
    assert failed["state"] == "WRITE_FAILED"
    assert "RLS" in failed["detail"]


def test_yahoo_direct_alternates_query_hosts_after_rate_limit(monkeypatch):
    calls = []

    class Response:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self.headers = {}
            self._payload = payload or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"{self.status_code} error")

        def json(self):
            return self._payload

    payload = {
        "chart": {
            "result": [{
                "timestamp": [1_700_000_000, 1_700_086_400],
                "indicators": {
                    "quote": [{
                        "open": [100, 101], "high": [102, 103], "low": [99, 100],
                        "close": [101, 102], "volume": [1_000_000, 1_100_000],
                    }],
                    "adjclose": [{"adjclose": [101, 102]}],
                },
            }]
        }
    }

    def fake_get(url, **kwargs):
        calls.append(url)
        return Response(429) if len(calls) == 1 else Response(200, payload)

    monkeypatch.setattr(dp.requests, "get", fake_get)
    monkeypatch.setattr(dp.time, "sleep", lambda *_: None)
    result = dp.yahoo_chart_direct("^JKSE", retries=2)
    assert not result.empty
    assert "query1.finance.yahoo.com" in calls[0]
    assert "query2.finance.yahoo.com" in calls[1]


def test_benchmark_stale_relative_to_universe_uses_proxy_gate():
    from data_providers import assess_benchmark_freshness
    benchmark = frame(n=259)
    frames = {f"T{i}.JK": frame(n=260) for i in range(25)}
    result = assess_benchmark_freshness(benchmark, frames, min_universe_count=20)
    assert result["benchmark_usable"] is False
    assert result["benchmark_freshness_state"] == "STALE_RELATIVE_TO_UNIVERSE"
    assert result["benchmark_business_lag_days"] >= 1
