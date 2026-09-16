from __future__ import annotations

import json

import pandas as pd

import database_only_runtime as runtime
from persistence import DatabaseConfig


class _Response:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _config() -> DatabaseConfig:
    return DatabaseConfig(True, "https://example.supabase.co", "secret", key_type="SECRET")


def test_snapshot_is_database_only_and_ready(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if url.endswith("cak_idx_database_health_v2"):
            return _Response({"latest_market_date": "2026-09-14", "latest_rank_date": "2026-09-14"})
        if url.endswith("cak_load_latest_idx_rank_v2"):
            return _Response([{"ticker": "DMAS", "overall_rank": 1}])
        if url.endswith("cak_load_latest_idx_top3_v3"):
            return _Response([
                {"ticker": value, "ownership": {"public_pct": 20}, "recent_events": []}
                for value in ("DMAS", "TAPG", "MIKA")
            ])
        if url.endswith("cak_idx_database_gap_report_v1"):
            return _Response({"summary": {"selected_tickers": 1}, "rows": []})
        raise AssertionError(url)

    monkeypatch.setattr(runtime.requests, "post", fake_post)
    snapshot = runtime.load_database_only_snapshot(_config())

    assert snapshot.state == "DATABASE_ONLY_DEEP_900_READY"
    assert snapshot.ranking["ticker"].tolist() == ["DMAS"]
    assert snapshot.top3["ticker"].tolist() == ["DMAS", "TAPG", "MIKA"]
    assert snapshot.top3["public_pct"].tolist() == [20, 20, 20]
    assert snapshot.top3["recent_event_count"].tolist() == [0, 0, 0]
    assert all("supabase.co/rest/v1/rpc/" in url for url in calls)


def test_snapshot_fails_closed_when_top3_is_incomplete(monkeypatch):
    def fake_post(url, **kwargs):
        if url.endswith("cak_idx_database_health_v2"):
            return _Response({"latest_market_date": "2026-09-14", "latest_rank_date": "2026-09-14"})
        if url.endswith("cak_idx_database_gap_report_v1"):
            return _Response({"summary": {"selected_tickers": 0}, "rows": []})
        return _Response([])

    monkeypatch.setattr(runtime.requests, "post", fake_post)
    snapshot = runtime.load_database_only_snapshot(_config())
    assert snapshot.state == "DATABASE_SOURCE_NOT_READY"


def test_snapshot_surfaces_missing_database_evidence(monkeypatch):
    def fake_post(url, **kwargs):
        if url.endswith("cak_idx_database_health_v2"):
            return _Response({"latest_market_date": "2026-09-15", "latest_rank_date": "2026-09-15"})
        if url.endswith("cak_load_latest_idx_rank_v2"):
            return _Response([{"ticker": "ABCD"}])
        if url.endswith("cak_load_latest_idx_top3_v3"):
            return _Response([{"ticker": value} for value in ("A", "B", "C")])
        if url.endswith("cak_idx_database_gap_report_v1"):
            return _Response({
                "summary": {"selected_tickers": 1, "critical_gap_tickers": 1},
                "rows": [{"ticker": "ABCD", "missing_required": ["OFFICIAL_FUNDAMENTAL_METRICS"]}],
            })
        raise AssertionError(url)

    monkeypatch.setattr(runtime.requests, "post", fake_post)
    snapshot = runtime.load_database_only_snapshot(_config())
    assert snapshot.gap_summary["critical_gap_tickers"] == 1
    assert snapshot.evidence_gaps.iloc[0]["ticker"] == "ABCD"


def test_market_panel_normalizes_frames_without_online_fallback(monkeypatch):
    def fake_post(url, **kwargs):
        assert url.endswith("cak_load_idx_market_panel_v1")
        return _Response([
            {"ticker": "MARK", "trade_date": "2026-09-11", "open": "1080", "high": "1120", "low": "1070", "close": "1110", "volume": "10"},
            {"ticker": "MARK", "trade_date": "2026-09-10", "open": "1070", "high": "1090", "low": "1060", "close": "1080", "volume": "8"},
        ])

    monkeypatch.setattr(runtime.requests, "post", fake_post)
    frames = runtime.load_database_market_panel(_config(), ["MARK.JK"])
    assert list(frames) == ["MARK"]
    assert isinstance(frames["MARK"].index, pd.DatetimeIndex)
    assert frames["MARK"]["Close"].tolist() == [1080, 1110]
