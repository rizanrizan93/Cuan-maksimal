from __future__ import annotations

import math

import pandas as pd

from dashboard_price_overlay import apply_current_market_price_overlay


class Backend:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    def read_rows(self, table, filters, *, limit):
        self.calls.append((table, dict(filters), limit))
        return [
            dict(row) for row in self.rows
            if all(str(row.get(key)) == str(value) for key, value in filters.items())
        ][:limit]


class Evidence:
    def __init__(self, rows):
        self.ready = True
        self.backend = Backend(rows)


def _row(last_price=615.0, last_date="2026-08-28"):
    return {
        "ticker": "PSAB.JK",
        "last_price": last_price,
        "last_date": last_date,
        "dashboard_price_change_pct": -4.65,
        "emir_final_score": 68.7,
    }


def _shared(close=565.0, previous=565.0):
    return {
        "provider": "ZAPI",
        "trade_date": "2026-09-02",
        "ticker": "PSAB",
        "close": close,
        "previous": previous,
        "freshness_state": "CURRENT",
        "validation_state": "VALID",
    }


def test_current_factual_price_overlays_stale_technical_display_only():
    original = pd.DataFrame([_row()])
    evidence = Evidence([_shared()])
    out, meta = apply_current_market_price_overlay(
        original, now="2026-09-03T10:00:00+07:00", evidence=evidence
    )
    assert original.iloc[0]["last_price"] == 615.0
    assert out.iloc[0]["last_price"] == 615.0
    assert out.iloc[0]["display_last_price"] == 565.0
    assert out.iloc[0]["display_price_asof"] == "2026-09-02"
    assert out.iloc[0]["display_price_source"] == "SHARED_STOCK_SUMMARY_ZAPI_CACHE"
    assert out.iloc[0]["display_technical_freshness_state"] == "TECHNICAL_STALE_RESCAN_REQUIRED"
    assert meta["provider_calls"] == 0
    assert meta["request_avoided"] is True
    assert meta["technical_stale_rows"] == 1
    assert evidence.backend.calls == [
        ("evidence_market_daily", {"provider": "ZAPI", "trade_date": "2026-09-02"}, 5000)
    ]


def test_current_technical_price_is_usable_when_shared_cache_missing():
    out, meta = apply_current_market_price_overlay(
        pd.DataFrame([_row(last_price=565.0, last_date="2026-09-02")]),
        now="2026-09-03T10:00:00+07:00",
        evidence=Evidence([]),
    )
    assert out.iloc[0]["display_last_price"] == 565.0
    assert out.iloc[0]["display_price_source"] == "CURRENT_TECHNICAL_OHLCV"
    assert out.iloc[0]["display_technical_freshness_state"] == "TECHNICAL_CURRENT"
    assert meta["technical_stale_rows"] == 0


def test_stale_technical_price_is_not_labelled_current_without_factual_cache():
    out, meta = apply_current_market_price_overlay(
        pd.DataFrame([_row()]),
        now="2026-09-03T10:00:00+07:00",
        evidence=Evidence([]),
    )
    assert math.isnan(float(out.iloc[0]["display_last_price"]))
    assert out.iloc[0]["display_price_state"] == "CURRENT_PRICE_UNAVAILABLE"
    assert out.iloc[0]["display_technical_freshness_state"] == "TECHNICAL_STALE_RESCAN_REQUIRED"
    assert meta["technical_stale_rows"] == 1


def test_invalid_shared_row_does_not_override_current_technical_price():
    bad = _shared(close=999.0)
    bad["validation_state"] = "STALE"
    out, _ = apply_current_market_price_overlay(
        pd.DataFrame([_row(last_price=565.0, last_date="2026-09-02")]),
        now="2026-09-03T10:00:00+07:00",
        evidence=Evidence([bad]),
    )
    assert out.iloc[0]["display_last_price"] == 565.0
    assert out.iloc[0]["display_price_source"] == "CURRENT_TECHNICAL_OHLCV"
