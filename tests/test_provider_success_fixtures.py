from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import autonomous_enrichment as ae  # noqa: E402
import data_providers as dp  # noqa: E402


class Response:
    def __init__(self, *, payload=None, text: str = "", status_code: int = 200):
        self._payload = payload
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status_code
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_yahoo_chart_direct_success_fixture(monkeypatch):
    timestamps = [pd.Timestamp("2026-07-30", tz="UTC").timestamp(), pd.Timestamp("2026-07-31", tz="UTC").timestamp()]
    payload = {
        "chart": {
            "result": [{
                "timestamp": timestamps,
                "indicators": {
                    "quote": [{
                        "open": [100, 105], "high": [110, 112], "low": [98, 103],
                        "close": [108, 110], "volume": [1_000_000, 1_200_000],
                    }],
                    "adjclose": [{"adjclose": [54, 55]}],
                },
            }]
        }
    }
    monkeypatch.setattr(dp.requests, "get", lambda *args, **kwargs: Response(payload=payload))
    frame = dp.yahoo_chart_direct("TEST", period="3y", retries=1)
    assert len(frame) == 2
    assert frame.iloc[-1]["Close"] == 55
    assert frame.iloc[-1]["Open"] == 52.5
    assert frame.iloc[-1]["Volume"] == 1_200_000


def test_google_news_rss_success_fixture(monkeypatch):
    rss = """<?xml version='1.0' encoding='UTF-8'?><rss><channel>
    <item><title>TEST ekspansi pabrik - IDX Channel</title><link>https://example.com/a</link>
    <pubDate>Fri, 31 Jul 2026 08:00:00 GMT</pubDate><source>IDX Channel</source></item>
    </channel></rss>"""
    monkeypatch.setattr(dp.requests, "get", lambda *args, **kwargs: Response(text=rss))
    items, audit = dp.fetch_google_news_rss("TEST", "Test Industri", limit=3)
    assert audit["status"] == "OK"
    assert len(items) == 1
    assert items[0]["ticker"] == "TEST.JK"
    assert items[0]["title"].startswith("TEST ekspansi")


def test_ksei_profile_fetch_success_fixture(monkeypatch):
    html = """<html><body><h2>Services</h2><h1>TEST INDUSTRI Tbk, PT</h1>
    <div>Security name</div><div>TEST INDUSTRI Tbk</div><div>Issuer</div><div>TEST INDUSTRI Tbk, PT</div>
    <div>Listing Date</div><div>May 31, 2020</div><div>Status</div><div>Active</div>
    <div>Current Amount</div><div>1,000,000,000.00</div><div>Activity Sector</div><div>ENERGY</div>
    <div>Number of Securities</div><div>1,000,000,000 (Total)</div>
    <div>As of 31 Jul 2026</div><div>42.55% Scripless = 425,500,000</div>
    <div>Local Percentage</div><div>26.48%</div><div>Foreign Percentage</div><div>58.62%</div>
    <table><tr><td>Type of CA</td><td>Ratio</td><td>Cum Date</td><td>Record Date</td><td>Distribution Date</td><td>Status</td></tr>
    <tr><td>Cash Dividend</td><td>1 TEST : 20 IDR</td><td></td><td>18 Jun 2026</td><td>26 Jun 2026</td><td>Active</td></tr></table>
    </body></html>"""
    monkeypatch.setattr(ae.requests, "get", lambda *args, **kwargs: Response(text=html))
    profile, actions, audit = ae.fetch_ksei_profile("TEST")
    assert audit["status"] == "OK"
    assert profile["company_name"] == "TEST INDUSTRI Tbk, PT"
    assert profile["sector"] == "ENERGY"
    assert len(actions) == 1


def test_fundamental_statement_columns_are_latest_first(monkeypatch):
    # Fixture intentionally supplies oldest column first; collector must use the latest quarter.
    columns = [pd.Timestamp("2026-03-31"), pd.Timestamp("2026-06-30")]
    income = pd.DataFrame([[1000, 1300], [80, 150]], index=["Total Revenue", "Net Income"], columns=columns)
    balance = pd.DataFrame([[700, 800], [350, 300]], index=["Stockholders Equity", "Total Debt"], columns=columns)
    cashflow = pd.DataFrame([[110, 180], [-30, -40]], index=["Operating Cash Flow", "Capital Expenditure"], columns=columns)
    fake = SimpleNamespace(quarterly_income_stmt=income, quarterly_balance_sheet=balance, quarterly_cashflow=cashflow)
    monkeypatch.setattr(ae, "yf", SimpleNamespace(Ticker=lambda symbol: fake))
    snapshot, audit = ae.fetch_yfinance_fundamental_snapshot("TEST")
    assert audit["status"] == "OK"
    assert snapshot["revenue_latest"] == 1300
    assert snapshot["revenue_growth_pct"] == 30.0
