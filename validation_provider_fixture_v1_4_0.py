from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import autonomous_enrichment as ae
import data_providers as dp


class Response:
    def __init__(self, *, payload=None, text="", status_code=200):
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


def main() -> None:
    report = {}
    original_dp_get = dp.requests.get
    original_ae_get = ae.requests.get
    original_yf = ae.yf
    try:
        timestamps = [pd.Timestamp("2026-07-30", tz="UTC").timestamp(), pd.Timestamp("2026-07-31", tz="UTC").timestamp()]
        yahoo_payload = {"chart": {"result": [{"timestamp": timestamps, "indicators": {"quote": [{"open": [100,105], "high": [110,112], "low": [98,103], "close": [108,110], "volume": [1_000_000,1_200_000]}], "adjclose": [{"adjclose": [54,55]}]}}]}}
        dp.requests.get = lambda *args, **kwargs: Response(payload=yahoo_payload)
        yahoo = dp.yahoo_chart_direct("TEST", retries=1)
        report["yahoo_chart_fixture"] = {"status": "PASS" if len(yahoo)==2 and yahoo.iloc[-1]["Close"]==55 else "FAIL", "bars": len(yahoo), "last_close": float(yahoo.iloc[-1]["Close"])}

        ksei_html = """<html><body><h2>Services</h2><h1>TEST INDUSTRI Tbk, PT</h1>
        <div>Security name</div><div>TEST INDUSTRI Tbk</div><div>Issuer</div><div>TEST INDUSTRI Tbk, PT</div>
        <div>Listing Date</div><div>May 31, 2020</div><div>Status</div><div>Active</div>
        <div>Current Amount</div><div>1,000,000,000.00</div><div>Activity Sector</div><div>ENERGY</div>
        <div>Number of Securities</div><div>1,000,000,000 (Total)</div>
        <div>As of 31 Jul 2026</div><div>42.55% Scripless = 425,500,000</div><div>Local Percentage</div><div>26.48%</div><div>Foreign Percentage</div><div>58.62%</div>
        <table><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th><th>Value</th><th>Freq</th></tr>
        <tr><td>31 Jul 2026</td><td>8,000</td><td>8,100</td><td>7,950</td><td>8,050</td><td>1,234</td><td>993,000,000</td><td>500</td></tr></table>
        <table><tr><td>Type of CA</td><td>Ratio</td><td>Cum Date</td><td>Record Date</td><td>Distribution Date</td><td>Status</td></tr>
        <tr><td>Cash Dividend</td><td>1 TEST : 20 IDR</td><td></td><td>18 Jun 2026</td><td>26 Jun 2026</td><td>Active</td></tr></table></body></html>"""
        profile, actions = ae.parse_ksei_profile_html("TEST", ksei_html, source_url="https://web.ksei.co.id/test")
        price = dp.parse_ksei_price_history_html(ksei_html)
        report["ksei_public_page_fixture"] = {
            "status": "PASS" if profile.get("company_name")=="TEST INDUSTRI Tbk, PT" and len(actions)==1 and len(price)==1 and price.iloc[0]["Volume"]==123_400 else "FAIL",
            "company_name": profile.get("company_name"), "sector": profile.get("sector"), "corporate_actions": len(actions), "price_rows": len(price), "volume_shares": float(price.iloc[0]["Volume"]),
        }

        rss = """<?xml version='1.0'?><rss><channel><item><title>TEST ekspansi pabrik - Media</title><link>https://example.com/a</link><pubDate>Fri, 31 Jul 2026 08:00:00 GMT</pubDate><source>Media</source></item></channel></rss>"""
        dp.requests.get = lambda *args, **kwargs: Response(text=rss)
        news, news_audit = dp.fetch_google_news_rss("TEST", "Test Industri", limit=2)
        report["google_news_rss_fixture"] = {"status": "PASS" if news_audit["status"]=="OK" and len(news)==1 else "FAIL", "items": len(news)}

        cols=[pd.Timestamp("2026-03-31"),pd.Timestamp("2026-06-30")]
        income=pd.DataFrame([[1000,1300],[80,150]],index=["Total Revenue","Net Income"],columns=cols)
        balance=pd.DataFrame([[700,800],[350,300]],index=["Stockholders Equity","Total Debt"],columns=cols)
        cashflow=pd.DataFrame([[110,180],[-30,-40]],index=["Operating Cash Flow","Capital Expenditure"],columns=cols)
        ae.yf=SimpleNamespace(Ticker=lambda symbol: SimpleNamespace(quarterly_income_stmt=income,quarterly_balance_sheet=balance,quarterly_cashflow=cashflow))
        fundamental, fundamental_audit=ae.fetch_yfinance_fundamental_snapshot("TEST")
        report["fundamental_fixture"]={"status":"PASS" if fundamental_audit["status"]=="OK" and fundamental.get("revenue_latest")==1300 else "FAIL","coverage_pct":fundamental.get("fundamental_coverage_pct"),"revenue_latest":fundamental.get("revenue_latest")}

        features={"smart_money_score":75,"accumulation_days20":5,"absorption_days20":2,"distribution_days20":0,"failed_absorption_days20":0,"close_acceptance20_pct":75,"up_value_ratio20_pct":65,"cmf20":0.12,"obv_slope20_pct":4,"pullback_volume_contraction_score":70,"volume_ratio20":1.4,"low20":500,"ema20":540,"last_price":560,"absorption_score":72,"execution_friction_score":20,"gap_risk_score":15,"high20":570,"last_date":"2026-07-31"}
        broker=ae.build_broker_inventory_proxy(features); orderbook=ae.build_orderbook_proxy(features)
        report["proxy_fixture"]={"status":"PASS" if broker["broker_inventory_evidence_type"]=="OHLCV_PROXY" and orderbook["orderbook_evidence_type"]=="OHLCV_EOD_PROXY" else "FAIL","broker_score":broker["broker_inventory_score"],"orderbook_score":orderbook["orderbook_trigger_score"]}
    finally:
        dp.requests.get = original_dp_get
        ae.requests.get = original_ae_get
        ae.yf = original_yf
    report["all_pass"] = all(item.get("status")=="PASS" for item in report.values() if isinstance(item,dict) and "status" in item)
    path=Path("validation_artifacts/PROVIDER_FIXTURE_VALIDATION_V1_4_0.json")
    path.write_text(json.dumps(report,indent=2,default=str)+"\n")
    print(json.dumps(report,indent=2,default=str))
    if not report["all_pass"]:
        raise SystemExit("provider fixture validation failed")


if __name__ == "__main__":
    main()
