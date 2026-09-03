from __future__ import annotations

from pathlib import Path

import numpy as np

from shared_fundamental_runtime import canonicalize_metric_rows, normalize_operational_snapshot_rows, normalize_pluang_payloads, normalize_yahoo_payloads
from shared_fundamental_runtime_patch import _official_payload, _proxy_payload


ROOT = Path(__file__).resolve().parents[1]


def test_operational_decimal_ratios_become_canonical_percentage_points() -> None:
    rows = normalize_operational_snapshot_rows([{
        "ticker":"TLKM.JK", "revenue_growth":0.0639, "earnings_growth":0.2485,
        "roe":0.1921, "roa":0.0911, "net_margin":0.1827, "debt_equity":0.5998,
        "fundamental_coverage":55.0, "fundamental_source_families":"IDX_OFFICIAL_XBRL • YAHOO",
        "fundamental_fetched_at":"2026-09-03T00:00:00+00:00", "content_hash":"abc",
    }])
    values={row["metric_name"]:row["metric_value"] for row in rows}
    assert values["revenue_growth_pct"] == 6.39
    assert values["roe_pct"] == 19.21
    assert values["debt_equity"] == 0.5998
    assert values["fundamental_coverage_pct"] == 55.0


def test_canonical_exact_official_builds_yoy_and_ratios() -> None:
    def r(period: str, metric: str, value: float) -> dict[str, object]:
        return {"provider":"OPERATIONAL_FINANCIAL_FACT_BRIDGE","ticker":"ABMM","period_end":period,"metric_name":metric,"metric_value":value,"metric_unit":"NORMALIZED","source_families":"IDX_OFFICIAL_XBRL","official_verified":True,"source_record_hash":f"{period}-{metric}","lineage_state":"OPERATIONAL_FINANCIAL_FACT_EXACT_LINEAGE","observed_at":"2026-08-14T00:00:00+00:00","validation_state":"VALID","fetched_at":"2026-09-03T00:00:00+00:00"}
    item=canonicalize_metric_rows([r("2025-06-30","revenue",100),r("2025-06-30","net_income",10),r("2026-06-30","revenue",120),r("2026-06-30","net_income",15),r("2026-06-30","equity",100),r("2026-06-30","total_debt",40),r("2026-06-30","cash",20)])["ABMM"]
    assert item["official_period_end"] == "2026-06-30"
    assert item["official_metrics"]["revenue_growth_yoy_pct"] == 20.0
    assert item["official_metrics"]["interest_bearing_debt_to_equity"] == 0.4


def test_pluang_and_yahoo_structured_facts_are_nonofficial() -> None:
    pluang = normalize_pluang_payloads("BBCA", {"code":"BBCA","source":"pluang","ratios":{"profitability":{"roe":"20.44%"}},"overview":{},"earnings":[{"quarter":"Q2 '26"}]}, {"code":"BBCA","source":"pluang","quarterly":{}}, observed_at="2026-09-04T00:00:00+00:00")
    yahoo = normalize_yahoo_payloads("BBCA.JK", {"symbol":"BBCA.JK","provider":"yahoo","revenueGrowthPercent":2.5,"totalCash":50,"totalDebt":20}, {"income":{"items":[{"date":"2026-06-30","revenue":120,"netIncome":60},{"date":"2025-06-30","revenue":100,"netIncome":50}]},"balance":{"items":[{"date":"2026-06-30","totalAssets":500,"stockholdersEquity":200,"totalDebt":20,"cash":50}]},"cashflow":{"items":[{"date":"2026-06-30","operatingCashFlow":70}]}}, observed_at="2026-09-04T00:00:00+00:00")
    assert {row["metric_name"]:row["metric_value"] for row in pluang}["roe_pct"] == 20.44
    yvalues={row["metric_name"]:row["metric_value"] for row in yahoo}
    assert yvalues["earnings_growth_pct"] == 20.0
    assert yvalues["debt_equity"] == 0.1
    assert all(not row["official_verified"] for row in pluang+yahoo)


def test_emir_proxy_adapter_keeps_percent_units_and_does_not_label_official_cashflow_ttm() -> None:
    item={"ticker":"TLKM","proxy_metrics":{"revenue_growth_pct":6.39,"earnings_growth_pct":24.85,"roe_pct":19.21,"net_margin_pct":18.27,"debt_equity":0.6,"operating_cash_flow":100},"proxy_period_end":"2026-06-30","proxy_observed_at":"2026-08-14T00:00:00+00:00","official_metrics":{},"official_period_end":None,"official_observed_at":None,"official_coverage_pct":0,"source_families":["YAHOO"]}
    payload=_proxy_payload(item)
    assert payload["revenue_growth_pct"] == 6.39
    assert payload["roe_ttm_pct"] == 19.21
    assert payload["interest_bearing_debt_to_equity"] == 0.6
    assert payload["fundamental_cashflow_state"] == "SHARED_OCF_AVAILABLE_PERIOD_BASIS"
    assert payload["fundamental_provenance_state"] == "SHARED_FACTUAL_EVIDENCE_PROXY"


def test_emir_official_adapter_is_exact_fact_only() -> None:
    item={"ticker":"ABMM","official_metrics":{"revenue":120,"net_income":15,"operating_cash_flow":20,"interest_bearing_debt_to_equity":0.4,"revenue_growth_yoy_pct":20},"official_period_end":"2026-06-30","official_observed_at":"2026-08-14T00:00:00+00:00","official_coverage_pct":50}
    payload=_official_payload(item)
    assert payload["idx_official_source_verified"] is True
    assert payload["idx_official_period_end"] == "2026-06-30"
    assert payload["idx_official_revenue_growth_yoy_pct"] == 20
    assert payload["idx_official_provenance_state"] == "SHARED_HUB_EXACT_OFFICIAL_FACTS"


def test_shared_runtime_contract_has_no_cross_scanner_decision_fields() -> None:
    text=(ROOT/"shared_fundamental_runtime.py").read_text(encoding="utf-8").lower()
    for forbidden in ("emir_score","pasticuan_score","entry_price","stop_loss","take_profit"):
        assert forbidden not in text
