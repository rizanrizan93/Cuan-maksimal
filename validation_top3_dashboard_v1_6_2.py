from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from top3_dashboard import BLOCKED_STATES, enrich_dashboard_scores, render_top3_dashboard_html, select_top3

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "validation_artifacts_v1_6_2" / "TWENTY_TICKER_RADAR_SOURCE_V1_6_2.csv"
OUT = ROOT / "validation_artifacts_v1_6_2"
OUT.mkdir(exist_ok=True)

radar = pd.read_csv(SOURCE)
enriched = enrich_dashboard_scores(radar)
top3 = select_top3(enriched, limit=3)
html = render_top3_dashboard_html(
    top3,
    scan_id="controlled-20-v1.6.2",
    as_of="2026-08-06",
    market_regime="SELECTIVE",
)

assert len(top3) == 3
assert not set(top3["emir_decision_state"]).intersection(BLOCKED_STATES)
assert top3["emir_final_score"].between(0, 100).all()
assert (top3["emir_final_score"].round(1) == top3["emir_conviction_score"].round(1)).all()
assert "OHLCV PROXY — BUKAN IDENTITAS BROKER" in html
assert "@media(max-width:580px)" in html
assert "FINAL SCORE" in html

preview = (
    '<!doctype html><html><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
    '<body style="margin:0;background:#040b12;padding:12px">' + html + "</body></html>"
)
(OUT / "TOP3_DASHBOARD_PREVIEW_V1_6_2.html").write_text(preview, encoding="utf-8")
top3.to_csv(OUT / "TOP3_DASHBOARD_SAMPLE_V1_6_2.csv", index=False)

summary = {
    "state": "PASS",
    "source_rows": int(len(radar)),
    "top3_rows": int(len(top3)),
    "tickers": top3["ticker"].tolist(),
    "scores": top3["emir_final_score"].tolist(),
    "recommendations": top3["dashboard_recommendation"].tolist(),
    "blocked_or_reject_included": False,
    "final_score_alias_verified": True,
    "proxy_disclosure_present": True,
    "mobile_css_present": True,
    "html_escaping_tested_by_pytest": True,
}
(OUT / "TOP3_DASHBOARD_VALIDATION_V1_6_2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
