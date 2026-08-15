from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import live_forward_evidence as lfe
import runtime_integrity_patch as rip


class _Response:
    def __init__(self, xml: str):
        self.content = xml.encode("utf-8")
    def raise_for_status(self) -> None:
        return None


def _rss(title: str) -> str:
    return f"""<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel><item><title>{title}</title><link>https://example.com/a</link>
    <pubDate>Fri, 14 Aug 2026 08:00:00 GMT</pubDate><source>Example Publisher</source></item></channel></rss>"""


def test_live_forward_event_is_research_only(monkeypatch):
    monkeypatch.setattr(lfe.requests, "get", lambda *a, **k: _Response(_rss("ABCD dapat kontrak baru dengan order book meningkat")))
    events, audit = lfe.collect_live_forward_evidence(["ABCD.JK"], max_workers=1)
    assert len(events) == 1
    assert bool(events.iloc[0]["forward_research_only"]) is True
    assert bool(events.iloc[0]["source_quorum_verified"]) is False
    assert audit.iloc[0]["state"] == "MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND"
    assert float(audit.iloc[0]["coverage_pct"]) == 100.0


def test_no_material_forward_event_records_complete_check(monkeypatch):
    monkeypatch.setattr(lfe.requests, "get", lambda *a, **k: _Response(_rss("ABCD mengadakan kegiatan sosial")))
    events, audit = lfe.collect_live_forward_evidence(["ABCD.JK"], max_workers=1)
    assert events.empty
    assert audit.iloc[0]["state"] == "FORWARD_CHECK_COMPLETED_NO_MATERIAL_EVENT"
    assert float(audit.iloc[0]["coverage_pct"]) == 100.0


def test_checkpoint_mismatch_with_full_readback_retries_to_success(monkeypatch):
    calls = {"n": 0}
    bad = pd.DataFrame([{"state": "CACHE_DATABASE_NOT_COMMITTED", "rows_expected": 20, "rows_verified": 20}])
    good = pd.DataFrame([{"state": "CACHE_DATABASE_COMMITTED", "rows_expected": 20, "rows_verified": 20}])

    def persist(*args, **kwargs):
        calls["n"] += 1
        return pd.DataFrame(), good if calls["n"] >= 2 else bad

    module = SimpleNamespace(persist_verify_cache_bundle=persist)
    cache = SimpleNamespace(cache_commit_succeeded=lambda frame: str(frame.iloc[0].get("state")) == "CACHE_DATABASE_COMMITTED")
    monkeypatch.setattr(rip.time, "sleep", lambda *_: None)
    rip._wrap_cache_checkpoint_retry(module, cache)
    _, verify = module.persist_verify_cache_bundle(None)
    assert calls["n"] == 2
    assert verify.iloc[0]["state"] == "CACHE_DATABASE_COMMITTED"


def test_smart_money_cost_is_one_block_per_card():
    import top3_dashboard

    lfe.install_dashboard_cost_integrity()
    top = pd.DataFrame([
        {"ticker": "AAA.JK", "last_price": 100.0, "research_accumulation_zone_low": 90.0, "research_accumulation_zone_high": 94.0, "silent_accumulation_score": 70.0, "emir_conviction_score": 70.0, "emir_decision_state": "EMIR_NO_EDGE_YET"},
        {"ticker": "BBB.JK", "last_price": 200.0, "research_accumulation_zone_low": 180.0, "research_accumulation_zone_high": 188.0, "silent_accumulation_score": 65.0, "emir_conviction_score": 65.0, "emir_decision_state": "EMIR_NO_EDGE_YET"},
        {"ticker": "CCC.JK", "last_price": 300.0, "research_accumulation_zone_low": 270.0, "research_accumulation_zone_high": 282.0, "silent_accumulation_score": 60.0, "emir_conviction_score": 60.0, "emir_decision_state": "EMIR_NO_EDGE_YET"},
    ])
    html = top3_dashboard.render_top3_dashboard_html(top)
    assert html.count('class="es-cost-basis"') == 3
    # Production cards render the IDX symbol without the .JK suffix.
    first_ticker = html.find(">AAA<")
    first_cost = html.find('class="es-cost-basis"')
    second_ticker = html.find(">BBB<", first_ticker + 1)
    second_cost = html.find('class="es-cost-basis"', first_cost + 1)
    third_ticker = html.find(">CCC<", second_ticker + 1)
    third_cost = html.find('class="es-cost-basis"', second_cost + 1)
    assert -1 not in (first_ticker, first_cost, second_ticker, second_cost, third_ticker, third_cost)
    assert first_ticker < first_cost < second_ticker < second_cost < third_ticker < third_cost
