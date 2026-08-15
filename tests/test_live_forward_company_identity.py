from __future__ import annotations

import live_forward_evidence as lfe


class _Response:
    def __init__(self, xml: str):
        self.content = xml.encode('utf-8')
    def raise_for_status(self):
        return None


def test_company_name_resolves_forward_event_without_ticker_in_title(monkeypatch):
    xml = """<rss><channel><item>
    <title>Mark Dynamics catat pesanan hingga kuartal III dan kapasitas penuh</title>
    <link>https://example.com/mark</link>
    <pubDate>Fri, 14 Aug 2026 08:00:00 GMT</pubDate><source>Example</source>
    </item></channel></rss>"""
    monkeypatch.setattr(lfe.requests, 'get', lambda *a, **k: _Response(xml))
    events, audit = lfe.collect_live_forward_evidence(
        ['MARK.JK'], company_names={'MARK.JK': 'MARK DYNAMICS INDONESIA Tbk, PT'}, max_workers=1,
    )
    assert len(events) == 1
    assert events.iloc[0]['entity_match_method'].startswith('COMPANY_TOKEN_MATCH_')
    assert bool(events.iloc[0]['source_verified']) is False
    assert bool(events.iloc[0]['source_quorum_verified']) is False
    assert audit.iloc[0]['state'] == 'MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND'
