import pandas as pd

import research_memory as rm


class _Config:
    ready = True


def test_replay_uses_raw_unique_memory_without_future_leak(monkeypatch):
    memory = {
        "AAA.JK": [
            {
                "ticker": "AAA.JK",
                "observed_at": "2026-08-10T00:00:00Z",
                "provider": "Issuer",
                "source_verified": True,
                "official_source": True,
                "content_sha256": "abc",
                "payload": {
                    "ticker": "AAA.JK",
                    "published_at": "2026-08-10T00:00:00Z",
                    "title": "AAA capacity expansion",
                    "summary": "Issuer disclosed capacity expansion",
                    "url": "https://issuer.example/a",
                    "source_verified": True,
                    "source_tier": "ISSUER",
                },
            },
            {
                "ticker": "AAA.JK",
                "observed_at": "2026-09-01T00:00:00Z",
                "provider": "Future",
                "source_verified": True,
                "official_source": True,
                "content_sha256": "future",
                "payload": {
                    "ticker": "AAA.JK",
                    "published_at": "2026-09-01T00:00:00Z",
                    "title": "future leak",
                    "url": "https://issuer.example/future",
                },
            },
        ]
    }
    monkeypatch.setattr(rm, "load_latest_research_memory", lambda *a, **k: memory)
    out = rm.load_replayable_narrative_events(
        _Config(), ["AAA.JK"], as_of="2026-08-14T12:00:00Z", limit_per_ticker=6
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["title"] == "AAA capacity expansion"
    assert bool(row["research_memory_replayed"]) is True
    assert row["collection_provider"] == "PERSISTED_RESEARCH_MEMORY"
    assert bool(row["source_verified"]) is True
    assert row["source_tier"] == "ISSUER"


def test_replay_deduplicates_same_raw_event(monkeypatch):
    base = {
        "ticker": "AAA.JK",
        "observed_at": "2026-08-10T00:00:00Z",
        "provider": "Issuer",
        "source_verified": True,
        "official_source": True,
        "payload": {
            "ticker": "AAA.JK",
            "published_at": "2026-08-10T00:00:00Z",
            "title": "AAA project update",
            "url": "https://issuer.example/project",
            "source_verified": True,
            "source_tier": "ISSUER",
        },
    }
    one = dict(base, content_sha256="a")
    two = dict(base, content_sha256="b")
    monkeypatch.setattr(rm, "load_latest_research_memory", lambda *a, **k: {"AAA.JK": [one, two]})
    out = rm.load_replayable_narrative_events(
        _Config(), ["AAA.JK"], as_of="2026-08-14T12:00:00Z"
    )
    assert len(out) == 1


def test_replay_does_not_promote_unverified_source(monkeypatch):
    memory = {
        "AAA.JK": [{
            "ticker": "AAA.JK",
            "observed_at": "2026-08-12T00:00:00Z",
            "provider": "Public News",
            "source_verified": False,
            "official_source": False,
            "content_sha256": "x",
            "payload": {
                "ticker": "AAA.JK",
                "published_at": "2026-08-12T00:00:00Z",
                "title": "AAA rumor",
                "url": "https://news.example/a",
            },
        }]
    }
    monkeypatch.setattr(rm, "load_latest_research_memory", lambda *a, **k: memory)
    out = rm.load_replayable_narrative_events(_Config(), ["AAA.JK"], as_of="2026-08-14T12:00:00Z")
    assert len(out) == 1
    row = out.iloc[0]
    assert bool(row.get("source_verified", False)) is False
    assert str(row.get("source_tier") or "") == ""


def test_derived_future_snapshot_family_is_not_replayed(monkeypatch):
    calls = []
    def fake_loader(config, tickers, family, **kwargs):
        calls.append(family)
        return {ticker: [] for ticker in tickers}
    monkeypatch.setattr(rm, "load_latest_research_memory", fake_loader)
    rm.load_replayable_narrative_events(_Config(), ["AAA.JK"], as_of="2026-08-14T12:00:00Z")
    assert calls == ["NARRATIVE_EVENT"]
