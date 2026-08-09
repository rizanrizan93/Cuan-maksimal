from __future__ import annotations

import pandas as pd

from scan_jobs import normalized_universe_records, next_chunk, stage_progress, universe_hash


def test_universe_hash_is_order_independent_and_deduplicated():
    first = pd.DataFrame({"ticker": ["BBCA", "ADMR.JK", "BBCA.JK"]})
    second = pd.DataFrame({"ticker": ["ADMR", "BBCA"]})
    assert universe_hash(first) == universe_hash(second)
    records = normalized_universe_records(first)
    assert [row["ticker"] for row in records] == ["BBCA.JK", "ADMR.JK"]


def test_next_chunk_is_bounded_and_resumable():
    items = [f"T{i}.JK" for i in range(53)]
    chunk, offset, done = next_chunk(items, 0, 20)
    assert len(chunk) == 20 and offset == 20 and not done
    chunk2, offset2, done2 = next_chunk(items, offset, 20)
    assert chunk2[0] == "T20.JK" and offset2 == 40 and not done2
    chunk3, offset3, done3 = next_chunk(items, offset2, 20)
    assert len(chunk3) == 13 and offset3 == 53 and done3


def test_stage_progress_is_monotonic():
    values = [
        stage_progress("BENCHMARK", 1, 1),
        stage_progress("OHLCV", 50, 100),
        stage_progress("FAST_RANKING", 1, 1),
        stage_progress("KSEI_SHORTLIST", 30, 30),
        stage_progress("NEWS_SHORTLIST", 30, 30),
        stage_progress("FUNDAMENTAL_SHORTLIST", 30, 30),
        stage_progress("FINALIZE", 1, 1),
        stage_progress("COMPLETED", 1, 1),
    ]
    assert values == sorted(values)
    assert values[-1] == 100.0


def test_create_job_settings_nan_are_json_safe(monkeypatch):
    import scan_jobs as sj
    from persistence import DatabaseConfig

    captured = {}

    class Response:
        def json(self):
            return captured["payload"]

    def fake_request(config, method, table, **kwargs):
        captured["payload"] = kwargs["payload"]
        return Response()

    monkeypatch.setattr(sj, "_request", fake_request)
    config = DatabaseConfig(True, "https://x.supabase.co", "sb_secret_x", key_type="SECRET")
    job = sj.create_scan_job(
        config,
        scan_id="s1",
        universe=pd.DataFrame({"ticker": ["BBCA"]}),
        settings={"manual_events": [{"score": float("nan")}]},
    )
    assert job["settings"]["manual_events"][0]["score"] is None
