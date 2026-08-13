from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import free_tier_storage as storage


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_research_memory_deduplicates_content_before_retention(monkeypatch):
    rows = [
        {"memory_id": "new", "ticker": "AAA.JK", "family": "FUNDAMENTAL", "content_sha256": "same", "effective_period": "2026-06-30", "observed_at": "2026-08-13", "updated_at": "2026-08-13"},
        {"memory_id": "dup", "ticker": "AAA.JK", "family": "FUNDAMENTAL", "content_sha256": "same", "effective_period": "2026-03-31", "observed_at": "2026-08-12", "updated_at": "2026-08-12"},
        {"memory_id": "u2", "ticker": "AAA.JK", "family": "FUNDAMENTAL", "content_sha256": "h2", "effective_period": "2025-12-31", "observed_at": "2026-08-11", "updated_at": "2026-08-11"},
        {"memory_id": "old", "ticker": "AAA.JK", "family": "FUNDAMENTAL", "content_sha256": "h3", "effective_period": "2025-09-30", "observed_at": "2026-08-10", "updated_at": "2026-08-10"},
    ]
    monkeypatch.setattr(storage, "_request", lambda *a, **k: _Response(rows))
    captured = []
    monkeypatch.setattr(storage, "_delete_ids", lambda config, table, column, ids, **kwargs: captured.extend(ids) or len(ids))
    report = storage.prune_research_memory_best_effort(SimpleNamespace(ready=True), keep_default=2, keep_narrative=6, max_rows_to_inspect=1000)
    assert report["duplicate_rows"] == 1
    assert set(captured) == {"dup", "old"}
    assert report["rows_deleted"] == 2


def test_free_tier_defaults_keep_single_heavy_scan():
    assert storage.prune_scan_history_best_effort.__kwdefaults__["keep_scan_runs"] == 1
    assert storage.prune_scan_history_best_effort.__kwdefaults__["keep_terminal_jobs"] == 1
    assert storage.prune_research_memory_best_effort.__kwdefaults__["keep_default"] == 4
    assert storage.prune_research_memory_best_effort.__kwdefaults__["keep_narrative"] == 6
