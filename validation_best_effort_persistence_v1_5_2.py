from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import persistence as ps
import persistent_cache as pc


class FakeResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._payload


def ready_config() -> ps.DatabaseConfig:
    return ps.config_from_mapping({
        "CAK_DATABASE_ENABLED": "true",
        "SUPABASE_URL": "https://x.supabase.co",
        "SUPABASE_SECRET_KEY": "sb_secret_fixture",
    })


def radar_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "ADMR.JK", "emir_decision_state": "EMIR_NO_EDGE_YET", "action": "WAIT_FOR_EDGE", "production_ready": False},
        {"ticker": "ELSA.JK", "emir_decision_state": "EMIR_WAIT_NARRATIVE", "action": "WAIT_NARRATIVE", "production_ready": False},
    ])


def validate_partial_persistence() -> dict:
    store: dict[str, list[dict]] = {}
    keys = {
        "cak_scan_runs": ("scan_id",),
        "cak_radar_snapshots": ("scan_id", "ticker"),
        "cak_narrative_events": ("event_id",),
        "cak_provider_audit": ("audit_id",),
        "cak_direct_evidence": ("evidence_id",),
        "cak_autonomous_evidence": ("evidence_id",),
        "cak_outcome_memory": ("outcome_id",),
    }

    original = ps._request

    def request(config, method, table, *, params=None, payload=None, **kwargs):
        scan_id = str((params or {}).get("scan_id", "eq.")).removeprefix("eq.")
        if method == "POST":
            if table == "cak_autonomous_evidence":
                raise RuntimeError("fixture partial table failure")
            incoming = list(payload or [])
            rows = store.setdefault(table, [])
            for item in incoming:
                keyset = keys[table]
                rows[:] = [row for row in rows if not all(row.get(k) == item.get(k) for k in keyset)]
                rows.append(dict(item))
            return FakeResponse(incoming)
        if method == "PATCH":
            updated = []
            for row in store.get(table, []):
                if row.get("scan_id") == scan_id:
                    row.update(dict(payload or {}))
                    updated.append(dict(row))
            return FakeResponse(updated)
        rows = [dict(row) for row in store.get(table, []) if row.get("scan_id") == scan_id]
        if kwargs.get("extra_headers", {}).get("Prefer") == "count=exact":
            return FakeResponse(rows[:1], {"Content-Range": f"0-0/{len(rows)}"})
        return FakeResponse(rows)

    ps._request = request
    try:
        write, verify, state = ps.persist_verify_scan_best_effort(
            ready_config(),
            scan_id="best-effort-partial",
            as_of="2026-08-06T09:00:00+07:00",
            radar=radar_frame(),
            events=pd.DataFrame(),
            provider_audit=pd.DataFrame([{"ticker": "ADMR.JK", "provider": "FIXTURE", "status": "OK"}]),
            direct_evidence=pd.DataFrame(),
            autonomous_evidence=pd.DataFrame([{"ticker": "ADMR.JK", "evidence_type": "AUTO"}]),
            outcomes=pd.DataFrame(),
        )
    finally:
        ps._request = original

    return {
        "write_state": str(write.iloc[0]["state"]),
        "readback_state": str(verify.iloc[0]["state"]),
        "persistence_state": str(state.iloc[0]["state"]),
        "publishable": bool(state.iloc[0]["publishable"]),
        "scan_run_status": store["cak_scan_runs"][0]["status"],
    }


def validate_memory_only() -> dict:
    disabled = ps.config_from_mapping({})
    write, verify, state = ps.persist_verify_scan_best_effort(
        disabled,
        scan_id="best-effort-memory",
        as_of="2026-08-06T09:00:00+07:00",
        radar=radar_frame(),
        events=pd.DataFrame(),
        provider_audit=pd.DataFrame(),
        direct_evidence=pd.DataFrame(),
    )
    return {
        "write_state": str(write.iloc[0]["state"]),
        "readback_state": str(verify.iloc[0]["state"]),
        "persistence_state": str(state.iloc[0]["state"]),
        "publishable": bool(state.iloc[0]["publishable"]),
    }


def sample_frame(start="2026-07-01", periods=20):
    index = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({
        "Open": range(100, 100 + periods),
        "High": range(102, 102 + periods),
        "Low": range(99, 99 + periods),
        "Close": range(101, 101 + periods),
        "Volume": [1_000_000 + i * 1000 for i in range(periods)],
    }, index=index)


def validate_mixed_cache() -> dict:
    cached_frame = sample_frame("2026-07-15", 15)
    cached_row = pc.build_ohlcv_cache_row(
        "ADMR.JK", cached_frame, period="5y", provider="CACHE", checked_at="2026-08-06T01:00:00Z"
    )
    fresh_frame = sample_frame("2026-07-20", 15)
    calls: list[str] = []
    original_read = pc.read_ohlcv_cache
    original_fetch = pc.fetch_many_ohlcv
    pc.read_ohlcv_cache = lambda *_: {"ADMR.JK": cached_row}

    def fetch(symbols, **kwargs):
        calls.extend(symbols)
        return {"ELSA.JK": fresh_frame}, pd.DataFrame([
            {"ticker": "ELSA.JK", "provider": "LIVE_FIXTURE", "status": "OK", "detail": ""}
        ])

    pc.fetch_many_ohlcv = fetch
    try:
        frames, audit, writes = pc.fetch_ohlcv_cache_first(
            ready_config(), ["ADMR", "ELSA"], now="2026-08-06T02:00:00Z", completed_only=False
        )
    finally:
        pc.read_ohlcv_cache = original_read
        pc.fetch_many_ohlcv = original_fetch

    return {
        "cache_hit_tickers": int((audit["status"] == "CACHE_HIT").sum()),
        "cold_refresh_tickers": int((audit["status"] == "COLD_REFRESH").sum()),
        "provider_calls": calls,
        "frames_returned": sorted(frames),
        "cache_rows_to_write": len(writes),
    }


def main():
    result = {
        "scanner_version": ps.SCANNER_VERSION,
        "partial_persistence": validate_partial_persistence(),
        "memory_only": validate_memory_only(),
        "mixed_cache": validate_mixed_cache(),
    }
    assert result["partial_persistence"]["persistence_state"] == "SCAN_COMPLETED_PARTIAL_PERSISTENCE"
    assert result["partial_persistence"]["publishable"] is True
    assert result["memory_only"]["persistence_state"] == "SCAN_COMPLETED_MEMORY_ONLY"
    assert result["memory_only"]["publishable"] is True
    assert result["mixed_cache"]["provider_calls"] == ["ELSA.JK"]
    output = Path("validation_artifacts_v1_5_2/BEST_EFFORT_PERSISTENCE_V1_5_2.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
