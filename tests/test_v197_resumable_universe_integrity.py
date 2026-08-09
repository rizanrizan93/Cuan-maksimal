from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from data_providers import parse_universe_frame
from persistence import DatabaseConfig
from scan_jobs import create_scan_job, normalized_universe_records, universe_hash


ROOT = Path(__file__).resolve().parents[1]


def test_supplied_300_style_metadata_is_preserved() -> None:
    universe = parse_universe_frame(StringIO(
        "rank_universe,ticker,yahoo_ticker,sector_idx_ic,universe_role,priority,active_scan\n"
        "1,AADI,AADI.JK,Energy,Sector Leader / Core,A,1\n"
    ))
    row = universe.iloc[0]
    assert row["ticker"] == "AADI.JK"
    assert row["yahoo_ticker"] == "AADI.JK"
    assert row["sector"] == "Energy"
    assert row["universe_rank"] == "1"
    assert row["universe_role"] == "Sector Leader / Core"
    assert row["priority"] == "A"
    assert row["active_scan"] == "1"


def test_blank_ticker_is_not_converted_to_nan_symbol() -> None:
    universe = parse_universe_frame(StringIO(
        "ticker,sector\n"
        ",Energy\n"
        "MARK,Basic Materials\n"
    ))
    assert universe["ticker"].tolist() == ["MARK.JK"]


def test_resumable_job_keeps_issuer_context_and_merges_duplicate_blanks() -> None:
    universe = pd.DataFrame([
        {
            "ticker": "MARK",
            "company_name": "PT Mark Dynamics Indonesia Tbk",
            "sector": "Basic Materials",
            "theme": "",
            "macro_theme": "downstream manufacturing",
            "catalyst": "capacity expansion",
        },
        {"ticker": "MARK.JK", "theme": "export growth"},
    ])
    records = normalized_universe_records(universe)
    assert len(records) == 1
    assert records[0]["ticker"] == "MARK.JK"
    assert records[0]["theme"] == "export growth"
    assert records[0]["macro_theme"] == "downstream manufacturing"
    assert records[0]["catalyst"] == "capacity expansion"

    job = create_scan_job(
        DatabaseConfig(False, "", ""),
        scan_id="metadata-contract",
        universe=universe,
        settings={"scan_mode": "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP"},
    )
    assert job["universe"][0]["theme"] == "export growth"
    assert job["universe"][0]["macro_theme"] == "downstream manufacturing"
    assert job["universe"][0]["catalyst"] == "capacity expansion"


def test_universe_hash_is_order_independent_but_metadata_sensitive() -> None:
    first = pd.DataFrame([
        {"ticker": "BBCA", "sector": "Financials", "theme": "digital banking"},
        {"ticker": "ADMR", "sector": "Basic Materials", "theme": "smelter"},
    ])
    reordered = first.iloc[::-1].reset_index(drop=True)
    changed = first.copy()
    changed.loc[changed["ticker"].eq("ADMR"), "theme"] = "different issuer thesis"
    assert universe_hash(first) == universe_hash(reordered)
    assert universe_hash(first) != universe_hash(changed)


def test_v8_verifier_covers_runtime_tables_and_write_privileges() -> None:
    sql = (ROOT / "database" / "verify_v8.sql").read_text(encoding="utf-8").lower()
    for table in (
        "cak_scan_runs", "cak_radar_snapshots", "cak_narrative_events",
        "cak_provider_audit", "cak_direct_evidence", "cak_autonomous_evidence",
        "cak_outcome_memory", "cak_ohlcv_cache", "cak_source_cache",
        "cak_scan_jobs", "cak_scan_job_chunks", "cak_research_memory",
    ):
        assert table in sql
    assert "has_table_privilege" in sql
    assert "'select'" in sql and "'insert'" in sql and "'update'" in sql


def test_streamlit_checkpoint_failure_is_resumable() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "Checkpoint berhenti aman" in source
    assert "Progress yang sudah committed tetap dapat dilanjutkan" in source
