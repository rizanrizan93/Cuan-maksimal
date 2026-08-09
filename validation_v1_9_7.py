from __future__ import annotations

from io import StringIO
from pathlib import Path
import py_compile

import pandas as pd

from data_providers import parse_universe_frame
from persistence import DatabaseConfig
from scan_jobs import create_scan_job, universe_hash


ROOT = Path(__file__).resolve().parent


def check_compile() -> None:
    for source in ROOT.glob("*.py"):
        py_compile.compile(str(source), doraise=True)


def check_uploaded_metadata_survives_checkpoint() -> None:
    universe = parse_universe_frame(StringIO(
        "rank_universe,ticker,yahoo_ticker,sector_idx_ic,universe_role,priority,active_scan,theme,macro_theme,catalyst\n"
        "1,MARK,MARK.JK,Basic Materials,Sector Leader / Core,A,1,export growth,downstream manufacturing,capacity expansion\n"
    ))
    job = create_scan_job(
        DatabaseConfig(False, "", ""),
        scan_id="v197-validation",
        universe=universe,
        settings={"scan_mode": "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP"},
    )
    record = job["universe"][0]
    assert record["ticker"] == "MARK.JK"
    assert record["sector"] == "Basic Materials"
    assert record["universe_rank"] == "1"
    assert record["universe_role"] == "Sector Leader / Core"
    assert record["priority"] == "A"
    assert record["theme"] == "export growth"
    assert record["macro_theme"] == "downstream manufacturing"
    assert record["catalyst"] == "capacity expansion"


def check_blank_ticker_does_not_become_nan_symbol() -> None:
    universe = parse_universe_frame(StringIO("ticker,sector\n,Energy\nMARK,Basic Materials\n"))
    assert universe["ticker"].tolist() == ["MARK.JK"]


def check_metadata_changes_invalidate_resume_hash() -> None:
    before = pd.DataFrame([{"ticker": "MARK", "sector": "Basic Materials", "catalyst": "capacity expansion"}])
    after = before.copy()
    after.loc[0, "catalyst"] = "commercial operation"
    assert universe_hash(before) != universe_hash(after)


def check_v8_preflight_contract() -> None:
    sql = (ROOT / "database" / "verify_v8.sql").read_text(encoding="utf-8").lower()
    assert sql.count("has_table_privilege") == 3
    assert "cak_scan_jobs" in sql and "cak_research_memory" in sql


if __name__ == "__main__":
    checks = (
        check_compile,
        check_uploaded_metadata_survives_checkpoint,
        check_blank_ticker_does_not_become_nan_symbol,
        check_metadata_changes_invalidate_resume_hash,
        check_v8_preflight_contract,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("VALIDATION_V1_9_7=PASS")
