from __future__ import annotations

from pathlib import Path

from narrative_flow_engine import ENGINE_VERSION
from persistence import SCANNER_VERSION
from resumable_scan import PIPELINE_VERSION
from scan_jobs import JOB_VERSION


def main() -> None:
    expected = "1.9.13-cashflow-evidence-integrity"
    assert ENGINE_VERSION == expected
    assert SCANNER_VERSION == expected
    assert PIPELINE_VERSION == expected
    assert JOB_VERSION == expected
    app = Path("app.py").read_text()
    assert "idx_emir_next_leader_v1_9_13.csv" in app
    assert "idx_emir_real_money_top3_v1_9_13.csv" in app
    print("PASS v1.9.13 cashflow/evidence integrity")


if __name__ == "__main__":
    main()
