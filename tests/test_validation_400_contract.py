from __future__ import annotations

import json

import validation_400_v1_6_3 as contract


def test_representative_400_positive_control_tracks_current_readiness_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    contract.main()

    result = json.loads(
        (tmp_path / "validation_artifacts_v1_6_3" / "VALIDATION_400_SYNTHETIC_V1_6_3.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["ticker_count"] == 400
    assert result["feature_state_ok"] == 400
    assert result["finite_feature_rows"] == 400
    assert result["auto_eod_ready"] >= 1
    assert result["invalid_production_gate_bypass"] == 0
    assert result["invalid_execution_hierarchy"] == 0
