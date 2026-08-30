from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import final_decision as decision
import public_idx_broker_flow as broker
import zapi_flow_enrichment as zapi


def _material_frozen_snapshot() -> pd.DataFrame:
    frame = pd.DataFrame([
        {
            "ticker": "AAA.JK", "decision_snapshot_version": decision.FINAL_DECISION_VERSION,
            "raw_research_rank": 1, "guarded_decision_priority_rank": 1,
            "production_real_money_rank": 1, "emir_final_score": 88.25,
            "production_real_money_score": 84.5, "production_ready": True,
            "real_money_ready": True, "real_money_candidate": True,
            "real_money_entry_candidate": True, "emir_decision_state": "READY",
            "real_money_gate_state": "REAL_MONEY_READY", "real_money_hard_block_count": 0,
            "real_money_block_reasons": "NONE", "execution_entry_low": 1000.0,
            "execution_entry_high": 1020.0, "execution_stop_loss": 950.0,
            "execution_tp1": 1100.0, "execution_tp2": 1200.0,
            "execution_rr_tp1": 2.0, "execution_rr_tp2": 4.0,
            "execution_geometry_valid": True,
            "future_forward_provenance_state": "OFFICIAL_DIRECT_VERIFIED",
            "future_direct_forward_authorization_eligible": True,
            "broker_flow_provenance": "OFFICIAL_IDX_PUBLIC_EOD",
            "entry_authorization_state": "SCANNER_AUTHORIZED_DIRECT_VERIFIED",
        },
        {
            "ticker": "BBB.JK", "decision_snapshot_version": decision.FINAL_DECISION_VERSION,
            "raw_research_rank": 2, "guarded_decision_priority_rank": 2,
            "production_real_money_rank": np.nan, "emir_final_score": 76.0,
            "production_real_money_score": 70.0, "production_ready": False,
            "real_money_ready": False, "real_money_candidate": False,
            "real_money_entry_candidate": False, "emir_decision_state": "WAIT",
            "real_money_gate_state": "REAL_MONEY_WAIT_TIMING", "real_money_hard_block_count": 0,
            "real_money_block_reasons": "NONE", "execution_entry_low": 500.0,
            "execution_entry_high": 510.0, "execution_stop_loss": 470.0,
            "execution_tp1": 550.0, "execution_tp2": 600.0,
            "execution_rr_tp1": 1.5, "execution_rr_tp2": 3.0,
            "execution_geometry_valid": True,
            "future_forward_provenance_state": "RESEARCH_ONLY",
            "future_direct_forward_authorization_eligible": False,
            "broker_flow_provenance": "OFFICIAL_IDX_PUBLIC_EOD",
            "entry_authorization_state": "WAIT_TIMING_NO_ENTRY",
        },
    ])
    frame["decision_snapshot_state"] = decision.FINAL_DECISION_STATE
    frame["decision_snapshot_fingerprint"] = decision._fingerprint(frame)
    return frame


def _radar() -> pd.DataFrame:
    rows = []
    for ticker, score in (("AAA.JK", 70.0), ("BBB.JK", 80.0), ("CCC.JK", 60.0)):
        rows.append({
            "ticker": ticker,
            "emir_conviction_score": score,
            "emir_final_score": score,
            "smart_money_score": 60.0,
            "smart_money_coverage_pct": 80.0,
            "emir_decision_state": "EMIR_WAIT_REACCUMULATION",
            "real_money_candidate": False,
            "real_money_entry_candidate": False,
            "real_money_ready": False,
            "real_money_gate_state": "REAL_MONEY_WAIT_TIMING",
            "entry_authorization_state": "WAIT_TIMING_NO_ENTRY",
            "real_money_block_reasons": "NONE",
            "real_money_hard_block_count": 0,
            "next_leader_score": score,
        })
    return pd.DataFrame(rows)


def _install_finalizer_fakes(monkeypatch) -> dict[str, int]:
    calls = {"zapi": 0, "broker": 0, "dashboard": 0}

    def zapi_once(frame: pd.DataFrame) -> pd.DataFrame:
        calls["zapi"] += 1
        out = frame.copy()
        out.loc[out["ticker"].eq("AAA.JK"), "emir_conviction_score"] += 20.0
        out["emir_final_score"] = out["emir_conviction_score"]
        return out

    def broker_once(frame: pd.DataFrame) -> pd.DataFrame:
        calls["broker"] += 1
        return frame.copy()

    def dashboard_once(frame: pd.DataFrame, frames=None) -> pd.DataFrame:
        calls["dashboard"] += 1
        out = frame.copy()
        out["dashboard_flow_score"] = out["smart_money_score"]
        out["dashboard_silent_accum_score"] = out["smart_money_score"]
        out["liquidity_score"] = 70.0
        out["distribution_score"] = 20.0
        return out

    monkeypatch.setattr(decision.zapi_flow_enrichment, "enrich_emir_radar", zapi_once)
    monkeypatch.setattr(decision.public_idx_broker_flow, "enrich_emir_broker", broker_once)
    monkeypatch.setattr(decision.top3_dashboard, "_canonical_enrich_dashboard_scores", dashboard_once)
    monkeypatch.setattr(decision.zapi_flow_enrichment, "blend_emir_dashboard_output", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_adjust_cost_confidence", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_apply_foreign_shock_guard", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_apply_gate_audit", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "refine_emir_proxy_authorization_tier", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "enrich_emir_shadow", lambda frame: frame.copy())
    return calls


def test_finalization_enriches_then_reranks_once_and_freezes(monkeypatch) -> None:
    calls = _install_finalizer_fakes(monkeypatch)
    frozen = decision.finalize_decision_snapshot(_radar())

    assert frozen.sort_values("raw_research_rank")["ticker"].tolist() == ["AAA.JK", "BBB.JK", "CCC.JK"]
    assert frozen["decision_snapshot_state"].eq(decision.FINAL_DECISION_STATE).all()
    assert frozen["decision_enrichment_pass_count"].eq(1).all()
    assert frozen["decision_ranking_pass_count"].eq(1).all()
    assert calls == {"zapi": 1, "broker": 1, "dashboard": 1}

    repeated = decision.finalize_decision_snapshot(frozen)
    pdt.assert_frame_equal(frozen, repeated)
    assert calls == {"zapi": 1, "broker": 1, "dashboard": 1}


def test_persisted_displayed_exported_and_selector_inputs_match(monkeypatch) -> None:
    _install_finalizer_fakes(monkeypatch)
    persisted = decision.finalize_decision_snapshot(_radar())
    displayed = decision.select_top3(persisted, 3)
    exported = decision.export_decision_snapshot(persisted)

    persisted_order = persisted.sort_values("raw_research_rank")["ticker"].tolist()
    assert displayed["ticker"].tolist() == persisted_order
    assert exported.sort_values("raw_research_rank")["ticker"].tolist() == persisted_order
    pdt.assert_series_equal(
        persisted.set_index("ticker")["raw_research_rank"].sort_index(),
        exported.set_index("ticker")["raw_research_rank"].sort_index(),
    )
    assert decision.select_real_money_top3(persisted, 3).empty


def test_snapshot_and_render_consumers_cannot_mutate_frozen_source(monkeypatch) -> None:
    _install_finalizer_fakes(monkeypatch)
    frozen = decision.finalize_decision_snapshot(_radar())
    original = frozen.copy(deep=True)
    detached = decision.export_decision_snapshot(frozen)
    detached.loc[:, "emir_conviction_score"] = 0.0

    monkeypatch.setattr(decision.top3_dashboard, "render_top3_dashboard_html", lambda frame, **kwargs: frame.assign(emir_conviction_score=0.0) is not None and "ok")
    assert decision.render_top3_dashboard_html(decision.select_top3(frozen), scan_id="S") == "ok"
    pdt.assert_frame_equal(frozen, original)


def test_persistence_numeric_dtype_normalization_preserves_freeze(monkeypatch) -> None:
    _install_finalizer_fakes(monkeypatch)
    frozen = decision.finalize_decision_snapshot(_radar())
    reloaded = frozen.copy()
    for column in ("raw_research_rank", "guarded_decision_priority_rank"):
        reloaded[column] = pd.to_numeric(reloaded[column], errors="coerce").astype(float)
    assert decision.is_final_decision_snapshot(reloaded) is True


def test_fingerprint_accepts_identical_and_reordered_rows() -> None:
    frozen = _material_frozen_snapshot()
    assert decision.is_final_decision_snapshot(frozen) is True
    assert decision.is_final_decision_snapshot(frozen.iloc[::-1].copy()) is True


def test_fingerprint_normalizes_persisted_boolean_representations() -> None:
    reloaded = _material_frozen_snapshot()
    for column in ("production_ready", "real_money_ready", "future_direct_forward_authorization_eligible"):
        reloaded[column] = reloaded[column].astype(float)
    assert decision.is_final_decision_snapshot(reloaded) is True


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("execution_entry_low", 1000.01),
        ("execution_stop_loss", 949.99),
        ("execution_tp1", 1100.01),
        ("execution_tp2", 1200.01),
        ("execution_rr_tp1", 2.000001),
        ("production_ready", False),
        ("real_money_ready", False),
        ("emir_decision_state", "WAIT"),
        ("real_money_gate_state", "REAL_MONEY_HARD_BLOCK"),
        ("real_money_block_reasons", "INVALID_GEOMETRY"),
        ("raw_research_rank", 2),
        ("emir_final_score", 88.250001),
        ("future_forward_provenance_state", "RESEARCH_ONLY"),
        ("future_direct_forward_authorization_eligible", False),
        ("broker_flow_provenance", "UNKNOWN"),
    ],
)
def test_fingerprint_rejects_decision_material_mutation(column: str, replacement: object) -> None:
    mutated = _material_frozen_snapshot()
    mutated.loc[mutated["ticker"].eq("AAA.JK"), column] = replacement
    assert decision.is_final_decision_snapshot(mutated) is False


@pytest.mark.parametrize("missing", ["decision_snapshot_fingerprint", "decision_snapshot_version"])
def test_fingerprint_fails_closed_when_required_metadata_is_missing(missing: str) -> None:
    assert decision.is_final_decision_snapshot(_material_frozen_snapshot().drop(columns=[missing])) is False


def test_fingerprint_fails_closed_for_unsupported_snapshot_version() -> None:
    unsupported = _material_frozen_snapshot()
    unsupported["decision_snapshot_version"] = "UNSUPPORTED"
    assert decision.is_final_decision_snapshot(unsupported) is False


def test_fingerprint_rejects_duplicate_ambiguous_identity() -> None:
    frozen = _material_frozen_snapshot()
    duplicate = pd.concat([frozen.iloc[[0]], frozen.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="DUPLICATE_ROW_IDENTITY"):
        decision._fingerprint(duplicate)
    assert decision.is_final_decision_snapshot(duplicate) is False


def test_decision_contract_is_independent_of_dashboard_patch_import_order() -> None:
    import execution_research_top3_runtime_patch as runtime_patch
    import top3_dashboard
    import top3_dashboard_legacy

    frozen = _radar()
    frozen = __import__("evidence_governance").apply_three_rank_contract(frozen)
    frozen["decision_snapshot_state"] = decision.FINAL_DECISION_STATE
    frozen["decision_snapshot_version"] = decision.FINAL_DECISION_VERSION
    frozen["decision_snapshot_fingerprint"] = decision._fingerprint(frozen)

    original_dashboard = top3_dashboard.select_top3
    original_legacy = top3_dashboard_legacy.select_top3
    try:
        before = decision.select_top3(frozen)["ticker"].tolist()
        runtime_patch.install()
        after_install = decision.select_top3(frozen)["ticker"].tolist()
        reloaded = importlib.reload(decision)
        after_reload = reloaded.select_top3(frozen)["ticker"].tolist()

        assert before == after_install == after_reload
        assert reloaded.select_top3 is not top3_dashboard.select_top3
        assert reloaded.select_top3 is not top3_dashboard_legacy.select_top3
    finally:
        top3_dashboard.select_top3 = original_dashboard
        top3_dashboard_legacy.select_top3 = original_legacy


def test_runtime_enrichment_compatibility_names_delegate_to_canonical_finalizer() -> None:
    import broker_runtime_patch
    import top3_dashboard
    import top3_dashboard_legacy
    import zapi_runtime_patch

    original_dashboard = top3_dashboard.enrich_dashboard_scores
    original_legacy = top3_dashboard_legacy.enrich_dashboard_scores
    try:
        zapi_runtime_patch.install()
        assert top3_dashboard.enrich_dashboard_scores is decision.finalize_decision_snapshot
        assert top3_dashboard_legacy.enrich_dashboard_scores is decision.finalize_decision_snapshot
        broker_runtime_patch.install()
        assert top3_dashboard.enrich_dashboard_scores is decision.finalize_decision_snapshot
        assert top3_dashboard_legacy.enrich_dashboard_scores is decision.finalize_decision_snapshot
    finally:
        top3_dashboard.enrich_dashboard_scores = original_dashboard
        top3_dashboard_legacy.enrich_dashboard_scores = original_legacy


def test_zapi_enrichment_is_decision_idempotent(monkeypatch) -> None:
    features = pd.DataFrame([{
        "ticker": "TEST",
        "zapi_foreign_flow_score": 100.0,
        "zapi_foreign_flow_coverage_pct": 100.0,
        "zapi_accumulation_confirmation_score": 90.0,
        "zapi_smart_money_confirmation_score": 90.0,
    }])
    monkeypatch.setattr(zapi, "get_zapi_features", lambda tickers: (features.copy(), {"state": "TEST"}))
    base = pd.DataFrame([{
        "ticker": "TEST.JK", "smart_money_score": 60.0,
        "smart_money_coverage_pct": 80.0, "emir_conviction_score": 70.0,
        "emir_final_score": 70.0,
    }])
    once = zapi.enrich_emir_radar(base)
    twice = zapi.enrich_emir_radar(once)
    fields = ["smart_money_score", "smart_money_coverage_pct", "emir_conviction_score", "emir_final_score", "zapi_emir_conviction_delta"]
    pdt.assert_frame_equal(once[fields], twice[fields])
    assert once["zapi_enrichment_compatibility_state"].eq("CANONICAL_ENRICHED").all()


def test_legacy_zapi_adjustment_is_preserved_without_second_delta(monkeypatch) -> None:
    monkeypatch.setattr(zapi, "get_zapi_features", lambda tickers: pytest.fail("legacy path fetched ZAPI features"))
    legacy = pd.DataFrame([{
        "ticker": "TEST.JK", "smart_money_score": 72.5,
        "emir_conviction_score": 72.5, "emir_final_score": 72.5,
        "zapi_emir_conviction_delta": 2.5,
    }])
    out = zapi.enrich_emir_radar(legacy)
    assert out.loc[0, "emir_conviction_score"] == 72.5
    assert out.loc[0, "emir_final_score"] == 72.5
    assert out.loc[0, "zapi_enrichment_compatibility_state"] == "LEGACY_ALREADY_ENRICHED_PRESERVED_NO_REAPPLY"


def test_broker_enrichment_is_decision_idempotent(monkeypatch) -> None:
    features = pd.DataFrame([{
        "ticker": "TEST", "broker_smart_money_confirmation_score": 90.0,
        "broker_flow_coverage_pct": 100.0,
    }])
    monkeypatch.setattr(broker, "load_public_cache", lambda: pd.DataFrame([{"ticker": "TEST"}]))
    monkeypatch.setattr(broker, "score_broker_history", lambda history, universe: features.copy())
    base = pd.DataFrame([{
        "ticker": "TEST.JK", "smart_money_score": 60.0,
        "emir_conviction_score": 70.0, "emir_final_score": 70.0,
    }])
    once = broker.enrich_emir_broker(base)
    twice = broker.enrich_emir_broker(once)
    fields = ["smart_money_score", "emir_conviction_score", "emir_final_score", "broker_emir_conviction_delta"]
    pdt.assert_frame_equal(once[fields], twice[fields])
    assert once["broker_enrichment_compatibility_state"].eq("CANONICAL_ENRICHED").all()


def test_legacy_broker_adjustment_is_preserved_without_second_delta(monkeypatch) -> None:
    monkeypatch.setattr(broker, "load_public_cache", lambda: pytest.fail("legacy path fetched broker cache"))
    legacy = pd.DataFrame([{
        "ticker": "TEST.JK", "smart_money_score": 72.5,
        "emir_conviction_score": 72.5, "emir_final_score": 72.5,
        "broker_emir_conviction_delta": 1.2,
    }])
    out = broker.enrich_emir_broker(legacy)
    assert out.loc[0, "emir_conviction_score"] == 72.5
    assert out.loc[0, "emir_final_score"] == 72.5
    assert out.loc[0, "broker_enrichment_compatibility_state"] == "LEGACY_ALREADY_ENRICHED_PRESERVED_NO_REAPPLY"


def _install_compatibility_finalizer_fakes(monkeypatch) -> None:
    monkeypatch.setattr(decision.top3_dashboard, "_canonical_enrich_dashboard_scores", lambda frame, frames=None: frame.copy())
    monkeypatch.setattr(decision.zapi_flow_enrichment, "blend_emir_dashboard_output", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_recompute_real_money", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_adjust_cost_confidence", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_apply_foreign_shock_guard", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_apply_gate_audit", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "refine_emir_proxy_authorization_tier", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "enrich_emir_shadow", lambda frame: frame.copy())


def test_legacy_both_overlays_finalize_without_stacking_and_repeat_identically(monkeypatch) -> None:
    _install_compatibility_finalizer_fakes(monkeypatch)
    monkeypatch.setattr(zapi, "get_zapi_features", lambda tickers: pytest.fail("legacy path fetched ZAPI features"))
    monkeypatch.setattr(broker, "load_public_cache", lambda: pytest.fail("legacy path fetched broker cache"))
    legacy = _radar()
    legacy["emir_conviction_score"] = 72.5
    legacy["emir_final_score"] = 72.5
    legacy["zapi_emir_conviction_delta"] = 2.5
    legacy["broker_emir_conviction_delta"] = 1.2

    once = decision.finalize_decision_snapshot(legacy)
    twice = decision.finalize_decision_snapshot(once)

    assert once["emir_conviction_score"].eq(72.5).all()
    assert once["emir_final_score"].eq(72.5).all()
    assert once["zapi_enrichment_compatibility_state"].eq("LEGACY_ALREADY_ENRICHED_PRESERVED_NO_REAPPLY").all()
    assert once["broker_enrichment_compatibility_state"].eq("LEGACY_ALREADY_ENRICHED_PRESERVED_NO_REAPPLY").all()
    pdt.assert_frame_equal(once, twice)


def test_ambiguous_legacy_overlay_preserves_scores_and_does_not_upgrade_authorization(monkeypatch) -> None:
    monkeypatch.setattr(zapi, "get_zapi_features", lambda tickers: pytest.fail("ambiguous path fetched ZAPI features"))
    monkeypatch.setattr(broker, "load_public_cache", lambda: pytest.fail("ambiguous path fetched broker cache"))
    legacy = pd.DataFrame([{
        "ticker": "TEST.JK", "smart_money_score": 72.5,
        "emir_conviction_score": 72.5, "emir_final_score": 72.5,
        "production_ready": False, "real_money_ready": False,
        "zapi_base_smart_money_score": 60.0,
        "broker_base_smart_money_score": 60.0,
    }])
    zapi_out = zapi.enrich_emir_radar(legacy)
    broker_out = broker.enrich_emir_broker(legacy)
    assert zapi_out.loc[0, "emir_conviction_score"] == 72.5
    assert broker_out.loc[0, "emir_conviction_score"] == 72.5
    assert zapi_out.loc[0, "zapi_enrichment_compatibility_state"] == "AMBIGUOUS_LEGACY_PRESERVED_NO_REAPPLY"
    assert broker_out.loc[0, "broker_enrichment_compatibility_state"] == "AMBIGUOUS_LEGACY_PRESERVED_NO_REAPPLY"
    assert not bool(zapi_out.loc[0, "production_ready"])
    assert not bool(broker_out.loc[0, "real_money_ready"])


def test_zapi_dashboard_blend_is_idempotent() -> None:
    base = pd.DataFrame([{
        "ticker": "TEST", "dashboard_flow_score": 60.0,
        "dashboard_silent_accum_score": 50.0, "distribution_score": 20.0,
        "zapi_foreign_flow_score": 100.0,
        "zapi_accumulation_confirmation_score": 90.0,
        "zapi_foreign_flow_coverage_pct": 100.0,
    }])
    once = zapi.blend_emir_dashboard_output(base)
    twice = zapi.blend_emir_dashboard_output(once)
    fields = ["dashboard_flow_score", "dashboard_silent_accum_score", "dashboard_accumulation_dominance_pct"]
    pdt.assert_frame_equal(once[fields], twice[fields])


def test_runtime_integrity_wrapper_never_reranks_valid_frozen_snapshot(monkeypatch) -> None:
    import runtime_integrity_patch as runtime_patch

    rank_calls = 0

    def rank_spy(frame: pd.DataFrame) -> pd.DataFrame:
        nonlocal rank_calls
        rank_calls += 1
        return frame.copy()

    monkeypatch.setattr(decision, "apply_three_rank_contract", rank_spy)
    module = SimpleNamespace(enrich_dashboard_scores=lambda frame: frame.copy(deep=True))
    runtime_patch._wrap_dashboard_scores(module)
    frozen = _material_frozen_snapshot()
    out = module.enrich_dashboard_scores(frozen)
    pdt.assert_frame_equal(out, frozen)
    assert rank_calls == 0


def test_runtime_integrity_wrapper_routes_unfrozen_output_through_finalizer_once(monkeypatch) -> None:
    import runtime_integrity_patch as runtime_patch

    calls = 0
    expected = _material_frozen_snapshot()

    def finalize_once(frame: pd.DataFrame) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return expected.copy(deep=True)

    monkeypatch.setattr(decision, "finalize_decision_snapshot", finalize_once)
    module = SimpleNamespace(enrich_dashboard_scores=lambda frame: frame.copy(deep=True))
    runtime_patch._wrap_dashboard_scores(module)
    out = module.enrich_dashboard_scores(_radar())
    pdt.assert_frame_equal(out, expected)
    assert calls == 1


def test_runtime_integrity_wrapper_is_repeatable_and_install_order_idempotent() -> None:
    import runtime_integrity_patch as runtime_patch

    module = SimpleNamespace(enrich_dashboard_scores=lambda frame: frame.copy(deep=True))
    runtime_patch._wrap_dashboard_scores(module)
    first_wrapper = module.enrich_dashboard_scores
    runtime_patch._wrap_dashboard_scores(module)
    assert module.enrich_dashboard_scores is first_wrapper

    frozen = _material_frozen_snapshot()
    once = module.enrich_dashboard_scores(frozen)
    twice = module.enrich_dashboard_scores(once)
    pdt.assert_frame_equal(once, twice)
    pdt.assert_frame_equal(
        once[["raw_research_rank", "guarded_decision_priority_rank", "production_real_money_rank", "production_ready"]],
        twice[["raw_research_rank", "guarded_decision_priority_rank", "production_real_money_rank", "production_ready"]],
    )


def test_runtime_integrity_and_canonical_binding_order_are_semantically_deterministic() -> None:
    import runtime_integrity_patch as runtime_patch

    frozen = _material_frozen_snapshot()
    compatibility_first = SimpleNamespace(enrich_dashboard_scores=lambda frame: frame.copy(deep=True))
    runtime_patch._wrap_dashboard_scores(compatibility_first)
    compatibility_first.enrich_dashboard_scores = decision.finalize_decision_snapshot

    canonical_first = SimpleNamespace(enrich_dashboard_scores=decision.finalize_decision_snapshot)
    runtime_patch._wrap_dashboard_scores(canonical_first)

    pdt.assert_frame_equal(
        compatibility_first.enrich_dashboard_scores(frozen),
        canonical_first.enrich_dashboard_scores(frozen),
    )
