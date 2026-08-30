from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

import final_decision as decision
import public_idx_broker_flow as broker
import zapi_flow_enrichment as zapi


def _canon(value: object) -> str:
    text = str(value).strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _install_provider_fixtures(monkeypatch) -> dict[str, list[str]]:
    calls: dict[str, list[str]] = {"zapi": [], "broker": []}

    def zapi_features(tickers):
        calls["zapi"].extend(list(tickers))
        return pd.DataFrame([
            {
                "ticker": _canon(ticker),
                "zapi_foreign_flow_score": 100.0,
                "zapi_foreign_flow_coverage_pct": 100.0,
                "zapi_accumulation_confirmation_score": 90.0,
                "zapi_smart_money_confirmation_score": 90.0,
            }
            for ticker in tickers
        ]), {"state": "TEST"}

    def broker_features(history, universe):
        calls["broker"].extend(list(universe))
        return pd.DataFrame([
            {
                "ticker": _canon(ticker),
                "broker_smart_money_confirmation_score": 90.0,
                "broker_flow_coverage_pct": 100.0,
            }
            for ticker in universe
        ])

    monkeypatch.setattr(zapi, "get_zapi_features", zapi_features)
    monkeypatch.setattr(broker, "load_public_cache", lambda: pd.DataFrame([{"ticker": "TEST"}]))
    monkeypatch.setattr(broker, "score_broker_history", broker_features)
    return calls


def _base(ticker: str, score: float = 72.5) -> dict[str, object]:
    return {
        "ticker": ticker,
        "smart_money_score": 60.0,
        "smart_money_coverage_pct": 80.0,
        "emir_conviction_score": score,
        "emir_final_score": score,
        "emir_decision_state": "WAIT",
        "production_ready": False,
        "production_authorized": False,
        "production_authorization_pass": False,
        "real_money_authorization_pass": False,
        "execution_authorized": False,
        "real_money_ready": False,
        "real_money_candidate": False,
        "real_money_entry_candidate": False,
        "future_direct_forward_authorization_eligible": False,
    }


def _zapi_mixed_frame() -> pd.DataFrame:
    return pd.DataFrame([
        dict(_base("PRISTINE.JK"), zapi_enrichment_compatibility_state="PRISTINE_BASE"),
        dict(
            _base("CANONICAL.JK", 75.0),
            zapi_base_smart_money_score=60.0,
            zapi_base_smart_money_coverage_pct=80.0,
            zapi_base_emir_conviction_score=72.5,
            zapi_emir_conviction_delta=2.5,
            zapi_enrichment_compatibility_state="CANONICAL_ENRICHED",
        ),
        dict(
            _base("LEGACY.JK"),
            zapi_emir_conviction_delta=2.5,
            zapi_enrichment_compatibility_state="LEGACY_ALREADY_ENRICHED",
        ),
        dict(
            _base("AMBIGUOUS.JK"),
            zapi_base_smart_money_score=60.0,
            zapi_enrichment_compatibility_state="AMBIGUOUS_LEGACY",
        ),
    ])


def _broker_mixed_frame() -> pd.DataFrame:
    return pd.DataFrame([
        dict(_base("PRISTINE.JK"), broker_enrichment_compatibility_state="PRISTINE_BASE"),
        dict(
            _base("CANONICAL.JK", 73.7),
            broker_base_smart_money_score=60.0,
            broker_base_emir_conviction_score=72.5,
            broker_base_emir_final_score=72.5,
            broker_emir_conviction_delta=1.2,
            broker_enrichment_compatibility_state="CANONICAL_ENRICHED",
        ),
        dict(
            _base("LEGACY.JK"),
            broker_emir_conviction_delta=1.2,
            broker_enrichment_compatibility_state="LEGACY_ALREADY_ENRICHED",
        ),
        dict(
            _base("AMBIGUOUS.JK"),
            broker_base_smart_money_score=60.0,
            broker_enrichment_compatibility_state="AMBIGUOUS_LEGACY",
        ),
    ])


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("ticker").sort_index()


def test_zapi_mixed_rows_are_isolated_safe_and_idempotent(monkeypatch) -> None:
    calls = _install_provider_fixtures(monkeypatch)
    mixed = _zapi_mixed_frame()
    once = zapi.enrich_emir_radar(mixed)
    twice = zapi.enrich_emir_radar(once)
    rows = _indexed(once)

    assert rows.at["PRISTINE.JK", "emir_conviction_score"] == 75.0
    assert rows.at["CANONICAL.JK", "emir_conviction_score"] == 75.0
    assert rows.at["LEGACY.JK", "emir_conviction_score"] == 72.5
    assert rows.at["AMBIGUOUS.JK", "emir_conviction_score"] == 72.5
    assert rows.at["LEGACY.JK", "zapi_enrichment_compatibility_state"] == "LEGACY_ALREADY_ENRICHED_PRESERVED_NO_REAPPLY"
    assert rows.at["AMBIGUOUS.JK", "zapi_enrichment_compatibility_state"] == "AMBIGUOUS_LEGACY_PRESERVED_NO_REAPPLY"
    assert not rows.loc[["LEGACY.JK", "AMBIGUOUS.JK"], ["production_ready", "production_authorized", "production_authorization_pass", "real_money_authorization_pass", "execution_authorized", "real_money_ready", "future_direct_forward_authorization_eligible"]].any().any()
    pdt.assert_frame_equal(
        _indexed(once)[["smart_money_score", "emir_conviction_score", "emir_final_score", "production_ready", "real_money_ready"]],
        _indexed(twice)[["smart_money_score", "emir_conviction_score", "emir_final_score", "production_ready", "real_money_ready"]],
    )
    assert zapi.classify_zapi_enrichment_state(mixed) == "AMBIGUOUS_LEGACY"
    assert calls["zapi"] == ["PRISTINE.JK"]


def test_broker_mixed_rows_are_isolated_safe_and_idempotent(monkeypatch) -> None:
    calls = _install_provider_fixtures(monkeypatch)
    mixed = _broker_mixed_frame()
    once = broker.enrich_emir_broker(mixed)
    twice = broker.enrich_emir_broker(once)
    rows = _indexed(once)

    assert rows.at["PRISTINE.JK", "emir_conviction_score"] == 73.7
    assert rows.at["CANONICAL.JK", "emir_conviction_score"] == 73.7
    assert rows.at["LEGACY.JK", "emir_conviction_score"] == 72.5
    assert rows.at["AMBIGUOUS.JK", "emir_conviction_score"] == 72.5
    assert rows.at["LEGACY.JK", "broker_enrichment_compatibility_state"] == "LEGACY_ALREADY_ENRICHED_PRESERVED_NO_REAPPLY"
    assert rows.at["AMBIGUOUS.JK", "broker_enrichment_compatibility_state"] == "AMBIGUOUS_LEGACY_PRESERVED_NO_REAPPLY"
    assert not rows.loc[["LEGACY.JK", "AMBIGUOUS.JK"], ["production_ready", "production_authorized", "production_authorization_pass", "real_money_authorization_pass", "execution_authorized", "real_money_ready", "future_direct_forward_authorization_eligible"]].any().any()
    pdt.assert_frame_equal(
        _indexed(once)[["smart_money_score", "emir_conviction_score", "emir_final_score", "production_ready", "real_money_ready"]],
        _indexed(twice)[["smart_money_score", "emir_conviction_score", "emir_final_score", "production_ready", "real_money_ready"]],
    )
    assert broker.classify_broker_enrichment_state(mixed) == "AMBIGUOUS_LEGACY"
    assert calls["broker"] == ["PRISTINE.JK"]


def test_row_local_null_and_zero_markers_do_not_cross_contaminate(monkeypatch) -> None:
    _install_provider_fixtures(monkeypatch)
    values = [2.5, None, np.nan, pd.NA, "", False, 0]
    tickers = ["POPULATED.JK", "NONE.JK", "NAN.JK", "PDNA.JK", "EMPTY.JK", "FALSE.JK", "ZERO.JK"]

    zapi_frame = pd.DataFrame([dict(_base(ticker), zapi_emir_conviction_delta=value) for ticker, value in zip(tickers, values)])
    zapi_rows = _indexed(zapi.enrich_emir_radar(zapi_frame))
    assert zapi_rows.at["POPULATED.JK", "emir_conviction_score"] == 72.5
    assert zapi_rows.at["ZERO.JK", "emir_conviction_score"] == 72.5
    for ticker in ("NONE.JK", "NAN.JK", "PDNA.JK", "EMPTY.JK", "FALSE.JK"):
        assert zapi_rows.at[ticker, "emir_conviction_score"] == 75.0

    broker_frame = pd.DataFrame([dict(_base(ticker), broker_emir_conviction_delta=value) for ticker, value in zip(tickers, values)])
    broker_rows = _indexed(broker.enrich_emir_broker(broker_frame))
    assert broker_rows.at["POPULATED.JK", "emir_conviction_score"] == 72.5
    assert broker_rows.at["ZERO.JK", "emir_conviction_score"] == 72.5
    for ticker in ("NONE.JK", "NAN.JK", "PDNA.JK", "EMPTY.JK", "FALSE.JK"):
        assert broker_rows.at[ticker, "emir_conviction_score"] == 73.7


def _composed_mixed_frame() -> pd.DataFrame:
    return pd.DataFrame([
        dict(
            _base("ZAPI_PRISTINE_BROKER_LEGACY.JK"),
            zapi_enrichment_compatibility_state="PRISTINE_BASE",
            broker_emir_conviction_delta=1.2,
            broker_enrichment_compatibility_state="LEGACY_ALREADY_ENRICHED",
        ),
        dict(
            _base("ZAPI_LEGACY_BROKER_CANONICAL.JK", 73.7),
            zapi_emir_conviction_delta=2.5,
            zapi_enrichment_compatibility_state="LEGACY_ALREADY_ENRICHED",
            broker_base_smart_money_score=60.0,
            broker_base_emir_conviction_score=72.5,
            broker_base_emir_final_score=72.5,
            broker_emir_conviction_delta=1.2,
            broker_enrichment_compatibility_state="CANONICAL_ENRICHED",
        ),
        dict(
            _base("ZAPI_AMBIGUOUS_BROKER_PRISTINE.JK"),
            zapi_base_smart_money_score=60.0,
            zapi_enrichment_compatibility_state="AMBIGUOUS_LEGACY",
            broker_enrichment_compatibility_state="PRISTINE_BASE",
        ),
        dict(
            _base("ZAPI_CANONICAL_BROKER_AMBIGUOUS.JK", 75.0),
            zapi_base_smart_money_score=60.0,
            zapi_base_smart_money_coverage_pct=80.0,
            zapi_base_emir_conviction_score=72.5,
            zapi_emir_conviction_delta=2.5,
            zapi_enrichment_compatibility_state="CANONICAL_ENRICHED",
            broker_base_smart_money_score=60.0,
            broker_enrichment_compatibility_state="AMBIGUOUS_LEGACY",
        ),
    ])


def test_composed_mixed_provider_lineage_isolated_and_repeatable(monkeypatch) -> None:
    _install_provider_fixtures(monkeypatch)
    mixed = _composed_mixed_frame()
    once = broker.enrich_emir_broker(zapi.enrich_emir_radar(mixed))
    twice = broker.enrich_emir_broker(zapi.enrich_emir_radar(once))
    rows = _indexed(once)

    assert rows.at["ZAPI_PRISTINE_BROKER_LEGACY.JK", "emir_conviction_score"] == 75.0
    assert rows.at["ZAPI_LEGACY_BROKER_CANONICAL.JK", "emir_conviction_score"] == 73.7
    assert rows.at["ZAPI_AMBIGUOUS_BROKER_PRISTINE.JK", "emir_conviction_score"] == 73.7
    assert rows.at["ZAPI_CANONICAL_BROKER_AMBIGUOUS.JK", "emir_conviction_score"] == 75.0
    assert rows.at["ZAPI_PRISTINE_BROKER_LEGACY.JK", "broker_enrichment_compatibility_state"].startswith("LEGACY_ALREADY_ENRICHED")
    assert rows.at["ZAPI_AMBIGUOUS_BROKER_PRISTINE.JK", "zapi_enrichment_compatibility_state"].startswith("AMBIGUOUS_LEGACY")
    pdt.assert_frame_equal(
        _indexed(once)[["smart_money_score", "emir_conviction_score", "emir_final_score", "production_ready", "real_money_ready"]],
        _indexed(twice)[["smart_money_score", "emir_conviction_score", "emir_final_score", "production_ready", "real_money_ready"]],
    )


def test_composed_mixed_finalization_remains_fail_closed(monkeypatch) -> None:
    _install_provider_fixtures(monkeypatch)
    monkeypatch.setattr(decision.top3_dashboard, "_canonical_enrich_dashboard_scores", lambda frame, frames=None: frame.copy())
    monkeypatch.setattr(decision.zapi_flow_enrichment, "blend_emir_dashboard_output", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_recompute_real_money", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_adjust_cost_confidence", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_apply_foreign_shock_guard", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "_apply_gate_audit", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "refine_emir_proxy_authorization_tier", lambda frame: frame.copy())
    monkeypatch.setattr(decision, "enrich_emir_shadow", lambda frame: frame.copy())

    once = broker.enrich_emir_broker(zapi.enrich_emir_radar(_composed_mixed_frame()))
    finalized = decision.finalize_decision_snapshot(once)
    protected = finalized[finalized["ticker"].isin([
        "ZAPI_PRISTINE_BROKER_LEGACY.JK",
        "ZAPI_LEGACY_BROKER_CANONICAL.JK",
        "ZAPI_AMBIGUOUS_BROKER_PRISTINE.JK",
        "ZAPI_CANONICAL_BROKER_AMBIGUOUS.JK",
    ])]
    assert not protected["production_ready"].any()
    assert not protected["real_money_ready"].any()
    assert protected["production_real_money_rank"].isna().all()
    assert decision.is_final_decision_snapshot(finalized) is True
