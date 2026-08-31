from __future__ import annotations

import numpy as np
import pandas as pd

import zapi_flow_enrichment as zapi


def _history() -> pd.DataFrame:
    dates = list(reversed(zapi._expected_idx_sessions(count=20)))
    rows = []
    for ticker, direction in (("POS1", 1.0), ("POS2", 0.6), ("NEG1", -0.6), ("NEG2", -1.0)):
        for i, day in enumerate(dates):
            volume = 10_000_000.0
            net = direction * volume * (0.006 + 0.0001 * (i % 3))
            buy = 1_100_000.0 + max(net, 0.0)
            sell = 1_100_000.0 + max(-net, 0.0)
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": day,
                    "foreign_buy_shares": buy,
                    "foreign_sell_shares": sell,
                    "foreign_net_shares": net,
                    "volume": volume,
                    "source": "SYNTHETIC_ZAPI",
                    "flow_unit": "SHARES",
                }
            )
    return pd.DataFrame(rows)


def test_foreign_coverage_uses_idx_sessions_and_does_not_fill_gaps() -> None:
    as_of = pd.Timestamp("2026-08-31 17:00", tz="Asia/Jakarta")
    expected = list(reversed(zapi._expected_idx_sessions(as_of, 20)))
    rows = [
        {"ticker": "TEST", "trade_date": day, "foreign_net_shares": 1.0,
         "foreign_buy_shares": 2.0, "foreign_sell_shares": 1.0, "volume": 10.0,
         "source": "ZAPI_IDX_FOREIGN_FLOW"}
        for day in expected[1:]
    ]
    rows.append({"ticker": "TEST", "trade_date": "2026-08-25", "foreign_net_shares": 99.0,
                 "foreign_buy_shares": 100.0, "foreign_sell_shares": 1.0, "volume": 100.0,
                 "source": "ZAPI_IDX_FOREIGN_FLOW"})

    row = zapi.score_foreign_history(pd.DataFrame(rows), ["TEST"], as_of=as_of).iloc[0]

    assert int(row["foreign_expected_sessions"]) == 20
    assert int(row["foreign_observed_sessions"]) == 19
    assert float(row["foreign_coverage_ratio"]) == 0.95
    assert row["foreign_freshness_state"] == "FRESH"
    assert row["foreign_window_state"] == "PARTIAL"
    assert float(row["zapi_foreign_net_shares_20d"]) == 19.0


def test_foreign_flow_score_separates_accumulation_and_distribution() -> None:
    scored = zapi.score_foreign_history(_history())
    pos = scored.set_index("ticker").loc["POS1"]
    neg = scored.set_index("ticker").loc["NEG2"]
    assert float(pos["zapi_foreign_flow_score"]) > float(neg["zapi_foreign_flow_score"])
    assert pos["zapi_foreign_state"] == "NET_ACCUMULATION"
    assert neg["zapi_foreign_state"] == "NET_DISTRIBUTION"
    assert float(pos["zapi_foreign_flow_coverage_pct"]) >= 95.0


def test_all_missing_history_preserves_schema_with_zero_coverage() -> None:
    scored = zapi.score_foreign_history(pd.DataFrame(), ["MISS1.JK", "MISS2"])

    assert scored["ticker"].tolist() == ["MISS1", "MISS2"]
    expected_columns = {
        "zapi_foreign_latest_trade_date",
        "zapi_foreign_observed_days",
        "zapi_foreign_net_participation_1d",
        "zapi_foreign_net_participation_5d",
        "zapi_foreign_net_participation_20d",
        "zapi_foreign_flow_score",
        "zapi_foreign_flow_coverage_pct",
        "zapi_accumulation_confirmation_score",
        "zapi_smart_money_confirmation_score",
        "zapi_smc_flow_confirmation_score",
    }
    assert expected_columns.issubset(scored.columns)
    assert scored["zapi_foreign_observed_days"].eq(0).all()
    assert scored["zapi_foreign_flow_coverage_pct"].eq(0.0).all()
    assert scored["zapi_foreign_flow_score"].isna().all()
    assert scored["zapi_accumulation_confirmation_score"].isna().all()
    assert scored["zapi_smart_money_confirmation_score"].isna().all()
    assert scored["zapi_smc_flow_confirmation_score"].isna().all()
    assert scored.filter(like="zapi_foreign_net_participation_").isna().all().all()


def test_mixed_history_preserves_real_score_and_fails_soft_for_missing_ticker() -> None:
    history = _history().loc[lambda frame: frame["ticker"].eq("POS1")].copy()
    real_only = zapi.score_foreign_history(history, ["POS1"]).set_index("ticker").loc["POS1"]
    mixed = zapi.score_foreign_history(history, ["POS1", "MISS.JK"]).set_index("ticker")

    pd.testing.assert_series_equal(mixed.loc["POS1"], real_only, check_names=False)
    missing = mixed.loc["MISS"]
    assert int(missing["zapi_foreign_observed_days"]) == 0
    assert float(missing["zapi_foreign_flow_coverage_pct"]) == 0.0
    assert pd.isna(missing["zapi_foreign_flow_score"])
    assert pd.isna(missing["zapi_foreign_net_participation_20d"])


def test_emir_conviction_and_smart_money_overlay_are_bounded(monkeypatch) -> None:
    features = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "zapi_foreign_flow_score": 100.0,
                "zapi_foreign_flow_coverage_pct": 100.0,
                "zapi_accumulation_confirmation_score": 90.0,
                "zapi_smart_money_confirmation_score": 90.0,
                "zapi_smc_flow_confirmation_score": 100.0,
                "zapi_foreign_state": "NET_ACCUMULATION",
            }
        ]
    )
    monkeypatch.setattr(zapi, "get_zapi_features", lambda tickers: (features.copy(), {"state": "TEST"}))
    radar = pd.DataFrame(
        [
            {
                "ticker": "TEST.JK",
                "smart_money_score": 60.0,
                "smart_money_coverage_pct": 80.0,
                "emir_conviction_score": 70.0,
            }
        ]
    )
    out = zapi.enrich_emir_radar(radar).iloc[0]
    assert np.isclose(float(out["smart_money_score"]), 69.0)
    assert np.isclose(float(out["zapi_smart_money_confirmation_weight_pct"]), 30.0)
    assert np.isclose(float(out["zapi_emir_conviction_delta"]), 2.5)
    assert np.isclose(float(out["emir_conviction_score"]), 72.5)


def test_emir_dashboard_flow_blend_is_bounded() -> None:
    frame = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "dashboard_flow_score": 60.0,
                "dashboard_silent_accum_score": 50.0,
                "distribution_score": 20.0,
                "zapi_foreign_flow_score": 100.0,
                "zapi_accumulation_confirmation_score": 90.0,
                "zapi_foreign_flow_coverage_pct": 100.0,
            }
        ]
    )
    out = zapi.blend_emir_dashboard_output(frame).iloc[0]
    assert np.isclose(float(out["dashboard_flow_score"]), 72.0)
    assert np.isclose(float(out["dashboard_silent_accum_score"]), 58.0)
    assert 58.0 <= float(out["dashboard_accumulation_dominance_pct"]) <= 80.0
