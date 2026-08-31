from __future__ import annotations

import pandas as pd

import idx_public_participant_provider as provider
import public_idx_broker_flow as broker


def _history() -> pd.DataFrame:
    rows = []
    for day in pd.bdate_range("2026-07-20", periods=20):
        for rank, value in enumerate((100, 80, 60)):
            rows.append({
                "trade_date": day, "ticker": "TEST", "broker_code": ["AB", "CD", "EF"][rank],
                "net_value": value * 1_000_000, "net_volume": value * 1_000, "side": "TOP_NET_BUYER",
                "net_rank": rank + 1, "buy_value": value * 1_000_000, "sell_value": 0,
                "buy_volume": value * 1_000, "sell_volume": 0, "buy_avg": 1000, "sell_avg": None,
                "gross_value": value * 1_000_000,
            })
    return pd.DataFrame(rows)


def test_score_is_bounded_and_identifies_persistent_participant():
    scored = broker.score_broker_history(_history(), ["TEST"])
    row = scored.iloc[0]
    assert 0 <= float(row["broker_accumulation_score"]) <= 100
    assert row["broker_top_buyer_code"] == "AB"
    assert row["broker_accumulation_state"] == "PARTICIPANT_ACCUMULATION"


def test_consumer_is_fail_soft_without_owned_cache(monkeypatch):
    empty_schema = pd.DataFrame(columns=["trade_date", "ticker", "broker_code", "side", "net_value", "net_rank"])
    monkeypatch.setattr(broker, "load_public_cache", lambda: empty_schema)
    frame = pd.DataFrame([{ "ticker": "TEST", "smart_money_score": 60.0 }])
    out = broker.enrich_emir_broker(frame)
    assert float(out.iloc[0]["smart_money_score"]) == 60.0


def test_emir_owned_provider_reconstructs_participant_flow(tmp_path):
    path = tmp_path / "trade.csv"
    pd.DataFrame([
        {"asset": "TEST", "participant_buy": "AB", "participant_sell": "CD", "volume": 1000, "value": 1_000_000, "tradingdate": "2026-08-14"},
        {"asset": "TEST", "participant_buy": "AB", "participant_sell": "EF", "volume": 500, "value": 600_000, "tradingdate": "2026-08-14"},
        {"asset": "TEST", "participant_buy": "CD", "participant_sell": "AB", "volume": 200, "value": 220_000, "tradingdate": "2026-08-14"},
    ]).to_csv(path, sep="|", index=False)
    out = provider.aggregate_trade_detail(path, pd.Timestamp("2026-08-14").date(), ["TEST"])
    ab = out.loc[out["broker_code"].eq("AB")].iloc[0]
    assert float(ab["buy_value"]) == 1_600_000
    assert float(ab["sell_value"]) == 220_000
    assert float(ab["net_value"]) == 1_380_000
    assert ab["provenance_state"] == provider.PROVENANCE
