import pandas as pd
from autonomous_enrichment import reconcile_fundamental_snapshot
from narrative_flow_engine import normalize_ticker


def _record_map(frame: pd.DataFrame):
    out = {}
    for _, row in frame.iterrows():
        ticker = normalize_ticker(row.get("ticker"))
        payload = row.to_dict(); payload["ticker"] = ticker
        out[ticker] = payload
    return out


def test_proxy_fundamental_reconciliation_preserves_ticker_join_key():
    frame = pd.DataFrame([
        {"ticker":"OMED.JK","fundamental_conversion_score":72.0,"fundamental_coverage_pct":83.5,"fundamental_data_quality_score":80.5,"fundamental_state":"FUTURE_FUNDAMENTAL_SUPPORTIVE"},
        {"ticker":"ELSA.JK","fundamental_conversion_score":72.0,"fundamental_coverage_pct":83.5,"fundamental_data_quality_score":80.5,"fundamental_state":"FUTURE_FUNDAMENTAL_SUPPORTIVE"},
        {"ticker":"MARK.JK","fundamental_conversion_score":72.0,"fundamental_coverage_pct":83.5,"fundamental_data_quality_score":80.5,"fundamental_state":"FUTURE_FUNDAMENTAL_SUPPORTIVE"},
    ])
    mapping = _record_map(frame)
    reconciled = [reconcile_fundamental_snapshot(mapping[t], None, now="2026-08-09") for t in ["OMED.JK","ELSA.JK","MARK.JK"]]
    out = pd.DataFrame(reconciled).set_index("ticker")
    assert list(out.index) == ["OMED.JK","ELSA.JK","MARK.JK"]
    assert out["fundamental_conversion_score"].notna().all()
    assert (out["fundamental_coverage_pct"] >= 35).all()
    assert (out["fundamental_data_quality_score"] >= 35).all()


def test_pandas_set_index_pattern_that_caused_v19_regression_is_guarded():
    frame = pd.DataFrame([{"ticker":"OMED.JK","fundamental_conversion_score":72.0}])
    broken = frame.set_index("ticker").to_dict(orient="index")["OMED.JK"]
    assert "ticker" not in broken  # documents the original failure mode
    fixed = _record_map(frame)["OMED.JK"]
    assert fixed["ticker"] == "OMED.JK"
